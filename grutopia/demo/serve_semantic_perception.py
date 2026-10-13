"""Serve local GroundingDINO, MobileSAM, or Qwen3-VL for semantic exploration."""

import argparse
import base64
import json
import os
import re

import cv2
import numpy as np
from flask import Flask, jsonify, request
from grutopia_extension.interactive_navigation.goal_matching import (
    description_category_matches, goal_verification_prompt, parse_goal_score,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('service', choices=('grounding-dino', 'mobile-sam', 'qwen-vl'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--device', default='cuda:5')
    parser.add_argument('--grounding-dino-model', default='IDEA-Research/grounding-dino-base')
    parser.add_argument(
        '--mobile-sam-checkpoint',
        default='grutopia/assets/models/mobile_sam.pt',
    )
    parser.add_argument('--box-threshold', type=float, default=0.30)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--qwen-vl-model', default=os.environ.get('QWEN_VL_MODEL', 'Qwen/Qwen3-VL-8B-Instruct'),
                        help='Local model directory or already cached model ID; no automatic download.')
    parser.add_argument('--vl-max-new-tokens', type=int, default=32)
    parser.add_argument('--vl-max-pixels', type=int, default=384 * 384)
    return parser.parse_args()


def _decode_image(value: str) -> np.ndarray:
    encoded = base64.b64decode(value)
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('invalid encoded image')
    # The client encodes an RGB ndarray through OpenCV, so the decoded channel
    # values already retain the original RGB ordering.
    return image


def _grounding_dino_app(args):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.grounding_dino_model, local_files_only=True)
    # GroundingDINO's text/vision mixer is not fully fp16-safe in transformers 4.49.
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.grounding_dino_model,
        local_files_only=True,
    ).to(args.device)
    model.eval()
    app = Flask('grounding-dino')

    @app.get('/health')
    def health():
        return jsonify({
            'ready': True,
            'service': 'grounding-dino',
            'model': args.grounding_dino_model,
            'box_threshold': args.box_threshold,
            'text_threshold': args.text_threshold,
        })

    @app.post('/gdino')
    def detect():
        payload = request.get_json(force=True)
        image = _decode_image(payload['image'])
        caption = str(payload.get('caption', '')).strip()
        inputs = processor(images=image, text=caption, return_tensors='pt')
        model_dtype = next(model.parameters()).dtype
        moved = {}
        for name, value in inputs.items():
            value = value.to(args.device)
            if torch.is_floating_point(value):
                value = value.to(dtype=model_dtype)
            moved[name] = value
        inputs = moved
        with torch.inference_mode():
            outputs = model(**inputs)
        post_process = processor.post_process_grounded_object_detection
        kwargs = {
            'outputs': outputs,
            'input_ids': inputs['input_ids'],
            'text_threshold': args.text_threshold,
            'target_sizes': [image.shape[:2]],
        }
        try:
            result = post_process(**kwargs, threshold=args.box_threshold)[0]
        except TypeError:
            result = post_process(**kwargs, box_threshold=args.box_threshold)[0]
        labels = result.get('text_labels', result.get('labels', []))
        return jsonify(
            {
                'boxes': result['boxes'].detach().cpu().tolist(),
                'logits': result['scores'].detach().cpu().tolist(),
                'phrases': [str(label) for label in labels],
            }
        )

    return app


def _mobile_sam_app(args):
    import torch
    from mobile_sam import SamPredictor, sam_model_registry

    model = sam_model_registry['vit_t'](checkpoint=args.mobile_sam_checkpoint)
    model.to(device=args.device)
    model.eval()
    predictor = SamPredictor(model)
    app = Flask('mobile-sam')

    @app.get('/health')
    def health():
        return jsonify({
            'ready': True,
            'service': 'mobile-sam',
            'checkpoint': args.mobile_sam_checkpoint,
        })

    @app.post('/mobile_sam')
    def segment():
        payload = request.get_json(force=True)
        image = _decode_image(payload['image'])
        bbox = np.asarray(payload['bbox'], dtype=np.float32)
        with torch.inference_mode():
            predictor.set_image(image)
            masks, _, _ = predictor.predict(box=bbox, multimask_output=False)
        mask = np.asarray(masks[0], dtype=np.uint8)
        return jsonify({'cropped_mask': base64.b64encode(mask.tobytes()).decode('utf-8')})

    return app


def _qwen_vl_app(args):
    """Independent VL service: receives one marked context crop, returns one category."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.vl_max_new_tokens < 8 or args.vl_max_pixels < 64 * 64:
        raise ValueError('VL output budget must be >=8 tokens and image budget >=4096 pixels')
    processor = AutoProcessor.from_pretrained(args.qwen_vl_model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen_vl_model, dtype='auto', local_files_only=True,
    ).to(args.device)
    model.eval()
    app = Flask('qwen-vl')
    app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

    def generate(images, prompt, budget):
        messages = [{'role': 'user', 'content': [
            *({'type': 'image', 'image': image} for image in images),
            {'type': 'text', 'text': prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors='pt',
                           min_pixels=64 * 64, max_pixels=args.vl_max_pixels).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=budget, do_sample=False)
        return processor.batch_decode(
            generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def parse_output(raw_text, labels):
        cleaned = raw_text
        if cleaned.startswith('```') and cleaned.endswith('```'):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)[:-3].strip()
        result = json.loads(cleaned)
        if not isinstance(result, dict) or result.get('label') not in labels:
            raise ValueError('label is outside requested categories')
        return result

    @app.get('/health')
    def health():
        return jsonify({'ready': True, 'service': 'qwen-vl', 'model': args.qwen_vl_model,
                        'goal_verification': True,
                        'goal_verification_mode': 'category_then_goal',
                        'multi_view': True, 'max_reference_images': 2,
                        'max_pixels': args.vl_max_pixels, 'max_new_tokens': args.vl_max_new_tokens})

    @app.post('/classify')
    def classify():
        payload = request.get_json(force=True)
        target_query = payload.get('target_query')
        verify_goal = payload.get('verify_goal', True)
        if not isinstance(verify_goal, bool):
            return jsonify({'error': 'verify_goal must be boolean'}), 400
        if target_query is not None and (not isinstance(target_query, str)
                                        or not target_query.strip() or len(target_query) > 512):
            return jsonify({'error': 'target_query must be nonempty text of at most 512 characters'}), 400
        labels = payload.get('labels')
        if (not isinstance(labels, list) or not 1 <= len(labels) <= 64
                or any(not isinstance(label, str) or not re.fullmatch(r'[a-z][a-z0-9 -]{0,63}', label)
                       for label in labels) or 'unknown' not in labels):
            return jsonify({'error': 'labels must contain up to 64 short English categories and unknown'}), 400
        if 'image' not in payload:
            return jsonify({'error': 'image is required'}), 400
        references = payload.get('reference_images', [])
        if (not isinstance(references, list) or len(references) > 2 or
                any(not isinstance(value, str) or not value for value in references)):
            return jsonify({'error': 'reference_images must contain at most two encoded images'}), 400
        images = []
        try:
            for encoded in [payload['image'], *references]:
                image = Image.fromarray(_decode_image(encoded))
                scale = min(1.0, (args.vl_max_pixels / (image.width * image.height)) ** 0.5)
                if scale < 1:
                    image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
                images.append(image)
        except (ValueError, TypeError, cv2.error):
            return jsonify({'error': 'invalid encoded image'}), 400
        prompt = (
            'Classify the main object inside the red rectangle. The surrounding real background '
            'is context only. Ignore the red outline itself. Do not classify a different background object. '
            'A television, computer monitor, picture, mirror and window are distinct categories. '
            'If the object is ambiguous, too small, not visible, or none of the listed categories fits, '
            'choose unknown. Do not guess a television simply because an object is rectangular. '
            'Choose exactly one category from: ' + ', '.join(labels) + '. '
        )
        if target_query is not None:
            prompt += 'Distinguish a bed and mattress from a sofa, dining chair or table. '
        if references:
            prompt += ('The first image is the current view. The other images are earlier views of the '
                       'same tracked object. Classify that object using all views. If the views appear '
                       'to show different objects, choose unknown. Do not combine contradictory attributes. ')
        category_prompt = prompt + 'Return ONLY a JSON object with one key, for example {"label":"chair"}.'
        # The navigation query is deliberately absent from the first inference.
        category_text = raw_text = generate(images, category_prompt, args.vl_max_new_tokens)
        reason = 'classified'
        goal_score = None
        verification_status = 'not_requested'
        try:
            label = parse_output(category_text, labels)['label']
            if target_query is not None:
                verification_status = 'abstained'
                if not verify_goal:
                    verification_status = 'prefiltered'
                elif label != 'unknown' and description_category_matches(target_query, label, labels):
                    raw_text = generate(images, prompt +
                        'The query-independent classification was ' + json.dumps(label) +
                        '. Evaluate that SAME object. Do not change its category to fit the goal. ' +
                        goal_verification_prompt(target_query), max(128, args.vl_max_new_tokens))
                    result = parse_output(raw_text, labels)
                    observed = result.get('observed_object')
                    if result['label'] != label or (isinstance(observed, str) and
                            not description_category_matches(observed, label, labels)):
                        reason = verification_status = 'category_conflict'
                    elif isinstance(observed, str) and observed.strip():
                        goal_score = parse_goal_score(result.get('goal_match_score'))
                        verification_status = 'verified' if goal_score is not None else 'abstained'
                elif label != 'unknown':
                    reason = verification_status = 'category_mismatch'
        except (ValueError, TypeError):
            label, reason = 'unknown', 'invalid_output'
            verification_status = 'invalid_output' if target_query is not None else 'not_requested'
        return jsonify({'label': label, 'model': args.qwen_vl_model,
                        'goal_match_score': goal_score, 'goal_verification_status': verification_status,
                        'category_raw_text': category_text,
                        'goal_verification_mode': 'category_then_goal',
                        'view_count': len(images),
                        'raw_text': raw_text, 'reason': reason})

    return app


def main():
    args = parse_args()
    if args.port is None:
        args.port = {'grounding-dino': 12181, 'mobile-sam': 12183, 'qwen-vl': 12185}[args.service]
    factories = {'grounding-dino': _grounding_dino_app, 'mobile-sam': _mobile_sam_app,
                 'qwen-vl': _qwen_vl_app}
    app = factories[args.service](args)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == '__main__':
    main()
