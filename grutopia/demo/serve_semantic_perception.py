"""Serve local GroundingDINO or MobileSAM for semantic exploration."""

import argparse
import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('service', choices=('grounding-dino', 'mobile-sam'))
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
        return jsonify({'ready': True, 'service': 'grounding-dino'})

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
        return jsonify({'ready': True, 'service': 'mobile-sam'})

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


def main():
    args = parse_args()
    if args.port is None:
        args.port = 12181 if args.service == 'grounding-dino' else 12183
    app = _grounding_dino_app(args) if args.service == 'grounding-dino' else _mobile_sam_app(args)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == '__main__':
    main()
