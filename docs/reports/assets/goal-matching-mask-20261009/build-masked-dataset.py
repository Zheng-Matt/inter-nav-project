"""Rebuild the small human-annotated dataset with the existing MobileSAM service."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    AgentVLMBackend,
    OpenVocabularyPerception,
    OpenVocabularyPerceptionConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record_dir', type=Path)
    parser.add_argument('--raw-dir', type=Path, default=Path(
        'tests/fixtures/goal_matching/foreground_real_fridge/raw_rgb'))
    parser.add_argument('--mobile-sam-url', default='http://127.0.0.1:12188/mobile_sam')
    args = parser.parse_args()
    destination = args.record_dir / 'masked-dataset'
    destination.mkdir(exist_ok=False)
    source = Path('tests/fixtures/goal_matching/clean_real_fridge')
    config = OpenVocabularyPerceptionConfig(
        semantic_classifier='qwen-vl', mobile_sam_url=args.mobile_sam_url)
    backend = AgentVLMBackend(config)
    perception = OpenVocabularyPerception(config)
    records, done = [], set()
    for name in ('development', 'heldout'):
        dataset = json.loads((source / (name + '.json')).read_text())
        for item in dataset['provenance']['sources']:
            crop = item['crop']
            if crop in done:
                continue
            done.add(crop)
            path = args.raw_dir / f"frame_{item['step']:06d}.png"
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f'Cannot read raw RGB: {path}')
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            bbox = tuple(item['bbox_xyxy'])
            mask = backend.segment(image, bbox)
            focused_crop = perception._qwen_context_crop(image, bbox, mask=mask)
            context_crop = perception._qwen_context_crop(image, bbox)
            cv2.imwrite(str(destination / crop), cv2.cvtColor(context_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(destination / ('masked_' + crop)),
                        cv2.cvtColor(focused_crop, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(destination / ('mask_' + crop)), mask.astype('uint8') * 255)
            records.append(dict(
                crop=crop, step=item['step'], bbox_xyxy=list(bbox),
                source_rgb_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                mask_pixels=int(mask.sum()), mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                segmenter='existing MobileSAM vit_t; independent local service',
                outside_mask_rgb=[127, 127, 127]))
        for case in dataset['cases']:
            for candidate in case['candidates']:
                candidate['masked_crop'] = 'masked_' + candidate['crop']
        dataset['provenance']['mask_processing'] = (
            'Existing SAM mask; pixels outside mask replaced with neutral RGB 127; '
            'default context dimensions and red box unchanged. '
            'No manually drawn masks or simulator labels.')
        (destination / (name + '.json')).write_text(json.dumps(dataset, indent=2) + '\n')
    (destination / 'mask-provenance.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(dict(crops=len(records), output=str(destination))))


if __name__ == '__main__':
    main()
