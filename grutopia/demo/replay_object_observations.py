"""Replay aligned sensor captures through perception and mapping without Isaac Sim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--record-dir', required=True)
    parser.add_argument('--baseline-root', help='Import the frozen earlier checkout instead of current code.')
    parser.add_argument('--qwen-vl-url', default='http://localhost:12185/classify')
    parser.add_argument('--grounding-dino-url', default='http://localhost:12181/gdino')
    parser.add_argument('--mobile-sam-url', default='http://localhost:12183/mobile_sam')
    parser.add_argument('--multi-view', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--geometry-filter', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.baseline_root).resolve() if args.baseline_root else Path(__file__).resolve().parents[2]))
    import numpy as np
    from grutopia_extension.interactive_navigation.mapping import MappingConfig, SceneGraphMap
    from grutopia_extension.interactive_navigation.mapping_runtime import _camera_cloud
    from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
        OpenVocabularyPerception, OpenVocabularyPerceptionConfig,
    )
    samples = sorted(Path(args.inputs).glob('frame_*.npz'))
    if not samples:
        raise ValueError('No aligned sensor captures were found')
    output = Path(args.record_dir)
    output.mkdir(parents=True, exist_ok=False)
    options = dict(semantic_classifier='qwen-vl', query_interval=0,
                   qwen_vl_url=args.qwen_vl_url, grounding_dino_url=args.grounding_dino_url,
                   mobile_sam_url=args.mobile_sam_url)
    optional = {'multi_view': args.multi_view, 'geometry_filter': args.geometry_filter}
    options.update({k:v for k,v in optional.items() if k in OpenVocabularyPerceptionConfig.__dataclass_fields__})
    config = OpenVocabularyPerceptionConfig(**options)
    pipeline = OpenVocabularyPerception(config)
    pipeline.check_ready(load_clip=False)
    graph = SceneGraphMap(MappingConfig())
    records = []
    from dataclasses import asdict
    with (output/'frames.jsonl').open('x') as file:
        for sample in samples:
            with np.load(sample, allow_pickle=False) as data:
                camera = {'rgba':data['rgba'], 'depth':data['depth'], 'pointcloud':data['pointcloud'],
                          'position':data['camera_position']}
                step = int(data['capture_step'])
            _, _, pixels = _camera_cloud(camera)
            started = time.monotonic()
            extra = ({'camera_position': camera['position'], 'capture_step':step}
                     if 'multi_view' in config.__dataclass_fields__ else {})
            detections = pipeline.perceive(camera['rgba'], camera['depth'], pixels, args.target, step=step, **extra)
            graph.update_detections(pipeline.as_semantic_detections(detections))
            record = {'capture_step':step, 'input':str(sample.resolve()),
                      'input_sha256':hashlib.sha256(sample.read_bytes()).hexdigest(),
                      'seconds':time.monotonic()-started, 'debug':pipeline.last_debug_frame}
            file.write(json.dumps(record)+'\n');file.flush()
            records.append(record)
    summary = {'scope':'Recorded sensor replay. No navigation, occupancy updates or accuracy labels.',
               'target':args.target,'config':asdict(config),'frames':len(records),
               'mean_recognition_seconds':float(np.mean([r['seconds'] for r in records])),
               'objects':[asdict(node) for node in graph.object_nodes()]}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'frames':len(records),'objects':len(graph.object_nodes()),'output':str(output)}))


if __name__ == '__main__':
    main()
