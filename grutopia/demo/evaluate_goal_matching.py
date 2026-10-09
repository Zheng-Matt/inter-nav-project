"""Compare frozen goal matchers on contracts or human-annotated Qwen crops.

This measures object matching only. A one-observation configuration isolates
the matcher; it never measures navigation success or temporal confirmation.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=root / 'tests/fixtures/goal_matching/contracts.json')
    parser.add_argument(
        '--baseline-root', type=Path, help='Clean original checkout; legacy worker imports it directly.'
    )
    parser.add_argument('--record-dir', type=Path)
    parser.add_argument('--live-qwen-url', help='Optional /classify URL; requires annotated red-box crops.')
    parser.add_argument('--qwen-vl-crop-mode', choices=('context', 'masked'), default='context',
                        help='Robust live evaluation can use candidate masked_crop; legacy keeps crop.')
    parser.add_argument('--description-threshold', type=float, default=0.80)
    parser.add_argument('--matching-repeats', type=int, default=100)
    parser.add_argument('--worker', choices=('legacy', 'robust'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.matching_repeats < 1:
        parser.error('--matching-repeats must be positive')
    args.dataset = args.dataset.resolve()
    if args.baseline_root:
        args.baseline_root = args.baseline_root.resolve()
    return args


def repository_revision(path):
    """Archives inside a checkout must not inherit that checkout's revision."""
    root = Path(path).resolve()
    if not (root / '.git').exists():
        return None
    result = subprocess.run(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def worker(args):
    root = args.baseline_root if args.worker == 'legacy' and args.baseline_root else Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import numpy as np
    from grutopia_extension.interactive_navigation.mapping import SemanticDetection
    from grutopia_extension.interactive_navigation.semantic_exploration_component import (
        SemanticExplorationComponent,
        SemanticExplorationConfig,
    )
    from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
        AgentVLMBackend,
        CLASSIFICATION_LABELS,
        OpenVocabularyPerception,
        OpenVocabularyPerceptionConfig,
    )

    dataset = json.loads(args.dataset.read_text())
    rows, latencies, service_latencies = [], [], []
    supports_config = 'goal_matching' in {field.name for field in fields(SemanticExplorationConfig)}
    for case in dataset['cases']:
        config_options = dict(
            target_query=case['query'],
            target_semantic_classifier=case['classifier'],
            target_min_observations=1,
            target_confirmation_min_label_observations=1,
        )
        goal = None
        if supports_config:
            from grutopia_extension.interactive_navigation.goal_matching import GoalMatchingConfig, GoalQuery

            config_options['goal_matching'] = GoalMatchingConfig(
                mode=args.worker,
                description_threshold=args.description_threshold,
            )
            goal = GoalQuery.parse(case['query'])
        component = SemanticExplorationComponent(SemanticExplorationConfig(**config_options))
        errors, candidate_records, image_hashes = [], [], {}
        backend = AgentVLMBackend(OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl'))
        if args.live_qwen_url:
            backend.config = OpenVocabularyPerceptionConfig(
                semantic_classifier='qwen-vl', qwen_vl_url=args.live_qwen_url
            )
            if case['classifier'] != 'qwen-vl':
                raise ValueError('live crop evaluation supports qwen-vl cases only')
        for index, candidate in enumerate(case['candidates']):
            label, score = candidate['label'], candidate.get('goal_match_score')
            if args.live_qwen_url:
                import cv2

                crop_key = 'masked_crop' if args.worker == 'robust' and args.qwen_vl_crop_mode == 'masked' else 'crop'
                image_path = (args.dataset.parent / candidate[crop_key]).resolve()
                crop = cv2.imread(str(image_path))
                if crop is None:
                    raise ValueError(f'cannot read annotated crop {image_path}')
                # Client expects RGB, identical to the navigation crop convention.
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                image_hashes[candidate['id']] = hashlib.sha256(image_path.read_bytes()).hexdigest()
                target_labels = OpenVocabularyPerception._target_labels(case['query'])
                if args.worker == 'robust' and goal:
                    target_labels = () if goal.descriptive else goal.alternatives
                labels = list(dict.fromkeys((*CLASSIFICATION_LABELS, *target_labels)))
                started = time.perf_counter()
                try:
                    kwargs = {'target_query': case['query']} if args.worker == 'robust' and goal.descriptive else {}
                    result = backend.classify(crop, labels, **kwargs)
                    label, score = result['label'], result.get('goal_match_score')
                    candidate_records.append({'id': candidate['id'], **result})
                except Exception as error:
                    errors.append({'id': candidate['id'], 'type': type(error).__name__, 'message': str(error)})
                    continue
                finally:
                    service_latencies.append((time.perf_counter() - started) * 1000)
            options = dict(
                label=label,
                position=(2.0 * index, 0.0, 0.7),
                step=1,
                confidence=candidate.get('confidence', 0.9),
                sources=('qwen_vl',) if case['classifier'] == 'qwen-vl' else ('open_vocabulary',),
            )
            if args.worker == 'robust' and goal and goal.descriptive:
                options.update(goal_query=goal.text, goal_match_score=score)
            component.mapping.map.scene_graph.update_detections([SemanticDetection(**options)])
        node = component._best_target_node()
        component.target_node = node
        predicted = None
        if node is not None and component.target_confirmed:
            predicted = case['candidates'][int(round(node.position[0] / 2))]['id']
        # Warm matcher cost, excluding service I/O, dataset loading and graph updates.
        for _ in range(args.matching_repeats):
            started = time.perf_counter_ns()
            component._best_target_node()
            latencies.append((time.perf_counter_ns() - started) / 1000)
        rows.append(
            dict(
                id=case['id'],
                query=case['query'],
                expected=case['expected'],
                predicted=predicted,
                correct=predicted == case['expected'],
                service_errors=errors,
                model_outputs=candidate_records,
                image_sha256=image_hashes,
            )
        )
    positives = sum(row['expected'] is not None for row in rows)
    negatives = len(rows) - positives
    accepted = sum(row['predicted'] is not None for row in rows)
    correct_positive = sum(row['predicted'] is not None and row['correct'] for row in rows)
    false_on_negative = sum(row['expected'] is None and row['predicted'] is not None for row in rows)
    metrics = dict(
        cases=len(rows),
        positives=positives,
        negatives=negatives,
        accuracy=sum(row['correct'] for row in rows) / len(rows),
        positive_recall=correct_positive / positives if positives else None,
        false_match_rate_on_absent_goals=false_on_negative / negatives if negatives else None,
        precision=correct_positive / accepted if accepted else None,
        service_errors=sum(len(row['service_errors']) for row in rows),
        warm_matching_median_us=statistics.median(latencies),
        warm_matching_p95_us=float(np.percentile(latencies, 95)),
        service_request_median_ms=statistics.median(service_latencies) if service_latencies else None,
    )
    source = sys.modules[SemanticExplorationComponent.__module__].__file__
    print(
        json.dumps(
            dict(
                mode=args.worker,
                config=asdict(component.config),
                metrics=metrics,
                cases=rows,
                matcher_source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            )
        )
    )


def main():
    args = parse_args()
    if args.worker:
        worker(args)
        return
    dataset = json.loads(args.dataset.read_text())
    if not dataset.get('cases'):
        raise ValueError('dataset must contain annotated cases')
    output = args.record_dir or Path(__file__).resolve().parents[2] / 'grutopia/results' / (
        'goal_matching_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ('legacy', 'robust'):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--worker',
            mode,
            '--dataset',
            str(args.dataset),
            '--matching-repeats',
            str(args.matching_repeats),
            '--description-threshold',
            str(args.description_threshold),
        ]
        if args.baseline_root:
            command += ['--baseline-root', str(args.baseline_root)]
        if args.live_qwen_url:
            command += ['--live-qwen-url', args.live_qwen_url]
        command += ['--qwen-vl-crop-mode', args.qwen_vl_crop_mode]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[2])},
        )
        results.append(json.loads(result.stdout))

    payload = dict(
        schema_version=1,
        scope='single-observation object matching; no navigation evaluation',
        evidence_kind='live-human-annotated-crops' if args.live_qwen_url else 'scripted-contracts',
        candidate_views={'legacy': 'context', 'robust': args.qwen_vl_crop_mode},
        dataset_provenance=dataset.get('provenance'),
        dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        python=sys.version,
        baseline_revision=repository_revision(args.baseline_root) if args.baseline_root else None,
        baseline_root=str(args.baseline_root) if args.baseline_root else None,
        implementation_revision=repository_revision(Path(__file__).resolve().parents[2]),
        results=results,
    )
    (output / 'comparison.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(
        json.dumps(
            {
                'output': str(output / 'comparison.json'),
                'evidence_kind': payload['evidence_kind'],
                'results': [
                    {key: value for key, value in result.items() if key in ('mode', 'metrics')} for result in results
                ],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
