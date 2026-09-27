"""CPU-only regressions for simulator-label isolation and unlimited steps."""
import ast
import dataclasses
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from grutopia_extension.interactive_navigation.mapping import SemanticDetection
from grutopia_extension.interactive_navigation.mapping_runtime import MapNavigationRuntime
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    OpenVocabularyPerception, OpenVocabularyPerceptionConfig,
)
from grutopia_extension.interactive_navigation.semantic_exploration_component import (
    SemanticExplorationComponent, SemanticExplorationConfig, SemanticExplorationStatus,
)
from grutopia_extension.interactive_navigation.visualization import (
    InteractionVideoRecorder, model_only_recording_observation,
)


class ModelOnlySemanticsTest(unittest.TestCase):
    def setUp(self):
        self.camera = {
            'rgba': np.zeros((3, 4, 4), dtype=np.uint8),
            'depth': np.ones((3, 4)),
            'bounding_box_2d_tight': {'info': {'idToLabels': {'1': {'class': 'refrigerator'}}},
                                      'data': [(1, 0, 0, 4, 3)]},
        }
        self.points = np.ones((3, 4, 3))

    def test_model_miss_does_not_import_simulator_target(self):
        perception = SimpleNamespace(perceive=lambda **kwargs: [])
        component = SemanticExplorationComponent(
            SemanticExplorationConfig(target_query='refrigerator', semantic_source='model'),
            perception=perception,
        )
        with patch('grutopia_extension.interactive_navigation.mapping_runtime._semantic_detections',
                   side_effect=AssertionError('simulator labels were accessed')):
            observations = component.mapping._camera_semantic_detections(self.camera, self.points, 0)
        component.mapping.map.scene_graph.update_detections(observations)
        self.assertEqual(observations, ())
        self.assertIsNone(component._best_target_node())
        self.assertEqual(component.statistics()['simulator_semantic_observations'], 0)

    def test_model_results_enter_map_without_simulator_labels(self):
        chair = SemanticDetection('chair', (1.0, 1.0, .5), confidence=.8)
        perception = SimpleNamespace(perceive=lambda **kwargs: [chair])
        runtime = MapNavigationRuntime(semantic_source='model', semantic_target='chair',
                                       open_vocabulary_perception=perception)
        result = runtime._camera_semantic_detections(self.camera, self.points, 24)
        runtime.map.scene_graph.update_detections(result)
        self.assertEqual([n.label for n in runtime.map.scene_graph.object_nodes()], ['chair'])
        self.assertEqual(runtime.model_semantic_observations, 1)
        self.assertEqual(runtime.simulator_semantic_observations, 0)
        self.assertNotIn('bounding_box_2d_tight', runtime.last_model_camera)

    def test_strict_service_failure_propagates_without_fallback(self):
        def fail(*args):
            raise RuntimeError('detector unavailable')
        fallback = unittest.mock.Mock(return_value=[])
        perception = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(strict_errors=True), detector=fail,
            segmenter=lambda *args: None, simulator_fallback=fallback,
        )
        runtime = MapNavigationRuntime(semantic_source='model', semantic_target='chair',
                                       open_vocabulary_perception=perception)
        with self.assertRaisesRegex(RuntimeError, 'detector unavailable'):
            runtime._camera_semantic_detections(self.camera, self.points, 0)
        fallback.assert_not_called()
        self.assertEqual(runtime.open_vocabulary_failures, 1)
        self.assertEqual(runtime.simulator_semantic_observations, 0)

    def test_mixed_mode_preserves_existing_fallback(self):
        truth = SemanticDetection('refrigerator', (2., 1., .5))
        runtime = MapNavigationRuntime()
        with patch('grutopia_extension.interactive_navigation.mapping_runtime._semantic_detections',
                   return_value=[truth]):
            self.assertEqual(runtime._camera_semantic_detections(self.camera, self.points, 0), (truth,))

    def test_model_mode_requires_a_model(self):
        with self.assertRaises(ValueError):
            MapNavigationRuntime(semantic_source='model', semantic_target='chair')

    def test_recording_removes_truth_and_uses_exact_model_frame(self):
        original = {'sensors': {'camera': self.camera, 'overview_camera': dict(self.camera)}}
        model_frame = {'rgba': np.ones((3, 4, 4)), 'model_step': 24,
                       'model_detections': [{'label': 'chair', 'confidence': .8, 'bbox': (0, 0, 3, 2)}]}
        runtime = SimpleNamespace(last_model_camera=model_frame)
        result = model_only_recording_observation(original, runtime)
        self.assertIs(result['sensors']['camera'], model_frame)
        self.assertNotIn('bounding_box_2d_tight', result['sensors']['overview_camera'])
        self.assertIn('bounding_box_2d_tight', original['sensors']['camera'])
        runtime.last_model_camera = None
        result = model_only_recording_observation(original, runtime)
        self.assertNotIn('bounding_box_2d_tight', result['sensors']['camera'])

    def test_model_video_draws_model_confidence_not_truth(self):
        recorder = object.__new__(InteractionVideoRecorder)
        recorder.rgb_size = (640, 360)
        camera = dict(self.camera, model_step=24, model_detections=[
            {'label': 'chair', 'confidence': .8, 'bbox': (0, 0, 3, 2)},
        ])
        with patch.object(recorder, '_draw_camera_boxes', side_effect=AssertionError('truth overlay')):
            with patch('cv2.putText') as draw:
                recorder._render_camera(40, 'explore', {'sensors': {'camera': camera}}, 'camera', 'RGB')
        text = [call.args[1] for call in draw.call_args_list]
        self.assertIn('chair 0.80', text)
        self.assertTrue(any('observed step 24' in x for x in text))
        self.assertFalse(any('refrigerator' in x for x in text))

    def test_unlimited_budget_still_stops_on_success_or_fall(self):
        component = SemanticExplorationComponent(SemanticExplorationConfig(target_query='chair', max_steps=0))
        obs = {'position': (0., 0., .4)}
        self.assertEqual(component.evaluate(10**15, obs).status, SemanticExplorationStatus.RUNNING)
        self.assertEqual(component.evaluate(10**15, {'position': (0., 0., 0.)}).failure_reason, 'robot_fell')
        component.target_node = SimpleNamespace(position=(1., 0., .4), node_id='object:chair:1')
        component.target_navigation_position = (0., 0., .4)
        self.assertTrue(component.evaluate(10**15, obs).success)

    def test_finite_budget_is_unchanged(self):
        component = SemanticExplorationComponent(SemanticExplorationConfig(target_query='chair', max_steps=2))
        self.assertEqual(component.evaluate(1, {'position': (0., 0., .4)}).failure_reason, 'global_step_limit')

    def test_cli_and_run_config_accept_model_and_unlimited_without_isaac(self):
        # Execute the real parser and dataclass definitions without importing
        # Isaac or loading a GPU scene just to test configuration validation.
        import argparse
        import sys
        import tempfile
        root = Path(__file__).resolve().parents[1]
        runner = ast.parse((root/'grutopia_extension/interactive_navigation/go2_navigation_runner.py').read_text())
        definition = next(n for n in runner.body if isinstance(n, ast.ClassDef) and n.name=='Go2SemanticExplorationRunConfig')
        namespace = {'dataclass': dataclasses.dataclass, 'os': os,
                     'DEFAULT_GO2_POLICY_PATH': 'policy', 'DEFAULT_GO2_USD_PATH': 'robot'}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), '<run-config>', 'exec'), namespace)
        config = namespace['Go2SemanticExplorationRunConfig']
        self.assertEqual(config(max_steps=0, semantic_source='model').max_steps, 0)
        with self.assertRaises(ValueError):
            config(semantic_source='model', enable_open_vocabulary=False)
        parser_tree = ast.parse((root/'grutopia/demo/go2_semantic_exploration.py').read_text())
        parser_def = next(n for n in parser_tree.body if isinstance(n, ast.FunctionDef) and n.name=='parse_args')
        namespace.update(argparse=argparse, sys=sys, has_display=lambda: False,
                         DEFAULT_GRSCENE_PROFILE='profile')
        exec(compile(ast.Module(body=[parser_def], type_ignores=[]), '<parse-args>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'INTERNAV_KIT_ROOT': directory}):
                with patch.object(sys, 'argv', ['test', '--semantic-source', 'model', '--max-steps', '0']):
                    args = namespace['parse_args']()
        self.assertEqual((args.semantic_source, args.max_steps), ('model', 0))


if __name__ == '__main__':
    unittest.main()
