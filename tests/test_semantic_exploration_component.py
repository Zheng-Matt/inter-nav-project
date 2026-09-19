import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from grutopia_extension.interactive_navigation.mapping import MappingConfig, SemanticDetection
from grutopia_extension.interactive_navigation.semantic_exploration_component import (
    SemanticExplorationComponent,
    SemanticExplorationConfig,
    SemanticExplorationStatus,
)
from grutopia_extension.interactive_navigation.semantic_voronoi import SemanticVoronoiConfig


class _FakeTextEmbeddingPerception:
    def embed_text(self, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _component(target='carry object', frontier_selection_interval=1, perception=None):
    return SemanticExplorationComponent(
        SemanticExplorationConfig(
            target_query=target,
            mapping=MappingConfig(
                x_limits=(0.0, 5.0),
                y_limits=(0.0, 4.0),
                grid_resolution=0.1,
                robot_radius=0.1,
                safe_recovery_y_limits=(0.1, 3.9),
            ),
            voronoi=SemanticVoronoiConfig(spur_length=0.0),
            target_min_observations=2,
            topology_update_interval=1,
            frontier_selection_interval=frontier_selection_interval,
        ),
        perception=perception,
    )


class SemanticExplorationComponentTest(unittest.TestCase):
    def test_detection_mode_reaches_the_mapping_runtime(self):
        component = _component('refrigerator')
        self.assertEqual(component.mapping.semantic_detection_mode.value, 'hybrid')
        self.assertEqual(
            component.statistics()['semantic_detection_effective_mode'],
            'isaac',
        )

        open_vocab = SemanticExplorationComponent(
            SemanticExplorationConfig(
                target_query='refrigerator',
                semantic_detection_mode='open_vocab',
            ),
            perception=_FakeTextEmbeddingPerception(),
        )
        self.assertEqual(open_vocab.mapping.semantic_detection_mode.value, 'open_vocab')

    def test_hybrid_preflight_failure_is_persisted_in_statistics(self):
        component = SemanticExplorationComponent(
            SemanticExplorationConfig(
                target_query='refrigerator',
                semantic_detection_mode='hybrid',
                open_vocabulary_startup_error='ConnectionError: service offline',
            ),
        )

        stats = component.statistics()
        self.assertEqual(stats['semantic_detection_mode'], 'hybrid')
        self.assertEqual(stats['semantic_detection_effective_mode'], 'isaac')
        self.assertFalse(stats['open_vocabulary_available'])
        self.assertEqual(
            stats['open_vocabulary_startup_error'],
            'ConnectionError: service offline',
        )

    def test_open_vocab_mode_without_perception_is_rejected(self):
        with self.assertRaises(ValueError):
            SemanticExplorationComponent(
                SemanticExplorationConfig(
                    target_query='refrigerator',
                    semantic_detection_mode='open_vocab',
                )
            )

    def test_unknown_detection_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            SemanticExplorationConfig(target_query='refrigerator', semantic_detection_mode='radar')

    def test_empty_map_rate_limits_frontier_reselection(self):
        component = _component('refrigerator', frontier_selection_interval=160)
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}

        for step in range(20):
            component.update(step, observation)

        self.assertEqual(len(component.decision_history), 1)
        self.assertIsNone(component.current_goal)

    def test_unknown_target_selects_reachable_frontier(self):
        component = _component('refrigerator')
        component.mapping.map.occupancy.observed[8:32, 5:30] = True
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}

        component.update(0, observation)

        self.assertIsNone(component.target_node)
        self.assertIsNotNone(component.current_frontier_id)
        self.assertIsNotNone(component.current_goal)
        self.assertIn(component.state, ('explore_geometric', 'explore_semantic'))

    def test_persistent_semantic_target_switches_to_target_navigation(self):
        component = _component()
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (3.0, 1.5, 0.7), confidence=0.9),
                SemanticDetection('carry_object', (3.0, 1.5, 0.7), confidence=0.9),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}

        component.update(0, observation)

        self.assertIsNotNone(component.target_node)
        self.assertEqual(component.state, 'navigate_to_semantic_target')
        self.assertIsNotNone(component.current_goal)
        self.assertNotEqual(component.current_goal[:2], component.target_node.position[:2])

    def test_embedding_target_upgrades_to_lexical_match(self):
        component = _component('refrigerator', perception=_FakeTextEmbeddingPerception())
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        look_alike = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('door', (3.0, 1.5, 0.7), confidence=0.9, embedding=tuple(look_alike)),
                SemanticDetection('door', (3.0, 1.5, 0.7), confidence=0.9, embedding=tuple(look_alike)),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}

        component.update(0, observation)

        self.assertIsNotNone(component.target_node)
        self.assertEqual(component.target_node.label, 'door')
        self.assertEqual(component.statistics()['target_match'], 'embedding')

        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection(
                    'refrigerator',
                    (4.0, 2.5, 0.8),
                    confidence=0.9,
                    sources=('isaac',),
                ),
                SemanticDetection(
                    'refrigerator',
                    (4.0, 2.5, 0.8),
                    confidence=0.9,
                    sources=('isaac',),
                ),
            ]
        )
        component.update(1, observation)

        self.assertEqual(component.target_node.label, 'refrigerator')
        self.assertEqual(component.statistics()['target_match'], 'lexical')
        self.assertEqual(component.statistics()['target_match_method'], 'lexical')
        self.assertEqual(component.statistics()['target_sources'], ['isaac'])
        self.assertIsNotNone(component.current_goal)

    def test_lexical_target_upgrades_only_to_strictly_better_node(self):
        component = _component()
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}
        component.update(0, observation)
        locked = component.target_node.node_id

        # A same-label node that merely ties on (confidence, observations)
        # must not steal the lock.
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (4.5, 3.0, 0.7)),
                SemanticDetection('carry_object', (4.5, 3.0, 0.7)),
            ]
        )
        component.update(1, observation)
        self.assertEqual(component.target_node.node_id, locked)

        # Once the other instance is strictly better observed (the real
        # object outpaces a long-range ghost estimate), the lock upgrades
        # and the approach position is recomputed.
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (4.5, 3.0, 0.7)),
                SemanticDetection('carry_object', (4.5, 3.0, 0.7)),
            ]
        )
        component.update(2, observation)

        self.assertNotEqual(component.target_node.node_id, locked)
        self.assertEqual(
            tuple(round(value, 2) for value in component.target_node.position[:2]),
            (4.5, 3.0),
        )
        self.assertEqual(component.statistics()['target_match'], 'lexical')

    def test_success_and_persistence_include_semantic_topology(self):
        component = _component()
        component.mapping.map.occupancy.observed[8:32, 5:45] = True
        component.mapping.map.scene_graph.update_detections(
            [
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
                SemanticDetection('carry_object', (3.0, 1.5, 0.7)),
            ]
        )
        observation = {'position': (1.0, 1.5, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0), 'sensors': {}}
        component.update(0, observation)
        reached = dict(observation)
        reached['position'] = component.current_goal

        result = component.evaluate(1, reached)

        self.assertEqual(result.status, SemanticExplorationStatus.SUCCEEDED)
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'semantic_map')
            component.save(prefix)
            payload = json.loads(Path(prefix + '.json').read_text(encoding='utf-8'))
            self.assertIn('semantic_voronoi', payload)
            self.assertIn('semantic_exploration', payload)


if __name__ == '__main__':
    unittest.main()
