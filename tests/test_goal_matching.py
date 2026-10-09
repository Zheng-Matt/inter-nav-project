"""Goal matching contracts; no simulator, network or model weights."""

import unittest
import json
import tempfile
from pathlib import Path
from dataclasses import replace

import numpy as np

from grutopia_extension.interactive_navigation.goal_matching import (
    GoalMatchingConfig,
    GoalQuery,
    normalized_embedding,
    parse_goal_score,
)
from grutopia_extension.interactive_navigation.mapping import MappingConfig, SemanticDetection
from grutopia_extension.interactive_navigation.mapping_runtime import _deduplicate_semantic_detections
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    OpenVocabularyPerception,
    OpenVocabularyPerceptionConfig,
)
from grutopia_extension.interactive_navigation.semantic_exploration_component import (
    SemanticExplorationComponent,
    SemanticExplorationConfig,
)


class GoalMatchingTest(unittest.TestCase):
    def component(self, query='fridge', classifier='qwen-vl', perception=None, **options):
        return SemanticExplorationComponent(
            SemanticExplorationConfig(
                target_query=query,
                target_semantic_classifier=classifier,
                goal_matching=GoalMatchingConfig(**options),
            ),
            perception=perception,
        )

    def evidence(self, component, label, count=8, score=None, query=None, position=(2.0, 2.0, 0.7), **options):
        for step in range(count):
            component.mapping.map.scene_graph.update_detections(
                [
                    SemanticDetection(
                        label,
                        position,
                        step=step,
                        sources=('open_vocabulary', 'qwen_vl'),
                        goal_query=query,
                        goal_match_score=score,
                        **options,
                    )
                ]
            )

    def test_aliases_and_prefixes_use_exact_category_evidence(self):
        for query, label in [
            ('fridge', 'refrigerator'),
            ('Find the fridge!', 'refrigerator'),
            ('couch', 'sofa'),
            ('TV', 'television'),
            ('potted_plant', 'plant'),
            ('refrigerator | chair', 'chair'),
        ]:
            with self.subTest(query=query):
                component = self.component(query)
                self.evidence(component, label)
                component.target_node = component._best_target_node()
                self.assertTrue(component.target_confirmed)
        for query, label in [
            ('chair', 'chair cushion'),
            ('television', 'monitor'),
            ('refrigerator cabinet', 'cabinet'),
            ('plant', 'unknown'),
        ]:
            component = self.component(query, classifier='clip')
            self.evidence(component, label)
            self.assertIsNone(component._best_target_node())

    def test_attributes_and_functional_descriptions_cannot_use_category_votes(self):
        for query in [
            'white fridge',
            'a large appliance used to keep food cold',
            'the white appliance in the kitchen',
            'comfortable chair',
            'chair with arms',
        ]:
            component = self.component(query)
            self.assertTrue(component._goal_query.descriptive)
            self.evidence(component, 'refrigerator')
            self.assertIsNone(component._best_target_node())
            self.assertFalse(component.target_confirmed)

    def test_description_scores_count_distinct_frames_and_query_identity(self):
        component = self.component('white fridge')
        query = component._goal_query.text
        self.evidence(component, 'refrigerator', count=8, query=query, score=0.95)
        node = component._best_target_node()
        component.target_node = node
        self.assertTrue(component.target_confirmed)
        self.assertEqual(component.statistics()['target_match_method'], 'description')
        self.evidence(component, 'refrigerator', count=8, query=query, score=0.95)
        self.assertEqual(component._target_label_support(), (8, 8))
        # New queries cannot inherit description confirmations.
        other = self.component('black fridge')
        other.mapping.map.scene_graph = component.mapping.map.scene_graph
        self.assertIsNone(other._best_target_node())

    def test_negative_and_unverifiable_attributes_fail_closed(self):
        for score in [None, 0.79, True, '0.99', float('nan'), float('inf'), -1, 2]:
            component = self.component('white fridge')
            self.evidence(component, 'refrigerator', score=score, query='white fridge')
            self.assertIsNone(component._best_target_node())
        component = self.component('white fridge')
        self.evidence(component, 'refrigerator', score=0.95, query='white fridge')
        # Category labels cannot hide contradictory attribute evidence.
        for step in range(8, 24):
            component.mapping.map.scene_graph.update_detections(
                [
                    SemanticDetection(
                        'refrigerator',
                        (2.0, 2.0, 0.7),
                        step=step,
                        sources=('qwen_vl',),
                        goal_query='white fridge',
                        goal_match_score=0.05,
                    )
                ]
            )
        self.assertIsNone(component._best_target_node())

    def test_same_category_different_instances_retain_separate_description_votes(self):
        component = self.component('white fridge')
        self.evidence(component, 'refrigerator', score=0.1, query='white fridge')
        self.evidence(component, 'refrigerator', score=0.95, query='white fridge', position=(4.0, 4.0, 0.7))
        self.assertEqual(component._best_target_node().position[:2], (4.0, 4.0))

    def test_verified_description_reaches_navigation_and_persists_evidence(self):
        component = self.component('white fridge')
        self.evidence(component, 'refrigerator', query='white fridge', score=0.95)
        component.mapping.map.occupancy.observed[:] = True
        component.mapping.update = lambda step, observation: None
        observation = {'position': (0.0, 0.0, 0.4), 'orientation': (1.0, 0.0, 0.0, 0.0)}
        component.update(20, observation)
        self.assertEqual(component.state, 'navigate_to_semantic_target')
        self.assertTrue(component.target_confirmed)
        near = dict(observation, position=component.current_goal)
        self.assertTrue(component.evaluate(21, near).success)
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'final_map')
            component.save(prefix)
            payload = json.loads(Path(prefix + '.json').read_text())
            self.assertEqual(
                payload['scene_graph']['goal_match_evidence'][component.target_node.node_id]['white fridge'], [0.95] * 8
            )

    def test_description_cannot_bypass_verification_when_lexical_requirement_disabled(self):
        component = self.component('white fridge')
        component.config = replace(component.config, require_lexical_confirmation=False)
        self.evidence(component, 'refrigerator')
        component.target_node = component.mapping.map.scene_graph.object_nodes()[0]
        component._target_lexical = True
        self.assertFalse(component.target_confirmed)

    def test_hybrid_fusion_keeps_vl_verification_without_promoting_ground_truth(self):
        vl = SemanticDetection(
            'refrigerator',
            (2.0, 2.0, 0.7),
            step=1,
            sources=('qwen_vl',),
            goal_query='white fridge',
            goal_match_score=0.95,
        )
        gt = replace(vl, sources=('isaac',), goal_query=None, goal_match_score=None, point_count=100)
        fused = _deduplicate_semantic_detections([vl, gt])[0]
        self.assertEqual(fused.goal_match_score, 0.95)
        component = self.component('white fridge')
        for step in range(8):
            component.mapping.map.scene_graph.update_detections([replace(fused, step=step)])
        self.assertIsNotNone(component._best_target_node())
        conflict = replace(vl, goal_match_score=0.05)
        self.assertEqual(_deduplicate_semantic_detections([vl, conflict])[0].goal_match_score, 0.05)

    def test_embedding_cosine_margin_and_failure_fallback(self):
        class Perception:
            calls = 0

            def embed_text(self, text):
                self.calls += 1
                return np.array([10.0, 0.0])

        perception = Perception()
        component = self.component('fridge', classifier='clip', perception=perception)
        for step in range(2):
            component.mapping.map.scene_graph.update_detections(
                [
                    SemanticDetection('door', (2.0, 2.0, 0.7), embedding=(10.0, 0.0), step=step),
                    SemanticDetection('cabinet', (4.0, 4.0, 0.7), embedding=(10.0, 0.1), step=step),
                ]
            )
        self.assertIsNone(component._best_target_node())
        self.assertEqual(perception.calls, 1)
        component._best_target_node()
        self.assertEqual(perception.calls, 1)
        for value in [None, (), (0.0, 0.0), (float('nan'), 1), [[1, 0]]]:
            self.assertIsNone(normalized_embedding(value))

        class Broken:
            def embed_text(self, text):
                raise RuntimeError('unavailable')

        component = self.component('fridge', classifier='clip', perception=Broken())
        self.evidence(component, 'door')
        self.assertIsNone(component._best_target_node())
        self.evidence(component, 'refrigerator', position=(4.0, 4.0, 0.7))
        self.assertEqual(component._best_target_node().label, 'refrigerator')

    def test_legacy_switch_retains_old_substring_baseline(self):
        component = self.component('white refrigerator', classifier='clip', mode='legacy')
        self.evidence(component, 'refrigerator')
        self.assertIsNotNone(component._best_target_node())

    def test_configuration_rejects_nonfinite_and_invalid_values(self):
        for options in [
            dict(mode='typo'),
            dict(description_threshold=float('nan')),
            dict(embedding_margin=-1),
            dict(min_node_confidence=2),
        ]:
            with self.assertRaises(ValueError):
                GoalMatchingConfig(**options)
        for query in ['', '|', ' . ']:
            with self.assertRaises(ValueError):
                GoalQuery.parse(query)
        self.assertEqual(parse_goal_score(0.8), 0.8)


class DescriptionPerceptionTest(unittest.TestCase):
    def test_full_description_is_verified_in_same_classification_and_recorded(self):
        calls = []

        class Classifier:
            def classify(self, image, labels, target_query=None):
                calls.append((labels, target_query))
                return {
                    'label': 'refrigerator',
                    'model': 'existing-qwen',
                    'goal_match_score': 0.91,
                    'goal_verification_status': 'verified',
                }

        perception = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl'),
            detector=lambda image, caption: [('fridge', 0.9, [0, 0, 8, 8])],
            segmenter=lambda image, bbox: np.ones((8, 8), bool),
            classifier=Classifier(),
        )
        image = np.ones((8, 8, 3), np.uint8)
        output = perception.perceive(image, np.ones((8, 8)), np.ones((8, 8, 3)), 'white fridge', step=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 'white fridge')
        self.assertNotIn('white fridge', calls[0][0])
        self.assertEqual(perception.build_target_query('white fridge'), 'white fridge')
        self.assertEqual(output[0].to_semantic_detection().goal_match_score, 0.91)
        self.assertEqual(perception.last_debug_frame['classified'][0]['goal_verification_status'], 'verified')

    def test_older_service_missing_goal_fields_cannot_confirm_description(self):
        class OldClassifier:
            def classify(self, image, labels, **kwargs):
                return {'label': 'refrigerator', 'model': 'existing-qwen'}

        perception = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl'),
            detector=lambda image, caption: [('fridge', 0.9, [0, 0, 8, 8])],
            segmenter=lambda image, bbox: np.ones((8, 8), bool),
            classifier=OldClassifier(),
        )
        result = perception.perceive(np.ones((8, 8, 3), np.uint8), np.ones((8, 8)), np.ones((8, 8, 3)), 'white fridge')
        self.assertIsNone(result[0].goal_match_score)


if __name__ == '__main__':
    unittest.main()
