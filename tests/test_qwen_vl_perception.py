"""Focused contract tests; no model weights or simulator required."""
import unittest
import numpy as np
from grutopia_extension.interactive_navigation.mapping import MappingConfig, SceneGraphMap, SemanticDetection
from grutopia_extension.interactive_navigation.open_vocabulary_perception import OpenVocabularyPerception, OpenVocabularyPerceptionConfig, AgentVLMBackend
from grutopia_extension.interactive_navigation.semantic_exploration_component import SemanticExplorationComponent, SemanticExplorationConfig
from tests.test_open_vocabulary_perception import _Session


class _Classifier:
    def __init__(self, label='television'):
        self.label, self.calls = label, []

    def classify(self, image, labels):
        self.calls.append((image.copy(), labels))
        return {'label': self.label, 'model': 'Qwen/Qwen3-VL-8B-Instruct', 'raw_text': '{"label":"' + self.label + '"}'}


class QwenVLPerceptionTest(unittest.TestCase):
    def make_pipeline(self, label='television', clock=lambda: 0, **options):
        self.captions, self.sam_calls = [], []
        classifier = _Classifier(label)
        def detector(image, caption):
            self.captions.append(caption)
            # The same box in both passes should be classified only once.
            return [('monitor', .8, [20, 20, 80, 80])]
        def segment(image, bbox):
            self.sam_calls.append(bbox)
            mask = np.zeros((100, 100), bool)
            mask[20:80, 20:80] = True
            return mask
        perception = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl', **options),
            detector=detector, segmenter=segment, classifier=classifier,
            embedder=lambda image: self.fail('Qwen mode must not load or call CLIP'), clock=clock,
        )
        rgb = np.full((100, 100, 3), 7, np.uint8)
        points = np.zeros((100, 100, 3), np.float32)
        points[20:80, 20:80] = (2, 3, 4)
        return perception, classifier, rgb, np.ones((100, 100)), points

    def test_two_pass_proposals_and_qwen_category_reach_map(self):
        p, classifier, rgb, depth, points = self.make_pipeline()
        result = p.perceive(rgb, depth, points, 'television', step=7)
        self.assertEqual(self.captions[1], 'monitor . screen')
        self.assertNotIn('television', self.captions[0])
        self.assertNotIn('monitor', self.captions[0])
        self.assertIn('mirror', self.captions[0])
        self.assertEqual(len(self.sam_calls), 1)
        self.assertEqual(len(classifier.calls), 1)
        detection = result[0]
        self.assertEqual((detection.raw_label, detection.label, detection.proposal_source), ('monitor', 'television', 'target'))
        self.assertEqual(detection.to_semantic_detection().sources, ('open_vocabulary', 'qwen_vl'))
        self.assertEqual(detection.centroid, (2, 3, 4))
        self.assertEqual(len(p.last_debug_frame['boxes']), 2)
        self.assertEqual(p.last_debug_frame['classified'][0]['classifier_model'], 'Qwen/Qwen3-VL-8B-Instruct')
        crop = classifier.calls[0][0]
        self.assertGreater(crop.shape[0], 60)
        self.assertEqual(int(crop[0, 0, 0]), 7)  # real background retained
        self.assertTrue(np.any(np.all(crop == (255, 0, 0), axis=2)))

    def test_unknown_is_recorded_but_not_mapping_evidence(self):
        p, _, rgb, depth, points = self.make_pipeline('unknown')
        self.assertEqual(p.perceive(rgb, depth, points, 'television'), [])
        self.assertEqual(p.last_debug_frame['classified'][0]['label'], 'unknown')
        self.assertFalse(p.last_debug_frame['classified'][0]['accepted'])
        self.assertEqual(p.last_debug_frame['mapped'], [])

    def test_rate_limit_does_not_replay_classifications(self):
        p, classifier, rgb, depth, points = self.make_pipeline()
        self.assertEqual(len(p.perceive(rgb, depth, points, 'television', step=1)), 1)
        self.assertEqual(p.perceive(rgb, depth, points, 'television', step=2), [])
        self.assertEqual(len(classifier.calls), 1)
        self.assertEqual(p.last_debug_frame['classified'], [])

    def test_candidate_budget_prioritizes_target_pass(self):
        p, *_ = self.make_pipeline(qwen_vl_max_candidates=1)
        selected = p._select_proposals([('chair', .99, (0, 0, 10, 10), 'context'), ('screen', .4, (30, 30, 60, 60), 'target')], 0)
        self.assertEqual(selected[0][3], 'target')

    def test_health_checks_vl_without_loading_clip(self):
        session = _Session(get_payloads=[{'ready': True}] * 3)
        backend = AgentVLMBackend(OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl'), session=session)
        p = OpenVocabularyPerception(backend.config, detector=backend, segmenter=backend, classifier=backend)
        p._ensure_clip = lambda: self.fail('CLIP loading in Qwen mode')
        self.assertIn('qwen_vl', p.check_ready()['services'])
        self.assertEqual(len(session.get_calls), 3)

    def test_qwen_classes_do_not_merge_by_embedding_or_count_same_frame_twice(self):
        graph = SceneGraphMap(MappingConfig())
        def detection(label, step, confidence=.8):
            return SemanticDetection(label=label, position=(1, 1, 1), confidence=confidence, step=step,
                                     embedding=(1., 0.), sources=('open_vocabulary', 'qwen_vl'))
        graph.update_detections([detection('monitor', 1), detection('television', 1), detection('television', 1, .9)])
        nodes = {n.label: n for n in graph.object_nodes()}
        self.assertEqual(set(nodes), {'monitor', 'television'})
        self.assertEqual(nodes['television'].observations, 1)
        graph.update_detections([detection('television', 1)])
        self.assertEqual(nodes['television'].observations, 1)
        graph.update_detections([detection('television', 2)])
        self.assertEqual(graph.label_counts(nodes['television'].node_id)['television'], 2)

    def test_target_keeps_existing_vote_requirement_without_extra_review(self):
        component = SemanticExplorationComponent(SemanticExplorationConfig(target_query='television', target_semantic_classifier='qwen-vl'))
        for step in range(8):
            component.mapping.map.scene_graph.update_detections([SemanticDetection(
                label='tv', position=(1,1,1), confidence=.8, step=step, sources=('qwen_vl',))])
        component.target_node = component._best_target_node()
        self.assertIsNotNone(component.target_node)
        self.assertTrue(component.target_confirmed)
        other = SemanticExplorationComponent(SemanticExplorationConfig(target_query='television', target_semantic_classifier='qwen-vl'))
        for step in range(8):
            other.mapping.map.scene_graph.update_detections([SemanticDetection(
                label='monitor', position=(1,1,1), confidence=.8, step=step, sources=('qwen_vl',))])
        self.assertIsNone(other._best_target_node())
