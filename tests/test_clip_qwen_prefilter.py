"""CLIP filtering must not manufacture votes or remove category observations."""

import unittest

import numpy as np

from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    AgentVLMBackend, OpenVocabularyPerception, OpenVocabularyPerceptionConfig,
)
from grutopia_extension.interactive_navigation.semantic_exploration_component import (
    SemanticExplorationComponent, SemanticExplorationConfig,
)
from tests.test_open_vocabulary_perception import _Session


class ClipQwenPrefilterTest(unittest.TestCase):
    def pipeline(self, image_vector=(1., 0.), text_vector=(0., 1.), score=1., **options):
        self.calls, self.text_calls = [], []
        outer = self

        class Classifier:
            def classify(self, image, labels, **kwargs):
                outer.calls.append(kwargs)
                # Model an old service that ignores the new review switch.
                return {'label': 'chair', 'model': 'test-qwen', 'goal_match_score': score}

        p = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl', query_interval=0,
                                          description_clip_prefilter=True, **options),
            detector=lambda image, caption: [('chair', .9, [0, 0, 8, 8])],
            segmenter=lambda image, bbox: np.ones((8, 8), bool),
            embedder=lambda image: np.asarray(image_vector), classifier=Classifier(),
        )
        p.embed_text = lambda text: self.text_calls.append(text) or np.asarray(text_vector)
        return p

    @staticmethod
    def perceive(p, query='a chair used for sitting', step=0):
        return p.perceive(np.ones((8, 8, 3), np.uint8), np.ones((8, 8)),
                          np.ones((8, 8, 3)), query, step=step)

    def test_filtered_old_service_score_cannot_confirm_but_category_still_maps(self):
        p = self.pipeline()
        component = SemanticExplorationComponent(SemanticExplorationConfig(
            target_query='a chair used for sitting', target_semantic_classifier='qwen-vl'))
        for step in range(8):
            detections = self.perceive(p, step=step)
            self.assertEqual(detections[0].label, 'chair')
            self.assertIsNone(detections[0].goal_match_score)
            component.mapping.map.scene_graph.update_detections(p.as_semantic_detections(detections))
        self.assertTrue(component.mapping.map.scene_graph.object_nodes())
        self.assertIsNone(component._best_target_node())
        self.assertFalse(component.target_confirmed)
        self.assertTrue(all(call['verify_goal'] is False for call in self.calls))
        self.assertEqual(self.text_calls, ['a chair used for sitting'])
        self.assertEqual(p.last_debug_frame['classified'][0]['clip_prefilter']['status'], 'filtered')

    def test_high_clip_score_still_needs_positive_qwen_review(self):
        p = self.pipeline(text_vector=(1., 0.), score=0.)
        result = self.perceive(p)
        self.assertEqual(result[0].goal_match_score, 0.)
        self.assertNotIn('verify_goal', self.calls[0])
        self.assertEqual(p.last_debug_frame['classified'][0]['clip_prefilter']['status'], 'passed')

    def test_unavailable_clip_falls_back_to_qwen_once_and_records_failure(self):
        p = self.pipeline()
        def unavailable(query):
            self.text_calls.append(query)
            raise FileNotFoundError('model unavailable')
        p.embed_text = unavailable
        for step in range(2):
            self.assertEqual(self.perceive(p, step=step)[0].goal_match_score, 1.)
        self.assertEqual(len(self.text_calls), 1)
        self.assertTrue(all('verify_goal' not in call for call in self.calls))
        self.assertEqual(p.last_debug_frame['classified'][0]['clip_prefilter']['error'], 'FileNotFoundError')

    def test_different_attributes_do_not_share_text_vectors(self):
        p = self.pipeline(text_vector=(1., 0.))
        for query in ('a red chair', 'a gray chair', 'a red chair'):
            self.perceive(p, query)
        self.assertEqual(self.text_calls, ['a red chair', 'a gray chair'])

    def test_named_categories_do_not_load_clip_in_qwen_mode(self):
        p = self.pipeline()
        self.perceive(p, 'chair')
        self.assertEqual(self.text_calls, [])
        self.assertNotIn('target_query', self.calls[0])

    def test_invalid_vectors_fall_back_instead_of_filtering(self):
        for vector in ((float('nan'), 1.), (0., 0.), (1., 2., 3.)):
            with self.subTest(vector=vector):
                p = self.pipeline(text_vector=vector)
                self.assertEqual(self.perceive(p)[0].goal_match_score, 1.)
                self.assertEqual(p.last_debug_frame['classified'][0]['clip_prefilter']['status'], 'fallback')

    def test_rest_client_transmits_false_review_switch(self):
        session = _Session([{'label': 'chair', 'model': 'test'}])
        backend = AgentVLMBackend(session=session)
        backend.classify(np.ones((8, 8, 3), np.uint8), ['chair', 'unknown'],
                         target_query='a red chair', verify_goal=False)
        self.assertIs(session.calls[0][1]['json']['verify_goal'], False)

    def test_invalid_thresholds_are_rejected(self):
        for value in (float('nan'), float('inf'), -1.1, 1.1):
            with self.assertRaises(ValueError):
                OpenVocabularyPerceptionConfig(description_clip_threshold=value)


if __name__ == '__main__':
    unittest.main()
