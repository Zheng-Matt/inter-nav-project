"""Candidate RGB/mask/geometry contracts without models or simulator."""
import unittest
from dataclasses import replace

import numpy as np

from grutopia_extension.interactive_navigation.goal_matching import GoalMatchingConfig
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    OpenVocabularyPerception, OpenVocabularyPerceptionConfig,
)


class ForegroundGoalCandidateTest(unittest.TestCase):
    def pipeline(self, height=.7, **options):
        self.images = []
        class Classifier:
            def classify(_, image, labels, **kwargs):
                self.images.append(image.copy())
                return {'label': 'refrigerator', 'model': 'test', 'goal_match_score': .99}
        rgb = np.full((10, 10, 3), (240, 30, 10), dtype=np.uint8)
        rgb[3:7, 3:7] = (60, 80, 100)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        points = np.zeros((10, 10, 3), dtype=np.float32)
        points[..., 2] = height
        p = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl', semantic_ground_height=.15, **options),
            detector=lambda image, query: [('fridge', .9, (2, 2, 8, 8))],
            segmenter=lambda image, bbox: mask.copy(), classifier=Classifier(), clock=lambda: 0,
        )
        return p, rgb, np.ones((10, 10)), points

    def test_mask_preserves_foreground_color_and_input(self):
        p, rgb, depth, points = self.pipeline()
        original = rgb.copy()
        result = p.perceive(rgb, depth, points, 'fridge', step=1)
        self.assertEqual(len(result), 1)
        crop = self.images[0]
        np.testing.assert_array_equal(crop[0, 0], (127, 127, 127))
        np.testing.assert_array_equal(crop[5, 5], (60, 80, 100))
        np.testing.assert_array_equal(rgb, original)
        self.assertEqual(p.last_debug_frame['classified'][0]['crop_mode'], 'masked')

    def test_floor_rejects_before_classifier_class_votes_and_cues(self):
        p, rgb, depth, points = self.pipeline(height=.151)
        self.assertEqual(p.perceive(rgb, depth, points, 'fridge', step=1), [])
        self.assertEqual(self.images, [])
        self.assertEqual(p.last_target_cues, [])
        self.assertEqual(p.last_debug_frame['mapped'], [])
        self.assertEqual(p.last_debug_frame['rejected_geometry'][0]['reason'], 'ground_surface')

    def test_visible_object_survives_floor_pixels_in_mask(self):
        p, rgb, depth, points = self.pipeline(height=.151)
        points[3:5, 3:7, 2] = .7
        result = p.perceive(rgb, depth, points, 'a large appliance used to keep food cold', step=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].goal_match_score, .99)
        self.assertFalse(p.last_debug_frame['rejected_geometry'])

    def test_context_opt_out_and_legacy_preserve_original_input(self):
        for options in ({'qwen_vl_crop_mode': 'context', 'semantic_ground_clearance': 0},
                        {'goal_matching': GoalMatchingConfig(mode='legacy')}):
            p, rgb, depth, points = self.pipeline(height=.151, **options)
            self.assertEqual(len(p.perceive(rgb, depth, points, 'fridge', step=1)), 1)
            np.testing.assert_array_equal(self.images[0][0, 0], (240, 30, 10))
            self.assertEqual(p.last_debug_frame['classified'][0]['crop_mode'], 'context')

    def test_invalid_or_empty_masks_cannot_restore_unfocused_context(self):
        p, rgb, _, _ = self.pipeline()
        for mask in (np.ones((3, 3), bool), np.ones((10, 10)), np.zeros((10, 10), bool)):
            with self.subTest(shape=mask.shape, dtype=str(mask.dtype)):
                with self.assertRaises(ValueError):
                    p._qwen_context_crop(rgb, (2, 2, 8, 8), mask=mask)
        for options in ({'qwen_vl_crop_mode': 'typo'}, {'semantic_ground_clearance': float('nan')},
                        {'semantic_ground_height': float('inf')}):
            with self.assertRaises(ValueError):
                OpenVocabularyPerceptionConfig(**options)


if __name__ == '__main__':
    unittest.main()
