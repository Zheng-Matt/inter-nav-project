import base64
import unittest

import numpy as np
import requests

from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    AgentVLMBackend,
    OpenVocabularyDetection,
    OpenVocabularyPerception,
    OpenVocabularyPerceptionConfig,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(next(self.payloads))


class OpenVocabularyPerceptionTest(unittest.TestCase):
    def setUp(self):
        self.rgba = np.zeros((3, 4, 4), dtype=np.uint8)
        self.depth = np.ones((3, 4), dtype=np.float32)
        self.point_image = np.zeros((3, 4, 3), dtype=np.float32)
        for row in range(3):
            for column in range(4):
                self.point_image[row, column] = (column, row, 10 + row * 4 + column)

    def test_mask_pixels_align_with_world_points_and_convert_to_mapping(self):
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 1] = True
        mask[2, 3] = True
        perception = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(query_interval=0),
            detector=lambda image, query: [
                {'label': 'mug', 'confidence': 0.8, 'bbox': [0.25, 0.0, 1.0, 1.0]}
            ],
            segmenter=lambda image, bbox: mask,
            embedder=lambda crop: np.array([3.0, 4.0]),
        )

        detections = perception.perceive(self.rgba, self.depth, self.point_image, 'mug', step=7)

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.bbox, (1.0, 0.0, 4.0, 3.0))
        self.assertEqual(detection.point_count, 2)
        np.testing.assert_allclose(detection.centroid, (2.0, 1.0, 16.0))
        np.testing.assert_allclose(detection.embedding, (0.6, 0.8))
        semantic = detection.to_semantic_detection()
        self.assertEqual(semantic.label, 'mug')
        self.assertEqual(semantic.position, detection.centroid)
        self.assertEqual(semantic.confidence, 0.8)
        self.assertEqual(semantic.point_count, 2)
        self.assertEqual(semantic.step, 7)
        self.assertEqual(semantic.sources, ('open_vocabulary',))
        np.testing.assert_allclose(semantic.embedding, (0.6, 0.8))

    def test_embedding_uses_tight_masked_crop_and_l2_normalizes(self):
        self.rgba[..., :3] = 7
        mask = np.zeros((3, 4), dtype=bool)
        mask[1:, 1:3] = True
        seen_crops = []

        perception = OpenVocabularyPerception(
            detector=lambda image, query: [('chair', 1.0, [0, 0, 4, 3])],
            segmenter=lambda image, bbox: mask,
            embedder=lambda crop: seen_crops.append(crop.copy()) or np.array([0.0, -5.0, 0.0]),
        )
        detection = perception.perceive(self.rgba, self.depth, self.point_image, 'chair')[0]

        self.assertEqual(seen_crops[0].shape, (2, 2, 3))
        np.testing.assert_array_equal(seen_crops[0], np.full((2, 2, 3), 7, dtype=np.uint8))
        np.testing.assert_allclose(detection.embedding, (0.0, -1.0, 0.0))
        self.assertAlmostEqual(float(np.linalg.norm(detection.embedding)), 1.0)

    def test_service_failure_calls_simulator_fallback_without_exiting(self):
        sentinel = OpenVocabularyDetection(
            label='simulator-door',
            confidence=1.0,
            bbox=(0, 0, 1, 1),
            mask=np.ones((3, 4), dtype=bool),
            centroid=(1.0, 2.0, 3.0),
            point_count=12,
        )
        fallback_calls = []

        def fail(_image, _query):
            raise requests.Timeout('short timeout')

        def fallback(**kwargs):
            fallback_calls.append(kwargs)
            return [sentinel]

        perception = OpenVocabularyPerception(
            detector=fail,
            segmenter=lambda image, bbox: np.ones(image.shape[:2], dtype=bool),
            embedder=lambda crop: np.ones(2),
            simulator_fallback=fallback,
        )

        self.assertEqual(perception.perceive(self.rgba, self.depth, self.point_image, 'door'), [sentinel])
        self.assertEqual(len(fallback_calls), 1)
        self.assertIn('door', fallback_calls[0]['query'])
        self.assertIn('chair', fallback_calls[0]['query'])

    def test_rest_backend_uses_short_timeout_and_parses_service_formats(self):
        mask = np.array(
            [[False, True, False, False], [True, True, False, False], [False, False, False, False]],
            dtype=np.uint8,
        )
        session = _Session(
            [
                {'boxes': [[0.1, 0.2, 0.7, 0.9]], 'logits': [[0.2, 0.75]], 'phrases': ['lamp']},
                {'cropped_mask': base64.b64encode(mask.tobytes()).decode()},
            ]
        )
        config = OpenVocabularyPerceptionConfig(request_timeout=0.125)
        backend = AgentVLMBackend(config, session=session)

        parsed = backend.detect(self.rgba[..., :3], 'lamp .')
        parsed_mask = backend.segment(self.rgba[..., :3], [0, 0, 2, 2])

        self.assertEqual(parsed, [{'bbox': [0.1, 0.2, 0.7, 0.9], 'confidence': 0.75, 'label': 'lamp'}])
        np.testing.assert_array_equal(parsed_mask, mask.astype(bool))
        self.assertEqual([call[1]['timeout'] for call in session.calls], [0.125, 0.125])
        self.assertEqual(session.calls[0][0], 'http://localhost:12181/gdino')
        self.assertEqual(session.calls[1][0], 'http://localhost:12183/mobile_sam')

    def test_repeated_query_is_rate_limited(self):
        now = [10.0]
        detector_calls = []

        def detector(image, query):
            detector_calls.append(query)
            return []

        perception = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(query_interval=1.0),
            detector=detector,
            segmenter=lambda image, bbox: np.zeros(image.shape[:2], dtype=bool),
            embedder=lambda crop: np.ones(2),
            clock=lambda: now[0],
        )
        perception.perceive(self.rgba, self.depth, self.point_image, 'mug')
        now[0] += 0.5
        skipped = perception.perceive(self.rgba, self.depth, self.point_image, 'mug')
        now[0] += 0.6
        perception.perceive(self.rgba, self.depth, self.point_image, 'mug')

        self.assertEqual(len(detector_calls), 2)
        self.assertEqual(skipped, [])

    def test_rate_limit_does_not_replay_stale_world_position(self):
        now = [10.0]
        mask = np.ones((3, 4), dtype=bool)
        perception = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(query_interval=1.0),
            detector=lambda image, query: [
                {'label': 'chair', 'confidence': 0.9, 'bbox': [0, 0, 4, 3]}
            ],
            segmenter=lambda image, bbox: mask,
            embedder=lambda crop: np.ones(2),
            clock=lambda: now[0],
        )

        first = perception.perceive(
            self.rgba,
            self.depth,
            self.point_image,
            'chair',
            step=4,
        )
        now[0] += 0.5
        moved_points = self.point_image + np.asarray((20.0, 0.0, 0.0))
        second = perception.perceive(
            self.rgba,
            self.depth,
            moved_points,
            'chair',
            step=5,
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

    def test_service_failure_without_fallback_is_observable(self):
        perception = OpenVocabularyPerception(
            detector=lambda _image, _query: (_ for _ in ()).throw(requests.Timeout('offline')),
            segmenter=lambda image, bbox: np.ones(image.shape[:2], dtype=bool),
            embedder=lambda crop: np.ones(2),
        )

        with self.assertRaises(requests.Timeout):
            perception.perceive(self.rgba, self.depth, self.point_image, 'door')


if __name__ == '__main__':
    unittest.main()
