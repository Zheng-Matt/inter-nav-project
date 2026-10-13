"""Exercise the real HTTP handler with deterministic model output on CPU."""

import base64
import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from grutopia.demo.serve_semantic_perception import _qwen_vl_app


class GoalVerificationServiceTest(unittest.TestCase):
    def setUp(self):
        self.raw = '{"observed_object":"a white refrigerator","label":"refrigerator","goal_match_score":0.95}'
        self.outputs = []
        self.prompt = ''
        self.prompts = []
        self.image_counts = []
        self.budgets = []
        outer = self

        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                outer.prompt = messages[0]['content'][-1]['text']
                outer.prompts.append(outer.prompt)
                return outer.prompt

            def __call__(self, **kwargs):
                outer.image_counts.append(len(kwargs['images']))
                return Inputs(input_ids=np.zeros((1, 2)))

            def batch_decode(self, *args, **kwargs):
                return [outer.outputs.pop(0) if outer.outputs else outer.raw]

        class Model:
            device = 'cpu'

            def to(self, device):
                return self

            def eval(self):
                pass

            def generate(self, **kwargs):
                outer.budgets.append(kwargs['max_new_tokens'])
                return np.zeros((1, 4))

        modules = {
            'torch': SimpleNamespace(inference_mode=contextlib.nullcontext),
            'transformers': SimpleNamespace(
                AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **kw: Processor()),
                Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=lambda *a, **kw: Model()),
            ),
        }
        with patch.dict(sys.modules, modules):
            self.client = _qwen_vl_app(
                SimpleNamespace(
                    vl_max_new_tokens=32,
                    vl_max_pixels=384 * 384,
                    qwen_vl_model='test-double',
                    device='cpu',
                )
            ).test_client()
        _, data = cv2.imencode('.jpg', np.zeros((64, 64, 3), np.uint8))
        self.payload = {'image': base64.b64encode(data).decode(), 'labels': ['refrigerator', 'cabinet', 'unknown']}

    def test_description_response_and_generation_budget(self):
        response = self.client.post('/classify', json=dict(self.payload, target_query='white fridge'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['goal_match_score'], 0.95)
        self.assertIn('ALL stated visual attributes', self.prompt)
        self.assertIn('white fridge', self.prompt)
        self.assertIn('observed_object FIRST', self.prompt)
        self.assertNotIn('white fridge', self.prompts[0])
        self.assertEqual(self.budgets, [32, 128])
        self.assertEqual(response.json['goal_verification_mode'], 'category_then_goal')
        self.assertTrue(self.client.get('/health').json['goal_verification'])

    def test_category_only_keeps_original_budget_and_response(self):
        response = self.client.post('/classify', json=self.payload)
        self.assertEqual(response.json['label'], 'refrigerator')
        self.assertIsNone(response.json['goal_match_score'])
        self.assertEqual(self.budgets, [32])

    def test_prefilter_skips_only_goal_review_and_keeps_category(self):
        response = self.client.post('/classify', json=dict(
            self.payload, target_query='white fridge', verify_goal=False))
        self.assertEqual(response.json['label'], 'refrigerator')
        self.assertIsNone(response.json['goal_match_score'])
        self.assertEqual(response.json['goal_verification_status'], 'prefiltered')
        self.assertEqual(self.budgets, [32])
        self.assertNotIn('white fridge', self.prompts[0])

    def test_invalid_review_switch_is_rejected_before_inference(self):
        for value in ['false', 0, None]:
            response = self.client.post('/classify', json=dict(self.payload, verify_goal=value))
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.budgets, [])

    def test_multiple_views_share_query_independent_classification(self):
        result = self.client.post('/classify', json=dict(self.payload,
            target_query='white fridge', reference_images=[self.payload['image'], self.payload['image']])).json
        self.assertEqual(result['view_count'], 3)
        self.assertEqual(self.image_counts, [3, 3])
        self.assertNotIn('white fridge', self.prompts[0])
        self.assertIn('same tracked object', self.prompts[0])

    def test_bad_or_excess_reference_images_are_rejected_before_inference(self):
        for images in ('encoded', [self.payload['image']]*3, ['bad'], [None]):
            response = self.client.post('/classify', json=dict(self.payload, reference_images=images))
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.budgets, [])

    def test_invalid_unknown_or_abstained_outputs_never_confirm(self):
        for raw in [
            '{"label":"refrigerator"}',
            '{"label":"refrigerator","goal_match_score":true}',
            '{"label":"refrigerator","goal_match_score":NaN}',
            '{"label":"unknown","goal_match_score":0.99}',
            '{"label":"other","goal_match_score":0.99}',
            'not JSON',
        ]:
            with self.subTest(raw=raw):
                self.raw = raw
                result = self.client.post('/classify', json=dict(self.payload, target_query='white fridge')).json
                self.assertIsNone(result['goal_match_score'])

    def test_invalid_queries_are_rejected_before_inference(self):
        for query in ['', 'a' * 513, [], 7]:
            self.assertEqual(
                self.client.post('/classify', json=dict(self.payload, target_query=query)).status_code, 400
            )
        self.assertEqual(self.budgets, [])

    def test_verification_cannot_reclassify_object_to_fit_goal(self):
        self.outputs = ['{"label":"refrigerator"}',
                        '{"observed_object":"a white cabinet","label":"cabinet","goal_match_score":1.0}']
        result = self.client.post('/classify', json=dict(self.payload, target_query='white fridge')).json
        self.assertEqual(result['label'], 'refrigerator')
        self.assertIsNone(result['goal_match_score'])
        self.assertEqual(result['goal_verification_status'], 'category_conflict')

    def test_explicit_category_conflict_rejects_high_score_before_review(self):
        self.outputs = ['{"label":"bed"}']
        payload = dict(self.payload, labels=['sofa', 'bed', 'unknown'], target_query='a gray sofa')
        result = self.client.post('/classify', json=payload).json
        self.assertIsNone(result['goal_match_score'])
        self.assertEqual(result['goal_verification_status'], 'category_mismatch')
        self.assertEqual(self.budgets, [32])

    def test_observed_object_and_label_cannot_contradict(self):
        self.outputs = ['{"label":"sofa"}',
                        '{"observed_object":"a gray bed","label":"sofa","goal_match_score":1.0}']
        payload = dict(self.payload, labels=['sofa', 'bed', 'unknown'], target_query='furniture used for sleeping')
        result = self.client.post('/classify', json=payload).json
        self.assertIsNone(result['goal_match_score'])
        self.assertEqual(result['goal_verification_status'], 'category_conflict')


if __name__ == '__main__':
    unittest.main()
