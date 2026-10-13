"""Geometry, instance identity and distinct-view regressions on CPU."""
import unittest
from types import SimpleNamespace

import numpy as np

from grutopia_extension.interactive_navigation.object_observations import (
    MultiViewMemory, robust_object_geometry,
)
from grutopia_extension.interactive_navigation.open_vocabulary_perception import (
    OpenVocabularyPerception, OpenVocabularyPerceptionConfig, TargetObservationCue,
)
from grutopia_extension.interactive_navigation.mapping import (
    MappingConfig, OccupancyGridMap, SceneGraphMap, SemanticDetection,
)
from grutopia_extension.interactive_navigation.mapping_runtime import _camera_cloud, _deduplicate_semantic_detections
from grutopia_extension.interactive_navigation.semantic_exploration_component import (
    SemanticExplorationComponent, SemanticExplorationConfig,
)


class ObjectObservationTest(unittest.TestCase):
    def geometry(self, center=(2., 2., .5)):
        return {'centroid': center, 'bounds': ((center[0]-.1, center[1]-.1, .4),
                                            (center[0]+.1, center[1]+.1, .6)), 'quality': 1.}

    def test_depth_background_does_not_pull_object_position(self):
        mask = np.ones((10, 10), bool)
        depth = np.full((10, 10), 2.)
        points = np.full((10, 10, 3), (2., 0., .5))
        depth[:2] = 8.
        points[:2] = (8., 0., .5)
        clean, retained, info = robust_object_geometry(mask, depth, points)
        self.assertEqual(len(retained), 80)
        self.assertEqual(info['centroid'], (2., 0., .5))
        self.assertFalse(clean[:2].any())
        self.assertEqual(info['rejected_points'], 20)

    def test_two_equal_depth_layers_abstain(self):
        depth = np.ones((8, 8)); depth[4:] = 5.
        with self.assertRaisesRegex(ValueError, 'ambiguous_depth'):
            robust_object_geometry(np.ones((8, 8), bool), depth, np.ones((8, 8, 3)))

    def test_cloud_mismatch_is_not_silently_truncated(self):
        camera = {'rgba': np.ones((2, 2, 4)), 'depth': np.ones((2, 2)),
                  'pointcloud': np.ones((3, 3))}
        with self.assertRaisesRegex(ValueError, 'exactly'):
            _camera_cloud(camera)

    def test_invalid_world_point_keeps_its_original_pixel(self):
        points = np.array([[1.,0,0], [np.nan,0,0], [3.,0,0], [4.,0,0]])
        cloud, _, pixels = _camera_cloud({'rgba': np.ones((2, 2, 4)),
                                         'depth': np.ones((2, 2)), 'pointcloud': points})
        self.assertEqual(len(cloud), 3)
        self.assertTrue(np.isnan(pixels[0, 1, 0]))
        self.assertEqual(pixels[1, 0, 0], 3.)

    def test_reference_images_require_distinct_positions_and_expire(self):
        bank = MultiViewMemory(max_age=10)
        rgb = np.full((8, 8, 3), 100, np.uint8); mask = np.ones((8, 8), bool)
        first, refs, _ = bank.observe(rgb, rgb, mask, self.geometry(), (0.,2.,.5), 0)
        _, refs, _ = bank.observe(rgb, rgb, mask, self.geometry(), (.1,2.,.5), 1)
        self.assertEqual(refs, [])
        same, refs, metadata = bank.observe(rgb, rgb, mask, self.geometry(), (2.,0.,.5), 2)
        self.assertEqual(same, first)
        self.assertEqual(len(refs), 1)
        self.assertEqual(metadata[0]['step'], 1)
        new, refs, _ = bank.observe(rgb, rgb, mask, self.geometry(), (0.,2.,.5), 20)
        self.assertEqual(new, first)  # Expired photos must not duplicate a persistent object.
        self.assertEqual(refs, [])

    def test_two_close_chairs_cannot_share_reference_images(self):
        bank = MultiViewMemory()
        rgb = np.full((8, 8, 3), 100, np.uint8); mask = np.ones((8, 8), bool)
        a, _, _ = bank.observe(rgb, rgb, mask, self.geometry(), (0.,2.,.5), 0)
        b, refs, _ = bank.observe(rgb, rgb, mask, self.geometry((2.5,2.,.5)), (2.,0.,.5), 1)
        self.assertNotEqual(a, b); self.assertEqual(refs, [])

    def test_view_memory_is_bounded(self):
        bank = MultiViewMemory(capacity=2)
        rgb = np.zeros((8,8,3),np.uint8)
        for i in range(5):
            bank.observe(rgb, rgb, np.ones((8,8),bool), self.geometry((i*2.,2.,.5)), (0.,0.,.5), i)
        self.assertEqual(len(bank.objects), 2)

    def test_different_side_colors_do_not_break_strong_geometry_identity(self):
        bank=MultiViewMemory(); mask=np.ones((8,8),bool)
        back=np.full((8,8,3),240,np.uint8); front=np.full((8,8,3),20,np.uint8)
        a,_,_=bank.observe(back,back,mask,self.geometry(),(0.,2.,.5),0)
        b,refs,_=bank.observe(front,front,mask,self.geometry(),(2.,0.,.5),1)
        self.assertEqual(a,b);self.assertEqual(len(refs),1)

    def test_two_proposals_in_one_capture_cannot_claim_one_instance(self):
        bank = MultiViewMemory()
        rgb = np.zeros((8,8,3),np.uint8); mask = np.ones((8,8),bool)
        a, _, _ = bank.observe(rgb,rgb,mask,self.geometry(),(0.,0.,.5),0)
        b, _, _ = bank.observe(rgb,rgb,mask,self.geometry(),(0.,0.,.5),0)
        self.assertNotEqual(a,b)

    def test_instance_labels_can_be_corrected_without_duplicate_objects(self):
        graph = SceneGraphMap(MappingConfig())
        for step, label in enumerate(('bed', 'sofa', 'sofa')):
            graph.update_detections([SemanticDetection(
                label, (2.,2.,.5), step=step, sources=('qwen_vl',), instance_id='one',
                bounds=self.geometry()['bounds'], goal_query='sleeping furniture', goal_match_score=.95)])
        self.assertEqual(len(graph.object_nodes()), 1)
        node = graph.object_nodes()[0]
        self.assertEqual(node.label, 'sofa')
        self.assertEqual(graph.goal_match_counts(node.node_id, 'sleeping furniture', .8), (1,1))

    def test_instance_ids_prevent_same_frame_and_map_false_merges(self):
        a=SemanticDetection('chair',(2.,2.,.5),instance_id='a',step=1)
        b=SemanticDetection('chair',(2.4,2.,.5),instance_id='b',step=1)
        self.assertEqual(len(_deduplicate_semantic_detections([a,b])),2)
        graph=SceneGraphMap(MappingConfig()); graph.update_detections([a,b])
        self.assertEqual(len(graph.object_nodes()),2)

    def test_evicted_view_hint_can_reassociate_by_strong_geometry(self):
        graph=SceneGraphMap(MappingConfig())
        for step, hint in ((1,'old'),(4,'new')):
            graph.update_detections([SemanticDetection('chair',(2.,2.,.5),instance_id=hint,
                bounds=self.geometry()['bounds'],step=step,sources=('qwen_vl',))])
        self.assertEqual(len(graph.object_nodes()),1)

    def test_old_capture_is_not_a_new_observation(self):
        classifier=SimpleNamespace(classify=lambda *a,**kw: {'label':'chair','model':'test'})
        p=OpenVocabularyPerception(OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl',query_interval=0,multi_view=True),
            detector=lambda *a:[('chair',.9,[0,0,8,8])],
            segmenter=lambda *a:np.ones((8,8),bool),classifier=classifier)
        rgb=np.full((8,8,3),100,np.uint8); points=np.full((8,8,3),(2.,2.,.5))
        self.assertEqual(len(p.perceive(rgb,np.ones((8,8)),points,'chair',step=2,capture_step=1)),1)
        self.assertEqual(p.perceive(rgb,np.ones((8,8)),points,'chair',step=3,capture_step=1),[])
        self.assertEqual(p.last_query_status,'duplicate_capture')

    def test_same_angle_positive_votes_do_not_unlock_multi_view_confirmation(self):
        c=SemanticExplorationComponent(SemanticExplorationConfig(target_query='chair',
            target_semantic_classifier='qwen-vl',target_min_views=2))
        graph=c.mapping.map.scene_graph
        for step in range(8):
            graph.update_detections([SemanticDetection('chair',(2.,0.,.5),step=step,sources=('qwen_vl',))])
        c.target_node=graph.object_nodes()[0];c._target_lexical=True
        self.assertFalse(c.target_confirmed)
        graph.update_detections([SemanticDetection('chair',(2.,0.,.5),step=8,sources=('qwen_vl',),view_support=2)])
        c.target_node=graph.object_nodes()[0]
        self.assertTrue(c.target_confirmed)

    def test_negative_multi_view_description_cannot_unlock_old_single_view_scores(self):
        query='a red chair'
        c=SemanticExplorationComponent(SemanticExplorationConfig(target_query=query,
            target_semantic_classifier='qwen-vl',target_min_views=2))
        graph=c.mapping.map.scene_graph
        for step in range(8):
            graph.update_detections([SemanticDetection('chair',(2.,0.,.5),step=step,sources=('qwen_vl',),
                goal_query=query,goal_match_score=.95,goal_view_support=1)])
        graph.update_detections([SemanticDetection('chair',(2.,0.,.5),step=8,sources=('qwen_vl',),
            goal_query=query,goal_match_score=0.,view_support=2,goal_view_support=0)])
        c.target_node=graph.object_nodes()[0];c._target_lexical=True
        self.assertFalse(c.target_confirmed)

    def test_overlapping_boxes_without_surface_overlap_do_not_share_views(self):
        bank=MultiViewMemory();rgb=np.zeros((8,8,3),np.uint8);mask=np.ones((8,8),bool)
        left=self.geometry();right=self.geometry()
        left['surface_points']=np.array([[2.,2.,.4],[2.,2.,.41]])
        right['surface_points']=np.array([[2.,2.,.59],[2.,2.,.6]])
        a,_,_=bank.observe(rgb,rgb,mask,left,(0.,2.,.5),0)
        b,refs,_=bank.observe(rgb,rgb,mask,right,(2.,0.,.5),1)
        self.assertNotEqual(a,b);self.assertEqual(refs,[])

    def test_perception_passes_other_angle_to_classifier(self):
        calls=[]
        def classify(*args,**kwargs):
            calls.append(kwargs); return {'label':'chair','model':'test'}
        p=OpenVocabularyPerception(OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl',query_interval=0,multi_view=True),
            detector=lambda *a:[('chair',.9,[0,0,8,8])],
            segmenter=lambda *a:np.ones((8,8),bool), classifier=SimpleNamespace(classify=classify))
        rgb=np.full((8,8,3),100,np.uint8); points=np.full((8,8,3),(2.,2.,.5))
        for step,pose in enumerate(((0.,2.,.5),(2.,0.,.5))):
            p.perceive(rgb,np.ones((8,8)),points,'chair',step=step,camera_position=pose,capture_step=step)
        self.assertNotIn('reference_images',calls[0])
        self.assertEqual(len(calls[1]['reference_images']),1)

    def test_service_cannot_claim_unrequested_views_as_goal_evidence(self):
        classifier = SimpleNamespace(classify=lambda *a, **kw: {
            'label': 'chair', 'model': 'test', 'view_count': 3, 'goal_match_score': .95})
        pipeline = OpenVocabularyPerception(
            OpenVocabularyPerceptionConfig(semantic_classifier='qwen-vl', query_interval=0),
            detector=lambda *a: [('chair', .9, [0, 0, 8, 8])],
            segmenter=lambda *a: np.ones((8, 8), bool), classifier=classifier)
        result = pipeline.perceive(np.ones((8, 8, 3), np.uint8), np.ones((8, 8)),
                                   np.ones((8, 8, 3)), 'a red chair')
        self.assertEqual(result[0].view_support, 1)
        self.assertEqual(result[0].goal_view_support, 1)
