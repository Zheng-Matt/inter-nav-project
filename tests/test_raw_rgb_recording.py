import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from grutopia_extension.interactive_navigation.navigation_recording import NavigationRecordingSession, write_raw_rgb_frame


class RawRGBRecordingTest(unittest.TestCase):
    def test_aligned_replay_inputs_preserve_pixels_pose_and_capture_identity(self):
        camera = {'rgba': np.zeros((2,2,4),np.uint8), 'depth': np.ones((2,2)),
                  'pointcloud': np.arange(12).reshape(4,3), 'position': np.array((1.,2.,3.)),
                  'orientation': np.array((1.,0.,0.,0.)), 'capture_step':24}
        with tempfile.TemporaryDirectory() as directory:
            recorder = NavigationRecordingSession.__new__(NavigationRecordingSession)
            recorder.input_directory = Path(directory)/'perception_inputs'
            recorder._input_steps=set(); recorder.raw_rgb_directory=None
            recorder.recorder=SimpleNamespace(record_perception=lambda *args:None)
            for step in (24,48):
                recorder.record_perception(step,{'sensors':{'camera':camera}},SimpleNamespace(rgb_interval=24))
            files=list(recorder.input_directory.glob('*.npz'))
            self.assertEqual(len(files),1)
            with np.load(files[0],allow_pickle=False) as sample:
                np.testing.assert_array_equal(sample['rgba'],camera['rgba'])
                np.testing.assert_array_equal(sample['pointcloud'],camera['pointcloud'])
                np.testing.assert_array_equal(sample['camera_position'],camera['position'])
                self.assertEqual(int(sample['capture_step']),24)

    def test_saved_pixels_match_sensor_rgb_without_mutation(self):
        rgba = np.array([[[255, 0, 0, 255], [0, 0, 255, 128]]], dtype=np.uint8)
        original = rgba.copy()
        observation = {'sensors': {'camera': {'rgba': rgba}}}
        with tempfile.TemporaryDirectory() as directory:
            path = write_raw_rgb_frame(directory, 24, observation)
            decoded = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(decoded, original[:, :, :3])
            np.testing.assert_array_equal(rgba, original)

    def test_missing_frames_skip_and_existing_frames_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(write_raw_rgb_frame(directory, 0, {}))
            observation = {'sensors': {'camera': {'rgba': np.zeros((2, 2, 4), dtype=np.uint8)}}}
            path = write_raw_rgb_frame(directory, 24, observation)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_raw_rgb_frame(directory, 24, observation)
            self.assertEqual(path.read_bytes(), before)
