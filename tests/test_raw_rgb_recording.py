import tempfile
import unittest

import cv2
import numpy as np

from grutopia_extension.interactive_navigation.navigation_recording import write_raw_rgb_frame


class RawRGBRecordingTest(unittest.TestCase):
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
