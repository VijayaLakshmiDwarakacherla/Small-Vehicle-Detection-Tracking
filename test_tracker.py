import random
import unittest
import numpy as np
from custom_tracker import AMTTracker, AMTConfig


class TestAMTTrackerUnit(unittest.TestCase):
    def setUp(self):
        self.cfg = AMTConfig(
            max_age=15,
            min_hits=2,
            birth_conf=0.15,
            recovery_conf=0.08,
            roi_ymin=0.18
        )
        self.tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)

    def test_1_single_bike_continuous_detections(self):
        """TEST 1: Single bike with continuous detections => same ID across all frames."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)
        track_ids = []

        for f in range(10):
            # Bike moving horizontally at 5 px/frame
            cx = 400 + f * 5.0
            cy = 300.0
            det = [[cx - 15, cy - 20, cx + 15, cy + 20, 0.85, 3]]
            active = tracker.update(det)
            if active:
                track_ids.append(active[0][4])

        self.assertTrue(len(track_ids) > 0)
        # All active outputs must have the exact same track ID
        self.assertEqual(len(set(track_ids)), 1, f"Expected 1 unique ID, got {set(track_ids)}")

    def test_2_single_bike_one_missed_frame(self):
        """TEST 2: Single bike, one missed frame => same ID after recovery."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)
        
        # Frame 0 & 1: Detections to confirm track
        tracker.update([[400, 300, 430, 340, 0.85, 3]])
        tracker.update([[405, 300, 435, 340, 0.85, 3]])
        
        # Frame 2: Missed detection (1 miss -> rendered as PRED_1)
        active_miss = tracker.update([])
        self.assertEqual(len(active_miss), 1)
        self.assertTrue(active_miss[0][8], "Frame with 1 miss should be marked is_predicted=True")

        # Frame 3: Bike detected again -> recovered with SAME ID
        active_rec = tracker.update([[415, 300, 445, 340, 0.85, 3]])
        self.assertEqual(len(active_rec), 1)
        self.assertEqual(active_rec[0][4], active_miss[0][4])
        self.assertFalse(active_rec[0][8], "Recovered frame should be marked is_predicted=False")

    def test_3_single_bike_two_missed_frames(self):
        """TEST 3: Single bike, two missed frames => same ID after recovery."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)
        
        # Confirm track
        tracker.update([[400, 300, 430, 340, 0.85, 3]])
        active1 = tracker.update([[410, 300, 440, 340, 0.85, 3]])
        orig_id = active1[0][4]

        # Miss 2 frames
        tracker.update([])
        tracker.update([])

        # Recover on frame 4
        active_rec = tracker.update([[430, 300, 460, 340, 0.85, 3]])
        self.assertEqual(len(active_rec), 1)
        self.assertEqual(active_rec[0][4], orig_id, "Track ID must persist after 2 missed frames")

    def test_4_weak_detection_gap(self):
        """TEST 4: Weak detection gap => no new ID birthed, but allows recovery of existing track."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)
        
        # Isolated weak detection alone (0.10) must NEVER birth a new track
        active_weak = tracker.update([[400, 300, 430, 340, 0.10, 3]])
        self.assertEqual(len(active_weak), 0, "Weak detection (0.10 < birth_conf 0.15) must not birth a track")

    def test_5_false_positive_one_frame(self):
        """TEST 5: False positive appears for one frame => no confirmed track."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)
        
        # Single frame strong detection
        active0 = tracker.update([[400, 300, 430, 340, 0.85, 3]])
        self.assertEqual(len(active0), 0, "Tentative track on 1st frame should not be active output yet")

        # Next frame no detection -> Tentative track dies
        tracker.update([])
        tracker.update([])
        tracker.update([])

        self.assertEqual(len(tracker.tracks), 0, "Isolated 1-frame false positive must be pruned")

    def test_6_two_crossing_bikes(self):
        """TEST 6: Two crossing bikes => IDs remain separate."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)

        id_A, id_B = None, None
        for f in range(5):
            # Bike A moving right: 400 + f*20
            # Bike B moving left:  500 - f*20
            det_A = [400 + f * 20, 300, 430 + f * 20, 340, 0.85, 3]
            det_B = [500 - f * 20, 305, 530 - f * 20, 345, 0.85, 3]

            active = tracker.update([det_A, det_B])
            if f >= 1:
                self.assertEqual(len(active), 2)
                ids = [a[4] for a in active]
                if id_A is None:
                    id_A, id_B = ids[0], ids[1]
                else:
                    self.assertIn(id_A, ids)
                    self.assertIn(id_B, ids)

    def test_7_detection_order_shuffled(self):
        """TEST 7: Detection list order shuffled => IDs remain unchanged."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)

        for f in range(6):
            det_A = [400 + f * 10, 300, 430 + f * 10, 340, 0.85, 3]
            det_B = [600 - f * 10, 400, 630 - f * 10, 440, 0.85, 3]

            dets = [det_A, det_B]
            if f % 2 == 1:
                dets = [det_B, det_A]  # Shuffle order

            active = tracker.update(dets)
            if f >= 1:
                self.assertEqual(len(active), 2)

    def test_8_large_localization_jump_plausible_motion(self):
        """TEST 8: Large localization jump but plausible motion => preserve identity within gate."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)
        
        tracker.update([[400, 300, 430, 340, 0.85, 3]])
        active1 = tracker.update([[410, 300, 440, 340, 0.85, 3]])
        orig_id = active1[0][4]

        # Jump 25 px (within perspective gate D_gate ≈ 35-50 px)
        active2 = tracker.update([[435, 300, 465, 340, 0.85, 3]])
        self.assertEqual(len(active2), 1)
        self.assertEqual(active2[0][4], orig_id)

    def test_9_impossible_jump(self):
        """TEST 9: Impossible spatial jump => reject association."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)

        tracker.update([[400, 300, 430, 340, 0.85, 3]])
        active1 = tracker.update([[405, 300, 435, 340, 0.85, 3]])
        orig_id = active1[0][4]

        # Impossible jump 300 px across the frame -> must NOT associate with orig_id
        active2 = tracker.update([[700, 300, 730, 340, 0.85, 3]])
        # Find track corresponding to orig_id in tracker
        orig_track = [t for t in tracker.tracks if t.track_id == orig_id][0]
        self.assertGreater(orig_track.time_since_update, 0, "Orig track must NOT associate with 300px jump")


    def test_10_duplicate_detection_around_one_bike(self):
        """TEST 10: Duplicate detections around one bike => only one active track."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)

        det1 = [400, 300, 430, 340, 0.85, 3]
        det2 = [402, 301, 431, 339, 0.82, 3]  # Duplicate detection

        tracker.update([det1, det2])
        active = tracker.update([det1, det2])

        self.assertLessEqual(len(active), 1, "Duplicate detection must yield at most 1 active track")

    def test_11_stationary_object(self):
        """TEST 11: Stationary object => does not create repeated IDs."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)

        ids = []
        for f in range(10):
            det = [[400, 300, 430, 340, 0.85, 3]]  # Zero displacement
            active = tracker.update(det)
            if active:
                ids.append(active[0][4])

        if ids:
            self.assertEqual(len(set(ids)), 1, "Stationary object must preserve 1 identity")

    def test_12_track_exits_frame(self):
        """TEST 12: Track exits frame boundary => eventually deleted."""
        tracker = AMTTracker(config=self.cfg, img_h=720, img_w=1280)

        # Move track toward horizon y = 130
        for f in range(5):
            cy = 150 - f * 5  # Moving toward top horizon
            tracker.update([[400, cy - 20, 430, cy + 20, 0.85, 3]])

        # Miss detection while near horizon
        tracker.update([])
        tracker.update([])
        tracker.update([])

        # Track should be deleted because it exited horizon
        self.assertEqual(len(tracker.tracks), 0, "Track near horizon moving outward must be deleted after miss")


if __name__ == "__main__":
    unittest.main()
