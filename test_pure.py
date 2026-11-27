"""Synthetic tests for the numpy modules.  Run: python test_pure.py"""
import numpy as np
from hybrid_proposals import iou_matrix, nms, hybrid_pseudo_unknown
import metrics as M

# --- IoU ---
assert abs(iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] - 1) < 1e-9
assert abs(iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] - 1 / 3) < 1e-9
assert iou_matrix([[0, 0, 1, 1]], [[5, 5, 6, 6]])[0, 0] == 0

# --- NMS ---
assert nms([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], [0.9, 0.8, 0.7]) == [0, 2]

# --- hybrid rule ---
boxes = [[0, 0, 10, 10],     # known object
         [100, 100, 150, 150],  # real unknown, SAM agrees
         [200, 200, 220, 220],  # low objectness
         [300, 300, 340, 340]]  # high objectness but SAM has nothing there
obj = [0.9, 0.8, 0.1, 0.9]
regions = [[100, 100, 148, 150], [0, 0, 10, 10], [200, 200, 220, 220]]
known = [[0, 0, 10, 10]]
assert hybrid_pseudo_unknown(boxes, obj, regions, known) == [1]
assert hybrid_pseudo_unknown([], [], regions, known) == []

# --- metrics ---
B = lambda x: [x, x, x + 10, x + 10]
gts = [[(B(0), "car"), (B(50), "unknown"), (B(100), "unknown")]]
# perfect open-world detector
perfect = [[(B(0), .9, "car"), (B(50), .8, "unknown"), (B(100), .7, "unknown")]]
r = M.unknown_scores(perfect, gts)
assert r["U-Recall"] == 1 and r["U-Precision"] == 1 and r["U-F1"] == 1
assert M.a_ose(perfect, gts) == 0
assert abs(M.wilderness_impact(perfect, gts)) < 1e-9
# closed-world detector: calls both unknowns "car"
closed = [[(B(0), .6, "car"), (B(50), .8, "car"), (B(100), .7, "car")]]
assert M.a_ose(closed, gts) == 2
assert M.unknown_scores(closed, gts)["U-Recall"] == 0
assert M.wilderness_impact(closed, gts) > 0.4, M.wilderness_impact(closed, gts)  # P_K=1, P_KU=1/3 -> 2
# half-recall detector
half = [[(B(0), .9, "unknown"), (B(50), .8, "unknown")]]
r = M.unknown_scores(half, gts)
assert abs(r["U-Recall"] - 0.5) < 1e-9 and abs(r["U-Precision"] - 0.5) < 1e-9
# mAP
assert abs(M.mean_ap(perfect, gts, ["car"]) - 1) < 1e-9
assert M.forgetting(60.0, 45.0) == 15.0
print("ALL TESTS PASSED")
