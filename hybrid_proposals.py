"""Hybrid pseudo-unknown selection (our proposed contribution).

A candidate box b (from the detector) becomes a "pseudo-unknown" when:
  1. objectness s(b)                       >= tau_o   (detector thinks "something is here")
  2. agreement a(b) = max_r IoU(b, r)      >= tau_a   (an input-driven region r, e.g. a SAM mask box,
                                                       or selective-search box, also covers it)
  3. max_g IoU(b, g)                       <  tau_g   (it is not an already-known labelled object)

Pure numpy, no GPU needed. Boxes are [x1, y1, x2, y2].
"""
import numpy as np


def iou_matrix(a, b):
    """IoU between every box in a (N,4) and every box in b (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes, scores, iou_thr=0.5):
    """Plain non-maximum suppression; returns kept indices."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores))
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return keep


def hybrid_pseudo_unknown(boxes, objectness, region_boxes, known_gt_boxes,
                          tau_o=0.5, tau_a=0.5, tau_g=0.3, top_k=5, nms_iou=0.5):
    """Return indices (into `boxes`) chosen as pseudo-unknown, best first.

    boxes          (N,4) candidate boxes from the detector
    objectness     (N,)  objectness score of each candidate
    region_boxes   (R,4) input-driven regions (SAM / selective search)
    known_gt_boxes (K,4) boxes of objects that are already labelled / confidently known
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    objectness = np.asarray(objectness, dtype=float)
    if len(boxes) == 0:
        return []
    agree = iou_matrix(boxes, region_boxes).max(axis=1) if len(region_boxes) else np.zeros(len(boxes))
    known = iou_matrix(boxes, known_gt_boxes).max(axis=1) if len(known_gt_boxes) else np.zeros(len(boxes))
    ok = (objectness >= tau_o) & (agree >= tau_a) & (known < tau_g)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return []
    kept = nms(boxes[idx], objectness[idx], nms_iou)
    return [int(idx[k]) for k in kept][:top_k]
