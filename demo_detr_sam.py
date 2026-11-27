"""One-day demo: pretrained DETR + SAM on Indian road photos, hybrid pseudo-unknown rule.

Install (Colab):  pip install transformers torch pillow numpy timm
Run:              python demo_detr_sam.py --images my_photos --out results

For each photo it draws
  GREEN  = known object (DETR confident, COCO class)
  RED    = pseudo-unknown (objectness high + SAM agrees + not a confident known object)
DETR objectness of a query = 1 - P(no-object).
"""
import argparse, glob, os
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import DetrForObjectDetection, DetrImageProcessor, pipeline
from hybrid_proposals import hybrid_pseudo_unknown


def mask_to_box(m):
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def run(img_path, processor, detr, sam, dev, a):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((a.max_side, a.max_side))   # big photos make SAM run out of GPU memory
    W, H = img.size

    # ---- DETR: every query gives a box + class probabilities (+ a "no-object" slot) ----
    inp = processor(images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = detr(**inp)
    prob = out.logits.softmax(-1)[0]                  # (Q, 92)
    objectness = (1.0 - prob[:, -1]).cpu().numpy()    # 1 - P(no-object)
    cls_prob, cls_id = prob[:, :-1].max(-1)
    cls_prob, cls_id = cls_prob.cpu().numpy(), cls_id.cpu().numpy()
    cx, cy, w, h = out.pred_boxes[0].cpu().numpy().T
    boxes = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1)

    known_idx = np.where(cls_prob >= a.known_thr)[0]  # confident known objects
    known_boxes = boxes[known_idx]

    # ---- SAM: class-agnostic regions from the image itself ----
    masks = sam(img, points_per_batch=32)["masks"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    regions = [b for b in (mask_to_box(np.asarray(m)) for m in masks) if b is not None]

    # ---- Hybrid rule ----
    unk = hybrid_pseudo_unknown(boxes, objectness, np.array(regions), known_boxes,
                                tau_o=a.tau_o, tau_a=a.tau_a, tau_g=a.tau_g, top_k=a.top_k)

    d = ImageDraw.Draw(img)
    for i in known_idx:
        d.rectangle(list(boxes[i]), outline="lime", width=3)
        d.text((boxes[i][0] + 3, boxes[i][1] + 3), detr.config.id2label.get(int(cls_id[i]), "?"), fill="lime")
    for i in unk:
        d.rectangle(list(boxes[i]), outline="red", width=3)
        d.text((boxes[i][0] + 3, boxes[i][1] + 3), "unknown?", fill="red")
    return img, len(known_idx), len(unk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--known-thr", dest="known_thr", type=float, default=0.7)
    p.add_argument("--tau-o", dest="tau_o", type=float, default=0.3)
    p.add_argument("--tau-a", dest="tau_a", type=float, default=0.5)
    p.add_argument("--tau-g", dest="tau_g", type=float, default=0.3)
    p.add_argument("--top-k", dest="top_k", type=int, default=5)
    p.add_argument("--max-side", dest="max_side", type=int, default=1024)
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(dev).eval()
    sam = pipeline("mask-generation", model="facebook/sam-vit-base", device=0 if dev == "cuda" else -1)

    os.makedirs(a.out, exist_ok=True)
    files = sorted(sum([glob.glob(os.path.join(a.images, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []))
    for f in files:
        try:
            img, nk, nu = run(f, processor, detr, sam, dev, a)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("%s  SKIPPED (GPU memory full, try --max-side 640)" % os.path.basename(f))
            continue
        img.save(os.path.join(a.out, os.path.basename(f)))
        print("%s  known=%d  pseudo-unknown=%d" % (os.path.basename(f), nk, nu))


if __name__ == "__main__":
    main()
