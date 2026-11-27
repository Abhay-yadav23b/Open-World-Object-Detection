# Open World Object Detection (OWOD) – Implementation

Application domain: Indian roads (India Driving Dataset, IDD). Method idea: Hybrid Pseudo-Unknown Proposals.

## Files
| File | Kaam |
|---|---|
| `demo_detr_sam.py` | 1-din ka demo: pretrained DETR + SAM on Indian road photos, known = green, pseudo-unknown = red |
| `hybrid_proposals.py` | Hybrid rule (objectness + SAM agreement + not-known) |
| `metrics.py` | U-Recall, U-Precision, U-F1, Wilderness Impact, A-OSE, mAP, forgetting |
| `split_idd.py` | IDD (Pascal VOC XML) -> Task 1 / Task 2 splits (COCO json) |
| `test_pure.py` | Synthetic tests for numpy code (passed) |

## Option A – One day (recommended)
1. Colab -> Runtime -> GPU.
2. `pip install transformers timm pillow numpy` and upload this folder.
3. Put 15-30 Indian road photos (cows, e-rickshaws, handcarts, potholes...) in `photos/`.
4. `python demo_detr_sam.py --images photos --out results`
5. Tune `--tau-o`, `--tau-a`, `--tau-g`. Show: (i) where DETR misses / mislabels Indian objects, (ii) red boxes the hybrid rule adds.
Do NOT claim numbers (U-Recall etc.) from this demo – it has no ground truth for unknowns.

## Option B – Measured numbers (needs more days)
1. `python split_idd.py --root IDD_Detection --list-classes` -> check real class names, edit `DEFAULT_TASKS` or pass `--tasks tasks.json`.
2. `python split_idd.py --root IDD_Detection --out splits --max-images 3000`
3. Train PROB (github.com/orrzohar/PROB) on `train_t1`, then fine-tune on `train_t2` with exemplar replay. PROB needs its own data layout and compiled CUDA ops – adapt the loader to the COCO json from step 2.
4. Dump detections in the format described in `metrics.py` and compute U-Recall / WI / A-OSE with it. Forgetting = Task-1 mAP before minus after Task 2.

## Honest status
- Tested here: `hybrid_proposals.py`, `metrics.py`, `split_idd.py` (synthetic data).
- Only syntax-checked (torch/transformers unavailable here): `demo_detr_sam.py`. Model download names (`facebook/detr-resnet-50`, `facebook/sam-vit-base`) must work on your Colab; if the SAM pipeline errors, update `transformers`.
- IDD class names in `DEFAULT_TASKS` are my recollection – verify them.
- Option B (PROB training) is not written/tested.
