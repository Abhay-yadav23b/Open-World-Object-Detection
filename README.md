# Open World Object Detection: Hybrid Pseudo-Unknown Proposals (proof-of-concept)

 Abhay Kumar Yadav (Roll No. 2301007), IIIT Guwahati

Open World Object Detection (OWOD) asks a detector to (1) recognise known objects, (2) flag unseen objects as *unknown*, and (3) learn new classes later without forgetting old ones. Indian roads are used only as an application setting: they contain objects (auto-rickshaws, e-rickshaws, hand-carts, stray animals) that standard detectors never saw in training.

This repository contains a **small proof-of-concept** of one idea: find unknown objects by combining a detector's objectness with input-driven regions. It is not a full OWOD system.

## What was done

1. Used two **pretrained** models without training them ourselves:
   - Detection Transformer (DETR), `facebook/detr-resnet-50`, trained on Common Objects in Context (COCO).
   - Segment Anything Model (SAM), `facebook/sam-vit-base`.
2. Wrote a **hybrid rule** (`hybrid_proposals.py`). A DETR candidate box becomes a *pseudo-unknown* (red box) only if all three hold:
   1. objectness >= `tau_o`, where objectness = 1 - P(no-object)
   2. agreement with a SAM region (IoU) >= `tau_a`
   3. IoU with every known box < `tau_g`; a box is *known* when its class confidence >= `known-thr`
3. Ran it on **8 Indian road photos** collected from the internet, with two settings.

| Setting | known-thr | tau_o | tau_a | tau_g |
|---|---|---|---|---|
| Default | 0.7 | 0.3 | 0.5 | 0.3 |
| Tuned | 0.9 | 0.5 | 0.6 | 0.3 |

## Results (counted by hand)

| | Default | Tuned |
|---|---|---|
| Red boxes | 9 | 14 |
| Correct | 3 | 4 |
| Wrong | 6 | 10 |
| Correct share | ~33% | ~29% |

Output images (green = known, red = pseudo-unknown) are in `results/default` and `results/tuned`.
Photos were taken from Google Images for a class demonstration.

## Limitations (please read)

- Only 8 photos, counted by eye. There is no ground truth, so **Unknown Recall and other metrics were not computed**. Percentages above are not accuracy figures.
- Candidates come only from DETR. Where DETR makes no box, SAM alone cannot recover the object.
- If DETR confidently gives an unknown object a known name (auto-rickshaw -> "truck", tiger -> known), the rule does not flag it.
- Many false red boxes on cut-off or partial objects. Tuning gave more red boxes, not a better share.
- Cow and elephant are COCO classes, so they are not true unknowns for DETR.
- Thresholds were chosen by estimate, not tuned on data.
- Results changed when the photos were re-saved (JPEG compression), so they are sensitive to small input changes.
- No training and **no incremental learning** (exemplar replay, knowledge distillation) was run; that part is theory only.
- Not compared against PROB, OW-DETR or USD.
- Code for metrics (Unknown Recall, Wilderness Impact) and for dataset splits was written but removed, because it was not run on real data.

## Related work

The idea is **not new**. USD (Unknown Sensitive Detector, https://arxiv.org/abs/2306.02275) also uses SAM masks to produce pseudo-unknown boxes, but it uses them to **train** the detector, whereas this project only applies a rule at inference time. Other OWOD methods: ORE, OW-DETR, PROB.

## Files

| File | Purpose |
|---|---|
| `demo_detr_sam.py` | DETR + SAM demo, draws green/red boxes |
| `hybrid_proposals.py` | The three-condition rule (lines 59-61), IoU, NMS |
| `Abhay_project.pdf` | Theory report |
| `Abhay_OWOD_Presentation.pptx` | Presentation |

## How to run (Google Colab, GPU)

```
!pip install transformers timm pillow numpy
!python demo_detr_sam.py --images photos --out results
# tuned setting
!python demo_detr_sam.py --images photos --out results_tuned --known-thr 0.9 --tau-o 0.5 --tau-a 0.6
```
Put your own JPG/PNG photos in a `photos` folder. Large photos are resized to 1024 px; use `--max-side 640` if the GPU runs out of memory.

## Future work

- Use SAM regions as extra candidates (with quality, size and edge filters).
- Collect more photos with ground truth, tune thresholds, compute Unknown Recall.
- Train an objectness head or fine-tune with SAM pseudo-labels.
- Incremental learning on the India Driving Dataset (IDD) with exemplar replay and knowledge distillation; compare with PROB.

## Credits and disclosure

DETR and SAM are existing pretrained models (Hugging Face `transformers`). The code in this repository was written with the help of an AI assistant (Claude); the author ran it, collected the photos and analysed the results.
