# Multimodal Image–Text Retrieval

This project learns a shared semantic space for images and captions. A user can
describe a scene to retrieve matching images, or provide an image to retrieve the
captions that best describe it.

![Fresh experiment comparison](assets/fresh_comparison.png)

## Fresh benchmark result

All five strategies were rerun with the same split, backbone, evaluation code, and
checkpoint-selection rule. LoRA OpenCLIP produced the strongest result:

| Model | I→T score | T→I score | Mean score | Trainable parameters |
|---|---:|---:|---:|---:|
| LoRA OpenCLIP | **0.9600** | **0.8929** | **0.9264** | 1,228,801 |
| Gradual unfreezing | 0.9547 | 0.8917 | 0.9232 | 149,620,737 |
| Full fine-tuning | 0.9537 | 0.8885 | 0.9211 | 149,620,737 |
| Projection tuning | 0.9150 | 0.8329 | 0.8739 | 657,921 |
| Zero-shot CLIP | 0.8990 | 0.7732 | 0.8361 | 0 |

The important result is not only that LoRA won: it did so while training less than
1% of the parameters used by full fine-tuning.

## Explore the project

- `visual_story.ipynb` explains the problem, experiments, results, qualitative
  examples, and limitations.
- `train_all_models_resumable.ipynb` reproduces the controlled benchmark and safely
  resumes after interruption.
- `app.py` provides an interactive text→image and image→caption demo.

## Run the demo

```bash
conda activate deep_trial
pip install -r requirements.txt
python app.py
```

The first launch creates a validation embedding cache. Subsequent launches reuse it
unless the winning checkpoint changes.

## Evaluation

Recall@K asks whether a correct match appears among the first K results. The project
reports Recall@1, Recall@5, and Recall@10 in both directions. Each direction score is
the mean of those three recalls, and the headline score is the mean of the two
direction scores.

## Reproducibility

Training state is written atomically every 100 batches and after every epoch.
Restarting the training notebook restores the current model, optimizer, scaler,
epoch, and batch position; completed experiments are skipped.
