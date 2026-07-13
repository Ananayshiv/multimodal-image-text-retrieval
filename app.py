"""Interactive demo for the best freshly trained retrieval model.

Run from this directory:
    python app.py

The first launch builds a validation embedding cache. Later launches reuse it as
long as the winning checkpoint has not changed.
"""
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from resumable_experiments import EXPERIMENTS, RetrievalSuite, SuiteConfig

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit(
        "Gradio is required for the demo UI. Install dependencies first with:\n"
        "    pip install -r requirements.txt"
    ) from exc


HERE = Path(__file__).resolve()
RUNS = HERE / "artifacts/fresh_runs"
RESULTS = RUNS / "fresh_comparison.csv"
CACHE = RUNS / "demo_validation_index.pt"


def load_winner():
    """Discover and restore the current winner—no model name is hardcoded."""
    table = pd.read_csv(RESULTS).sort_values("mean_score", ascending=False)
    winner = table.iloc[0]
    name = str(winner["experiment"])
    if name not in EXPERIMENTS:
        raise KeyError(f"Winning experiment {name!r} is absent from EXPERIMENTS")

    cfg = SuiteConfig(
        data_dir=str(HERE/ "data"),
        run_dir=str(RUNS),
        eval_batch_size=256,
        num_workers=0,
    )
    suite = RetrievalSuite(cfg)
    model, preprocess, tokenizer = suite._build_model(EXPERIMENTS[name])
    checkpoint_path = RUNS / name / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return suite, model, preprocess, tokenizer, winner, checkpoint_path


suite, model, preprocess, tokenizer, winner, checkpoint_path = load_winner()


def build_or_load_index():
    """Cache validation embeddings to keep interactive searches responsive."""
    stamp = checkpoint_path.stat().st_mtime_ns
    if CACHE.exists():
        value = torch.load(CACHE, map_location="cpu", weights_only=False)
        if value.get("checkpoint_stamp") == stamp:
            return value

    image_ids = sorted(suite.val.image_id.unique().astype(int).tolist())
    caption_ids = sorted(suite.val.caption_id.unique().astype(int).tolist())
    image_embeddings = suite._encode_images(model, preprocess, image_ids, suite.val_dir)
    text_embeddings = suite._encode_text(model, tokenizer, caption_ids)
    value = {
        "checkpoint_stamp": stamp,
        "image_ids": image_ids,
        "caption_ids": caption_ids,
        "image_embeddings": image_embeddings,
        "text_embeddings": text_embeddings,
    }
    torch.save(value, CACHE)
    return value


index = build_or_load_index()


def image_path(image_id: int) -> Path:
    return suite.val_dir / Path(suite.image_names[int(image_id)]).name


@torch.inference_mode()
def search_images(text: str, number: int = 5) -> List[Tuple[str, str]]:
    """Return validation images whose embeddings best match free-form text."""
    if not text or not text.strip():
        return []
    tokens = tokenizer([text.strip()]).to(suite.cfg.device)
    query = F.normalize(model.encode_text(tokens), dim=-1).cpu()[0]
    scores = query @ index["image_embeddings"].T
    positions = scores.topk(min(int(number), len(scores))).indices.tolist()
    return [
        (str(image_path(index["image_ids"][p])), f"image_id={index['image_ids'][p]} · similarity={scores[p]:.3f}")
        for p in positions
    ]


@torch.inference_mode()
def search_captions(image: Image.Image, number: int = 5) -> str:
    """Return ranked validation captions for an uploaded image."""
    if image is None:
        return "Upload an image to retrieve matching captions."
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(suite.cfg.device)
    query = F.normalize(model.encode_image(tensor), dim=-1).cpu()[0]
    scores = query @ index["text_embeddings"].T
    positions = scores.topk(min(int(number), len(scores))).indices.tolist()
    lines = []
    for rank, p in enumerate(positions, 1):
        caption_id = index["caption_ids"][p]
        lines.append(f"**{rank}.** {suite.captions[caption_id]}  \n`caption_id={caption_id} · similarity={scores[p]:.3f}`")
    return "\n\n".join(lines)


model_summary = (
    f"**Active model:** `{winner['experiment']}` · "
    f"**Mean validation score:** {winner['mean_score']:.4f} · "
    f"**Trainable parameters:** {int(winner['trainable_parameters']):,}"
)

with gr.Blocks(title="Multimodal Retrieval Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Multimodal Retrieval Explorer")
    gr.Markdown("Search images with natural language, or upload an image to retrieve matching captions.")
    gr.Markdown(model_summary)

    with gr.Tab("Text → Images"):
        with gr.Row():
            text_query = gr.Textbox(label="Describe an image", placeholder="A dog running through snow")
            image_count = gr.Slider(1, 10, value=5, step=1, label="Results")
        image_button = gr.Button("Search", variant="primary")
        gallery = gr.Gallery(label="Ranked images", columns=5, height="auto")
        image_button.click(search_images, [text_query, image_count], gallery)
        text_query.submit(search_images, [text_query, image_count], gallery)

    with gr.Tab("Image → Captions"):
        with gr.Row():
            uploaded_image = gr.Image(type="pil", label="Upload an image")
            caption_count = gr.Slider(1, 10, value=5, step=1, label="Results")
        caption_button = gr.Button("Retrieve captions", variant="primary")
        captions = gr.Markdown("Upload an image to retrieve matching captions.")
        caption_button.click(search_captions, [uploaded_image, caption_count], captions)

    gr.Markdown("The demo searches the held-out validation collection. Similarity is cosine similarity in the learned CLIP embedding space.")


if __name__ == "__main__":
    demo.launch(allowed_paths=[str(suite.val_dir)])
