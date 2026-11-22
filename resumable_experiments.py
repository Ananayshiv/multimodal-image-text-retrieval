"""Resumable, fair multimodal-retrieval experiment suite.

All strategies use the same OpenCLIP backbone, data split, loss, optimizer family,
and evaluation code. Only the adaptation strategy changes. State is saved atomically
throughout each epoch; completed experiments are skipped on subsequent invocations.

Splits (v2):
  train  = data/train.csv minus a held-out DEV set of `selection_images` images
  dev    = images carved from train (fixed `split_seed`); used ONLY to pick the epoch
  test   = data/val.csv (1,000 images / 5,000 captions); never used for any decision
data/test.csv is the unlabeled competition query set and is not used for metrics.
Set selection_images=0 to reproduce the v1 protocol (epoch picked on val.csv).
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import open_clip

from retrieval_metrics import metrics_from_ranks, per_query_frame, ranks_from_similarity


@dataclass
class SuiteConfig:
    data_dir: str = "data"
    run_dir: str = "artifacts/fresh_runs"
    model_name: str = "ViT-B-16"
    pretrained: str = "openai"
    epochs: int = 5
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    seed: int = 42
    selection_images: int = 500   # dev images carved from train for epoch selection (0 = v1 protocol)
    split_seed: int = 0           # fixed across training seeds so only training randomness varies
    patience: int = 0             # early-stop after N epochs without DEV improvement (0 = off)
    checkpoint_every_batches: int = 100
    fast_dev_run: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


EXPERIMENTS = {
    # Recreates the major model/adaptation families found in the source notebooks.
    "clip_zero_shot": {"strategy": "zero_shot", "lr": 0.0, "epochs": 0},
    "clip_projection_tuning": {"strategy": "projection", "lr": 1e-4},
    "openclip_full_finetune": {"strategy": "full", "lr": 1e-6},
    "openclip_gradual_unfreeze": {"strategy": "gradual", "lr": 5e-6},
    "openclip_lora": {"strategy": "lora", "lr": 1e-4, "lora_rank": 8, "lora_alpha": 16},
}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _remove_empty_directory_artifact(path: Path) -> None:
    if path.is_dir():
        try:
            path.rmdir()
        except OSError as exc:
            raise IsADirectoryError(
                f"{path} is a non-empty directory, but this path must be a file. "
                "Move or remove it before resuming."
            ) from exc


def atomic_torch_save(value, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    _remove_empty_directory_artifact(path)
    _remove_empty_directory_artifact(tmp)
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json_save(value, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    _remove_empty_directory_artifact(path)
    _remove_empty_directory_artifact(tmp)
    tmp.write_text(json.dumps(value, indent=2, default=str))
    os.replace(tmp, path)


class LoRALinear(nn.Module):
    """Drop-in linear layer that keeps the original weight frozen."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__(); self.base = base; self.scale = alpha / rank
        for p in self.base.parameters(): p.requires_grad = False
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=np.sqrt(5))

    @property
    def weight(self):
        """Expose the base weight for modules that inspect Linear metadata."""
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def inject_lora(module: nn.Module, rank: int, alpha: int) -> int:
    """Adapt OpenCLIP transformer MLP projections.

    ``nn.MultiheadAttention`` consumes ``out_proj.weight`` directly through its
    functional implementation, so replacing ``out_proj`` with a wrapper breaks
    the module contract. OpenCLIP's ``c_fc``/``c_proj`` layers are invoked
    normally and are therefore safe LoRA targets.
    """
    changed = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name.lower() in {"c_fc", "c_proj", "fc1", "fc2"}:
            setattr(module, name, LoRALinear(child, rank, alpha)); changed += 1
        else:
            changed += inject_lora(child, rank, alpha)
    return changed


class PairDataset(Dataset):
    def __init__(self, pairs, image_dir, image_names, captions, preprocess):
        self.pairs = pairs.reset_index(drop=True); self.image_dir = Path(image_dir)
        self.image_names = image_names; self.captions = captions; self.preprocess = preprocess
    def __len__(self): return len(self.pairs)
    def __getitem__(self, index):
        row = self.pairs.iloc[index]; iid, cid = int(row.image_id), int(row.caption_id)
        path = self.image_dir / Path(self.image_names[iid]).name
        return self.preprocess(Image.open(path).convert("RGB")), self.captions[cid]


class RetrievalSuite:
    def __init__(self, cfg: SuiteConfig):
        self.cfg = cfg; self.data = Path(cfg.data_dir).resolve(); self.runs = Path(cfg.run_dir).resolve()
        self.runs.mkdir(parents=True, exist_ok=True); seed_all(cfg.seed)
        self.train = pd.read_csv(self.data / "train.csv")
        self.val = pd.read_csv(self.data / "val.csv")
        self.test = pd.read_csv(self.data / "test.csv")
        self.captions_df = pd.read_csv(self.data / "caption_pool.csv")
        self.images_df = pd.read_csv(self.data / "image_pool.csv")
        self.captions = self.captions_df.set_index("caption_id").caption.astype(str).to_dict()
        self.image_names = self.images_df.set_index("image_id").image_name.astype(str).to_dict()
        self.train_dir = self.data / "train_images/train_images"
        self.val_dir = self.data / "val_images/val_images"
        self._validate_data()
        if cfg.selection_images > 0:
            ids = np.array(sorted(self.train.image_id.unique()))
            if cfg.selection_images >= len(ids):
                raise ValueError("selection_images must be smaller than the number of training images")
            dev_ids = set(np.random.default_rng(cfg.split_seed).choice(ids, cfg.selection_images, replace=False).tolist())
            self.dev = self.train[self.train.image_id.isin(dev_ids)].reset_index(drop=True)
            self.train = self.train[~self.train.image_id.isin(dev_ids)].reset_index(drop=True)
            self.dev_dir = self.train_dir
            assert not set(self.dev.image_id) & set(self.train.image_id)
        else:  # v1 protocol: selection on the reported split (optimistically biased)
            self.dev, self.dev_dir = self.val, self.val_dir
        self.selection_split = "dev (carved from train)" if cfg.selection_images > 0 else "val.csv (v1, biased)"

    def _validate_data(self):
        assert {"image_id", "caption_id"} <= set(self.train)
        assert {"image_id", "caption_id"} <= set(self.val)
        missing = [i for i in self.train.image_id.unique() if not (self.train_dir / Path(self.image_names[int(i)]).name).exists()]
        if missing: raise FileNotFoundError(f"Missing {len(missing)} training images; first ID: {missing[0]}")

    def _build_model(self, spec):
        model, _, preprocess = open_clip.create_model_and_transforms(self.cfg.model_name, pretrained=self.cfg.pretrained)
        tokenizer = open_clip.get_tokenizer(self.cfg.model_name)
        strategy = spec["strategy"]
        if strategy in {"zero_shot", "projection", "lora", "gradual"}:
            for p in model.parameters(): p.requires_grad = False
        if strategy in {"projection", "gradual"}:
            for n, p in model.named_parameters():
                if any(x in n for x in ("visual.proj", "text_projection", "ln_post", "ln_final", "logit_scale")): p.requires_grad = True
        elif strategy == "lora":
            count = inject_lora(model, spec.get("lora_rank", 8), spec.get("lora_alpha", 16))
            model.logit_scale.requires_grad = True
            if count == 0: raise RuntimeError("No transformer MLP projections received LoRA adapters")
        elif strategy == "full":
            for p in model.parameters(): p.requires_grad = True
        return model.to(self.cfg.device), preprocess, tokenizer

    def _loader(self, dataset, tokenizer, epoch):
        def collate(items):
            images, texts = zip(*items); return torch.stack(images), tokenizer(list(texts))
        generator = torch.Generator().manual_seed(self.cfg.seed + epoch)
        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True, generator=generator,
                          num_workers=self.cfg.num_workers, pin_memory=True, collate_fn=collate)

    @torch.no_grad()
    def _encode_images(self, model, preprocess, ids, directory):
        model.eval(); output=[]
        for start in range(0, len(ids), self.cfg.eval_batch_size):
            batch = [preprocess(Image.open(Path(directory) / Path(self.image_names[i]).name).convert("RGB")) for i in ids[start:start+self.cfg.eval_batch_size]]
            z = model.encode_image(torch.stack(batch).to(self.cfg.device)); output.append(F.normalize(z, dim=-1).cpu())
        return torch.cat(output)

    @torch.no_grad()
    def _encode_text(self, model, tokenizer, ids):
        model.eval(); output=[]
        for start in range(0, len(ids), self.cfg.eval_batch_size):
            tok = tokenizer([self.captions[i] for i in ids[start:start+self.cfg.eval_batch_size]]).to(self.cfg.device)
            output.append(F.normalize(model.encode_text(tok), dim=-1).cpu())
        return torch.cat(output)

    @torch.no_grad()
    def evaluate(self, model, preprocess, tokenizer, split: str = "dev", per_query: bool = False):
        """Exact-rank retrieval metrics on `dev` (selection) or `test` (= data/val.csv)."""
        frame, directory = (self.dev, self.dev_dir) if split == "dev" else (self.val, self.val_dir)
        image_ids = sorted(frame.image_id.unique().astype(int)); caption_ids = sorted(frame.caption_id.unique().astype(int))
        ie = self._encode_images(model, preprocess, image_ids, directory)
        te = self._encode_text(model, tokenizer, caption_ids); sim = (ie @ te.T).float().numpy()
        igt = frame.groupby("image_id").caption_id.apply(list).to_dict(); tgt = frame.set_index("caption_id").image_id.to_dict()
        i2t, t2i = ranks_from_similarity(sim, image_ids, caption_ids, igt, tgt)
        metrics = metrics_from_ranks(i2t, t2i)
        if per_query:
            return metrics, per_query_frame(image_ids, caption_ids, i2t, t2i)
        return metrics

    def _state(self, model, optimizer, scaler, epoch, next_batch, history, best):
        return {"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "scaler": scaler.state_dict() if scaler else None, "epoch": epoch, "next_batch": next_batch,
                "history": history, "best": best, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    def run_one(self, name: str, spec: Dict):
        folder = self.runs / name; done = folder / "completed.json"; last = folder / "last.pt"
        folder.mkdir(parents=True, exist_ok=True)
        _remove_empty_directory_artifact(done)
        _remove_empty_directory_artifact(last)
        _remove_empty_directory_artifact(folder / "failure.json")
        if done.exists():
            print(f"[skip] {name} is complete"); return json.loads(done.read_text())
        model, preprocess, tokenizer = self._build_model(spec)
        epochs = 0 if spec["strategy"] == "zero_shot" else spec.get("epochs", self.cfg.epochs)
        unfreeze_epoch = max(2, epochs // 2)
        history=[]; best=-1.; start_epoch=1; start_batch=0; state=None
        if last.exists():
            state=torch.load(last, map_location=self.cfg.device, weights_only=False); model.load_state_dict(state["model"])
            history, best = state["history"], state["best"]; start_epoch, start_batch = state["epoch"], state["next_batch"]
            print(f"[resume] {name}: epoch {start_epoch}, batch {start_batch}")
        full_gradual = spec["strategy"] == "gradual" and (start_epoch > unfreeze_epoch or (start_epoch == unfreeze_epoch and start_batch > 0))
        if full_gradual:
            for p in model.parameters(): p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        if epochs > 0 and not trainable:
            raise RuntimeError(f"Experiment {name!r} selected no trainable parameters")
        optimizer = torch.optim.AdamW(trainable, lr=spec["lr"], weight_decay=1e-4) if trainable else None
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.device == "cuda" and optimizer is not None)
        if state is not None and optimizer and state["optimizer"] and not (spec["strategy"] == "gradual" and start_epoch == unfreeze_epoch and start_batch == 0):
            optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        pairs = self.train.head(512) if self.cfg.fast_dev_run else self.train
        dataset = PairDataset(pairs, self.train_dir, self.image_names, self.captions, preprocess)
        if epochs == 0:
            metrics=self.evaluate(model,preprocess,tokenizer,"dev"); history=[{"epoch":0,"train_loss":np.nan,**metrics}]
        else:
            for epoch in range(start_epoch, epochs+1):
                if spec["strategy"] == "gradual" and epoch == unfreeze_epoch and not full_gradual:
                    for p in model.parameters(): p.requires_grad=True
                    optimizer=torch.optim.AdamW(model.parameters(),lr=spec["lr"],weight_decay=1e-4)
                loader=self._loader(dataset,tokenizer,epoch); model.train(); losses=[]
                resume_batch=start_batch if epoch==start_epoch else 0
                for bi,(images,tokens) in enumerate(tqdm(loader,desc=f"{name} e{epoch}")):
                    if bi < resume_batch: continue
                    images,tokens=images.to(self.cfg.device),tokens.to(self.cfg.device); optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",enabled=self.cfg.device=="cuda"):
                        ie=F.normalize(model.encode_image(images),dim=-1); te=F.normalize(model.encode_text(tokens),dim=-1)
                        logits=model.logit_scale.exp().clamp(max=100)*(ie@te.T); labels=torch.arange(len(images),device=self.cfg.device)
                        loss=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(loss.item())
                    if (bi+1)%self.cfg.checkpoint_every_batches==0:
                        atomic_torch_save(self._state(model,optimizer,scaler,epoch,bi+1,history,best),last)
                metrics=self.evaluate(model,preprocess,tokenizer,"dev"); record={"epoch":epoch,"train_loss":float(np.mean(losses)),**metrics}; history.append(record)
                pd.DataFrame(history).to_csv(folder/"history.csv",index=False)
                if metrics["mean_score"]>best:
                    best=metrics["mean_score"]; atomic_torch_save({"model":model.state_dict(),"metrics":metrics},folder/"best.pt")
                atomic_torch_save(self._state(model,optimizer,scaler,epoch+1,0,history,best),last); start_batch=0
                # Early stopping on the DEV split only (never the reported test split). For gradual
                # unfreezing, patience starts counting once the backbone has been unfrozen.
                best_epoch=max(history,key=lambda x:x["mean_score"])["epoch"]
                past_unfreeze = spec["strategy"] != "gradual" or epoch >= unfreeze_epoch
                if self.cfg.patience and past_unfreeze and epoch - max(best_epoch, unfreeze_epoch if spec["strategy"]=="gradual" else 0) >= self.cfg.patience:
                    print(f"[early stop] {name}: no dev improvement for {self.cfg.patience} epochs (best epoch {best_epoch})")
                    break
        best_record=max(history,key=lambda x:x["mean_score"])
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
        # Final, one-time evaluation of the dev-selected checkpoint on the held-out test split.
        if epochs > 0 and (folder/"best.pt").exists():
            model.load_state_dict(torch.load(folder/"best.pt", map_location=self.cfg.device, weights_only=False)["model"])
        test_metrics, test_queries = self.evaluate(model, preprocess, tokenizer, "test", per_query=True)
        test_queries.to_csv(folder/"test_per_query.csv", index=False)
        result={"experiment":name,"strategy":spec["strategy"],"status":"completed","best_epoch":best_record["epoch"],
                "seed":self.cfg.seed,"epochs_run":len(history),"patience":self.cfg.patience,"selection_split":self.selection_split,"model":f"{self.cfg.model_name}/{self.cfg.pretrained}",
                "trainable_parameters":trainable_parameters,
                **{f"dev_{k}":v for k,v in best_record.items() if k not in ("epoch",)},
                **{f"test_{k}":v for k,v in test_metrics.items()}}
        atomic_json_save(result,done); return result

    def run_all(self, experiments=None):
        experiments = experiments or EXPERIMENTS; results=[]
        for name,spec in experiments.items():
            try: results.append(self.run_one(name,spec))
            except Exception as exc:
                atomic_json_save({"experiment":name,"status":"failed","error":repr(exc)},self.runs/name/"failure.json")
                raise
            finally:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        table=pd.DataFrame(results).sort_values("test_mean_score",ascending=False); table.to_csv(self.runs/"fresh_comparison.csv",index=False)
        return table
