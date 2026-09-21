"""The training loop, shared by both towers.

There is one loop because there is one method. A tower produces a ``Repr``, the
cached teacher target is wrapped in another ``Repr``, and a registered
objective compares them. Nothing about the loop knows whether it is looking at
pages or queries, or whether the representation is one vector or a set.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, DistributedSampler

from ..align import AlignLoss, Repr

__all__ = ["TrainConfig", "setup_distributed", "train"]


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 8
    peak_lr: float = 1e-3
    warmup_pct: float = 0.03
    weight_decay: float = 0.01
    num_workers: int = 8
    amp_dtype: str = "bfloat16"          # bfloat16 | float16 | none
    log_every: int = 50
    output_dir: str = "outputs/run"
    seed: int = 42
    max_grad_norm: float = 1.0
    extra: dict = field(default_factory=dict)


def setup_distributed() -> tuple[int, int, int]:
    """Initialise torch.distributed from torchrun's environment.

    Returns ``(rank, world_size, local_rank)``; ``(0, 1, 0)`` when run
    single-process.
    """
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local)
    return rank, world, local


def _target_repr(
    target: torch.Tensor,
    geometry: str,
    mask: torch.Tensor | None = None,
) -> Repr:
    """Wrap a cached teacher target, L2-normalising it as the losses expect.

    ``mask`` comes from a ragged multi-vector collate, which right-pads token
    sets of different lengths. Without it every padded row would count as a
    teacher atom and the transport marginals would be wrong.
    """
    if geometry == "single":
        return Repr(vec=F.normalize(target, p=2, dim=-1))
    tokens = F.normalize(target, p=2, dim=-1)
    if mask is None:
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
    return Repr(tokens=tokens, mask=mask)


def train(
    tower: torch.nn.Module,
    dataset,
    collate_fn: Callable,
    loss_fn: AlignLoss,
    cfg: TrainConfig,
    forward_fn: Callable[[torch.nn.Module, dict], Repr],
    param_groups: Optional[list] = None,
    val_dataset=None,
) -> Path:
    """Distil ``tower`` against cached teacher targets.

    ``forward_fn(tower, batch) -> Repr`` is the only tower-specific piece; the
    two entry points differ in that one line.
    """
    rank, world, local = setup_distributed()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed + rank)
    out_dir = Path(cfg.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    tower = tower.to(device)
    core = tower
    if world > 1:
        tower = DDP(tower, device_ids=[local], find_unused_parameters=False)

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    groups = param_groups if param_groups is not None else [{"params": core.parameters(), "lr": cfg.peak_lr}]
    optim = AdamW(groups, lr=cfg.peak_lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    sched = OneCycleLR(
        optim,
        max_lr=[g.get("lr", cfg.peak_lr) for g in groups],
        total_steps=total_steps,
        pct_start=cfg.warmup_pct,
        anneal_strategy="cos",
    )

    amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp is torch.float16))
    geometry = loss_fn.geometry

    if rank == 0:
        n_par = sum(p.numel() for p in core.parameters() if p.requires_grad)
        print(f"trainable {n_par/1e6:.1f}M | {len(dataset)} samples | "
              f"{steps_per_epoch} steps/epoch x {cfg.epochs} | world {world}", flush=True)

    best = float("inf")
    history: list[dict] = []
    step = 0
    for epoch in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        tower.train()
        running, seen, t0 = 0.0, 0, time.time()
        optim.zero_grad(set_to_none=True)

        for it, batch in enumerate(loader):
            target = batch.pop("target").to(device, non_blocking=True)
            tmask = batch.pop("target_mask", None)
            if tmask is not None:
                tmask = tmask.to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                student = forward_fn(tower, batch)
                out = loss_fn(student, _target_repr(target, geometry, tmask))
                loss = out["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()

            running += float(out["loss"].detach()) * target.size(0)
            seen += target.size(0)

            if (it + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(core.parameters(), cfg.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if step < total_steps - 1:
                    sched.step()
                step += 1
                if rank == 0 and step % cfg.log_every == 0:
                    rate = seen / max(time.time() - t0, 1e-6)
                    print(f"  epoch {epoch+1} step {step}/{total_steps} "
                          f"loss {running/max(seen,1):.4f} lr {sched.get_last_lr()[0]:.2e} "
                          f"{rate:.1f} samples/s", flush=True)

        epoch_loss = running / max(seen, 1)
        if world > 1:
            t = torch.tensor([epoch_loss], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            epoch_loss = float(t.item())

        val_loss = None
        if val_dataset is not None:
            val_loss = evaluate(core, val_dataset, collate_fn, loss_fn, cfg, forward_fn, device, geometry)

        if rank == 0:
            msg = f"epoch {epoch+1}/{cfg.epochs} train {epoch_loss:.4f}"
            if val_loss is not None:
                msg += f" val {val_loss:.4f}"
            print(msg, flush=True)
            history.append({"epoch": epoch + 1, "train_loss": epoch_loss, "val_loss": val_loss})
            score = val_loss if val_loss is not None else epoch_loss
            if score < best:
                best = score
                _save(core, out_dir / "best")
                print(f"  saved {out_dir/'best'} (loss {best:.4f})", flush=True)
            (out_dir / "metrics.json").write_text(
                json.dumps({"best_loss": best, "history": history}, indent=2)
            )

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return out_dir / "best"


@torch.no_grad()
def evaluate(core, dataset, collate_fn, loss_fn, cfg, forward_fn, device, geometry) -> float:
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    core.eval()
    total, seen = 0.0, 0
    for batch in loader:
        target = batch.pop("target").to(device)
        tmask = batch.pop("target_mask", None)
        if tmask is not None:
            tmask = tmask.to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        student = forward_fn(core, batch)
        out = loss_fn(student, _target_repr(target, geometry, tmask))
        total += float(out["loss"]) * target.size(0)
        seen += target.size(0)
    core.train()
    return total / max(seen, 1)


def _save(core, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(core, "save_pretrained"):
        core.save_pretrained(str(path))
    else:
        torch.save(core.state_dict(), path / "model.pt")
