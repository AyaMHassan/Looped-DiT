"""Training data streams.

chunks      pretraining: `chunk_*.pt` files holding uint8 images [N, 3, H, W] and
            T5 token ids / attention masks [N, prompt_length]
            (written by tools/make_chunks.py); several directories are read as one pool.
webdataset  fine-tuning: a weighted mixture of WebDataset sources whose samples
            are an image (.jpg/.jpeg/.png) plus a caption (.txt), tokenized on the fly.

Both streams are infinite and shard their files over (rank, dataloader worker).
"""

from __future__ import annotations

import itertools
import random
from pathlib import Path

import torch
import webdataset as wds
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from transformers import AutoTokenizer

from .config import TrainConfig
from .utils import rank, world_size

ImageFile.LOAD_TRUNCATED_IMAGES = True


def worker_shard(files: list, min_one: bool = False) -> list:
    """The part of `files` read by this (rank, dataloader worker)."""
    info = get_worker_info()
    num_workers, worker_id = (info.num_workers, info.id) if info else (1, 0)
    index, count = rank() * num_workers + worker_id, world_size() * num_workers
    shard = files[index::count]
    if not shard and min_one and files:
        shard = [files[index % len(files)]]
    return shard


def worker_seed(cfg: TrainConfig, offset: int = 0) -> int:
    info = get_worker_info()
    num_workers, worker_id = (info.num_workers, info.id) if info else (1, 0)
    return cfg.seed + offset + rank() * num_workers + worker_id


class ChunkStream(IterableDataset):
    def __init__(self, cfg: TrainConfig, seed_offset: int = 0):
        self.cfg, self.seed_offset = cfg, seed_offset
        self.files = sorted(p for d in cfg.chunk_dirs for p in Path(d).glob("chunk_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"no chunk_*.pt files in {cfg.chunk_dirs}")

    def samples(self, files: list[Path], rng: random.Random):
        while True:
            for path in rng.sample(files, len(files)):
                chunk = torch.load(path, map_location="cpu", weights_only=False)
                for i in rng.sample(range(len(chunk["input_ids"])), len(chunk["input_ids"])):
                    yield {
                        "pixel_values": chunk["pixel_values"][i].clone(),  # don't pin the whole chunk
                        "input_ids": chunk["input_ids"][i].long(),
                        "attention_mask": chunk["attention_mask"][i].long(),
                    }

    def __iter__(self):
        files = worker_shard(self.files)
        if not files:
            raise RuntimeError(f"{len(self.files)} chunks cannot be split over every rank and dataloader worker")
        rng = random.Random(worker_seed(self.cfg, self.seed_offset))
        buffer: list[dict] = []
        # Sample-level shuffle buffer, so that micro-batches mix chunks.
        for sample in self.samples(files, rng):
            if len(buffer) < self.cfg.shuffle_buffer:
                buffer.append(sample)
                continue
            j = rng.randrange(len(buffer))
            yield buffer[j]
            buffer[j] = sample


class WebDatasetMix(IterableDataset):
    def __init__(self, cfg: TrainConfig, seed_offset: int = 0, tokenizer=None):
        self.cfg, self.seed_offset = cfg, seed_offset
        self.names = list(cfg.webdataset_sources)
        self.weights = [float(cfg.webdataset_sources[n]["weight"]) for n in self.names]
        self.shards = []
        for name in self.names:
            root = Path(cfg.webdataset_sources[name]["path"])
            shards = sorted(str(p) for p in root.rglob("*.tar") if not any(part.startswith(".") for part in p.relative_to(root).parts))
            if not shards:
                raise FileNotFoundError(f"no .tar shards for source {name!r} under {root}")
            self.shards.append(shards)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.text_encoder, model_max_length=cfg.prompt_length)
        self.transform = transforms.Compose(
            [
                transforms.Resize(cfg.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(cfg.image_size),
                transforms.PILToTensor(),
            ]
        )

    def to_sample(self, sample: dict) -> dict:
        image = sample.get("jpg") or sample.get("jpeg") or sample.get("png")
        if not isinstance(image, Image.Image):
            raise ValueError(f"sample {sample.get('__key__')} has no image")
        caption = sample.get("txt", "")
        caption = caption.decode("utf-8", errors="replace") if isinstance(caption, bytes) else str(caption)
        tokens = self.tokenizer(
            caption, max_length=self.cfg.prompt_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "pixel_values": self.transform(image.convert("RGB")),
            "input_ids": tokens["input_ids"][0].long(),
            "attention_mask": tokens["attention_mask"][0].long(),
        }

    def source(self, index: int, epoch: int):
        """One pass over this worker's shards of source `index`."""
        urls = worker_shard(self.shards[index], min_one=True)
        rng = random.Random(worker_seed(self.cfg, self.seed_offset + 1000 * epoch + index))
        return (
            wds.WebDataset(
                urls,
                handler=wds.warn_and_continue,
                empty_check=False,
                shardshuffle=max(1, len(urls)),
                nodesplitter=lambda src: src,
                workersplitter=lambda src: src,
            )
            .shuffle(self.cfg.shuffle_buffer, rng=rng)
            .decode("pil", handler=wds.warn_and_continue)
            .map(self.to_sample, handler=wds.warn_and_continue)
        )

    def endless(self, index: int):
        """Samples of source `index`, starting a new pass whenever its shards run out."""
        for epoch in itertools.count():
            count = 0
            for count, sample in enumerate(self.source(index, epoch), 1):
                yield sample
            if count == 0:
                raise RuntimeError(f"source {self.names[index]!r} has no readable samples")

    def __iter__(self):
        rng = random.Random(worker_seed(self.cfg, self.seed_offset + 1009))
        streams = [self.endless(i) for i in range(len(self.names))]
        while True:
            (i,) = rng.choices(range(len(streams)), weights=self.weights)
            yield next(streams[i])


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in batch]) for key in ("pixel_values", "input_ids", "attention_mask")}


def make_loader(cfg: TrainConfig, seed_offset: int = 0, batch_size: int | None = None) -> DataLoader:
    stream = ChunkStream if cfg.dataset == "chunks" else WebDatasetMix
    return DataLoader(
        stream(cfg, seed_offset),
        batch_size=batch_size or cfg.micro_batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )


def make_packed_loader(cfg: TrainConfig, level_sizes: list[int], seed_offset: int = 0) -> DataLoader:
    """Loader for --mlmc training: every batch packs the samples of all MLMC levels.

    A batch holds sum(level_sizes) samples; level k owns the next level_sizes[k-1] of
    them, in level order (level 1 first), and mlmc_training_loss splits it the same way.
    The streams shuffle i.i.d., so the slices are independent samples, as the telescope
    needs. cfg.micro_batch_size is not used.
    """
    if not level_sizes or min(level_sizes) < 1:
        raise ValueError(f"every level needs at least one sample, got {level_sizes}")
    return make_loader(cfg, seed_offset, batch_size=sum(level_sizes))
