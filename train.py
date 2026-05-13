from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm
import random

DATA_DIR = "data"
OUT_DIR = "checkpoints"
IMAGE_SIZE = 256
TIMESTEPS = 200
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 20
NUM_WORKERS = 2
SAVE_EVERY_EPOCHS = 1
SUBSET_FRACTION = 0.1
GRAD_ACCUM_STEPS = 4
# Resume training from checkpoints/ckpt_latest.pt if present
RESUME_TRAINING = True

class FlatImageDataset(Dataset):
    def __init__(self, data_dir: str | Path, image_size: int):
        data_dir = Path(data_dir)
        self.paths = sorted([p for p in data_dir.iterdir()])
        if not self.paths:
            raise FileNotFoundError(f"No images found in {data_dir}")

        if float(SUBSET_FRACTION) < 1.0:
            frac = max(0.0, min(1.0, float(SUBSET_FRACTION)))
            k = max(1, int(round(len(self.paths) * frac)))
            rng = random.Random()
            paths = list(self.paths)
            rng.shuffle(paths)
            self.paths = sorted(paths[:k])

        self.tfm = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tfm(img)

def mkdirp(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / (half - 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=device)], dim=1)
    return emb

# =============================
# U-Net (64 -> 128 -> 256)
# =============================


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2)) # time embedding injected
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        #silu after each norm layer
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.res(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, time_emb_dim: int = 256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4  # 64, 128, 256

        self.init = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.d1a = ResidualBlock(c1, c1, time_emb_dim)
        self.d1b = ResidualBlock(c1, c1, time_emb_dim)
        self.ds1 = Downsample(c1)

        self.d2a = ResidualBlock(c1, c2, time_emb_dim)
        self.d2b = ResidualBlock(c2, c2, time_emb_dim)
        self.ds2 = Downsample(c2)

        self.d3a = ResidualBlock(c2, c3, time_emb_dim)
        self.d3b = ResidualBlock(c3, c3, time_emb_dim)

        self.m1 = ResidualBlock(c3, c3, time_emb_dim)
        self.m2 = ResidualBlock(c3, c3, time_emb_dim)

        self.us2 = Upsample(c3)
        self.u2a = ResidualBlock(c3 + c2, c2, time_emb_dim)
        self.u2b = ResidualBlock(c2, c2, time_emb_dim)

        self.us1 = Upsample(c2)
        self.u1a = ResidualBlock(c2 + c1, c1, time_emb_dim)
        self.u1b = ResidualBlock(c1, c1, time_emb_dim)

        self.out_norm = nn.GroupNorm(8, c1)
        self.out = nn.Conv2d(c1, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        x = self.init(x)

        x1 = self.d1a(x, t_emb)
        x1 = self.d1b(x1, t_emb)
        x = self.ds1(x1)

        x2 = self.d2a(x, t_emb)
        x2 = self.d2b(x2, t_emb)
        x = self.ds2(x2)

        x3 = self.d3a(x, t_emb)
        x3 = self.d3b(x3, t_emb)

        x = self.m1(x3, t_emb)
        x = self.m2(x, t_emb)

        x = self.us2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.u2a(x, t_emb)
        x = self.u2b(x, t_emb)

        x = self.us1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.u1a(x, t_emb)
        x = self.u1b(x, t_emb)

        return self.out(F.silu(self.out_norm(x)))


# =============================
# DDPM (train objective only)
# =============================


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

@dataclass
class Schedule:
    betas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor


def make_schedule(T: int, device: torch.device) -> Schedule:
    betas = linear_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return Schedule(
        betas=betas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
    )


class DDPMTrainer:
    def __init__(self, model: nn.Module, timesteps: int):
        self.model = model
        self.timesteps = timesteps

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, sch: Schedule, noise: torch.Tensor) -> torch.Tensor:
        return extract(sch.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(
            sch.sqrt_one_minus_alphas_cumprod, t, x0.shape
        ) * noise

    def loss(self, x0: torch.Tensor, t: torch.Tensor, sch: Schedule) -> torch.Tensor:
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, sch, noise)
        pred = self.model(xt, t)
        loss = (pred - noise) ** 2
        return (loss).mean()


def _load_latest_checkpoint(out_dir: Path, device: torch.device) -> dict | None:
    ckpt_path = out_dir / "ckpt_latest.pt"
    if not ckpt_path.exists():
        return None
    return torch.load(ckpt_path, map_location=device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = mkdirp(OUT_DIR)
    log_path = out_dir / "losses.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "avg_loss"])

    ds = FlatImageDataset(DATA_DIR, image_size=int(IMAGE_SIZE))
    dl = DataLoader(
        ds,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        num_workers=int(NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet().to(device)
    trainer = DDPMTrainer(model, timesteps=int(TIMESTEPS))
    sch = make_schedule(int(TIMESTEPS), device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(LR))

    global_step = 0
    start_epoch = 1
    if bool(RESUME_TRAINING):
        ckpt = _load_latest_checkpoint(out_dir, device=device)
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            global_step = int(ckpt.get("global_step", 0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1

            # keep schedule aligned with the checkpoint if it differs
            ckpt_timesteps = int(ckpt.get("timesteps", int(TIMESTEPS)))
            sch = make_schedule(ckpt_timesteps, device=device)
            trainer.timesteps = ckpt_timesteps

            print(f"Resuming from {out_dir / 'ckpt_latest.pt'} at epoch {start_epoch}/{EPOCHS}.")

    for epoch in range(start_epoch, int(EPOCHS) + 1):
        model.train()
        pbar = tqdm(dl, desc=f"epoch {epoch}/{EPOCHS}")
        loss_sum = 0.0
        loss_n = 0
        accum_steps = int(GRAD_ACCUM_STEPS)
        opt.zero_grad(set_to_none=True)
        for x0 in pbar:
            x0 = x0.to(device)
            t = torch.randint(0, int(trainer.timesteps), (x0.shape[0],), device=device, dtype=torch.long)
            loss = trainer.loss(x0, t, sch)

            (loss / accum_steps).backward()

            do_step = ((loss_n + 1) % accum_steps) == 0
            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

                global_step += 1

            loss_sum += float(loss.item())
            loss_n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # flush remainder if epoch ends mid-accumulation
        if (loss_n % accum_steps) != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            global_step += 1

        avg_loss = loss_sum / max(1, loss_n)
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([epoch, avg_loss])

        if epoch % int(SAVE_EVERY_EPOCHS) == 0:
            ckpt = {
                "epoch": epoch,
                "global_step": global_step,
                "timesteps": int(TIMESTEPS),
                "image_size": int(IMAGE_SIZE),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": {
                    "data_dir": DATA_DIR,
                    "out_dir": OUT_DIR,
                    "image_size": IMAGE_SIZE,
                    "timesteps": TIMESTEPS,
                    "batch_size": BATCH_SIZE,
                    "lr": LR,
                    "epochs": EPOCHS,
                    "num_workers": NUM_WORKERS,
                    "save_every_epochs": SAVE_EVERY_EPOCHS,
                    "subset_fraction": SUBSET_FRACTION,
                },
            }
            torch.save(ckpt, out_dir / "ckpt_latest.pt")
            torch.save(ckpt, out_dir / f"ckpt_epoch_{epoch:03d}.pt")

    print(f"Done. Checkpoints saved in: {out_dir}")


if __name__ == "__main__":
    main()

