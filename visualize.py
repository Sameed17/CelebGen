from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image

import train  # reuse UNet + training-time config constants


# =============================
# Config (edit here)
# =============================


CKPT_PATH = Path(train.OUT_DIR) / "ckpt_latest.pt"
TARGET_IMAGE = Path(train.DATA_DIR) / "img_0.png"
OUT_DIR = Path("viz_outputs")

N_FORWARD_STEPS = 5
N_REVERSE_STEPS = 5
N_GENERATED_IMAGES = 5


# =============================
# DDPM sampling utilities (reverse process)
# =============================


def linear_beta_schedule(T_: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T_, dtype=torch.float32)


@dataclass
class FullSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    sqrt_recip_alphas: torch.Tensor
    posterior_variance: torch.Tensor


def make_full_schedule(T_: int, device: torch.device) -> FullSchedule:
    betas = linear_beta_schedule(T_).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return FullSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
    )


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


@torch.no_grad()
def q_sample(x0: torch.Tensor, t: torch.Tensor, sch: FullSchedule, noise: torch.Tensor | None = None) -> torch.Tensor:
    noise = torch.randn_like(x0) if noise is None else noise
    return extract(sch.sqrt_alphas_cumprod, t, x0.shape) * x0 + extract(
        sch.sqrt_one_minus_alphas_cumprod, t, x0.shape
    ) * noise


@torch.no_grad()
def p_sample(model: torch.nn.Module, xt: torch.Tensor, t: torch.Tensor, sch: FullSchedule) -> torch.Tensor:
    betas_t = extract(sch.betas, t, xt.shape)
    sqrt_one_minus_acp_t = extract(sch.sqrt_one_minus_alphas_cumprod, t, xt.shape)
    sqrt_recip_alphas_t = extract(sch.sqrt_recip_alphas, t, xt.shape)

    eps = model(xt, t)
    mean = sqrt_recip_alphas_t * (xt - betas_t * eps / sqrt_one_minus_acp_t)

    var = extract(sch.posterior_variance, t, xt.shape)
    noise = torch.randn_like(xt)
    nonzero = (t != 0).float().reshape(xt.shape[0], *((1,) * (len(xt.shape) - 1)))
    return mean + nonzero * torch.sqrt(var) * noise


@torch.no_grad()
def sample_from_noise(model: torch.nn.Module, image_size: int, timesteps: int, n: int, device: torch.device) -> torch.Tensor:
    sch = make_full_schedule(timesteps, device)
    x = torch.randn((n, 3, image_size, image_size), device=device)
    for i in reversed(range(timesteps)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, sch)
    return x


# =============================
# Metrics (PSNR / SSIM)
# =============================


def to_01(x: torch.Tensor) -> torch.Tensor:
    # x in [-1, 1] -> [0, 1]
    return (x.clamp(-1, 1) + 1.0) * 0.5


def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10) -> float:
    x01 = to_01(x)
    y01 = to_01(y)
    mse = torch.mean((x01 - y01) ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(1.0 / mse))


def _gaussian_window(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return w


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> float:
    """
    SSIM computed per-channel and averaged. Expects x,y in [-1,1], shape [1,3,H,W].
    """
    x01 = to_01(x)
    y01 = to_01(y)
    device = x01.device

    window = _gaussian_window(window_size, sigma, device=device)
    window = window.repeat(3, 1, 1, 1)  # [C,1,H,W]

    mu_x = F.conv2d(x01, window, padding=window_size // 2, groups=3)
    mu_y = F.conv2d(y01, window, padding=window_size // 2, groups=3)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x01 * x01, window, padding=window_size // 2, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y01 * y01, window, padding=window_size // 2, groups=3) - mu_y2
    sigma_xy = F.conv2d(x01 * y01, window, padding=window_size // 2, groups=3) - mu_xy

    C1 = (0.01**2)
    C2 = (0.03**2)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return float(ssim_map.mean().item())


# =============================
# Visualization
# =============================


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, int, int]:
    ckpt = torch.load(ckpt_path, map_location=device)
    timesteps = int(ckpt.get("timesteps", train.TIMESTEPS))
    image_size = int(ckpt.get("image_size", train.IMAGE_SIZE))
    model = train.UNet().to(device)
    state = ckpt["model"]
    model.load_state_dict(state)
    model.eval()
    epoch = int(ckpt.get("epoch", 0))
    return model, timesteps, image_size


def load_target(path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    tfm = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)


def read_losses_csv(path: Path) -> tuple[list[int], list[float]] | None:
    if not path.exists():
        return None
    epochs: list[int] = []
    losses: list[float] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["avg_loss"]))
    return epochs, losses


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, timesteps, image_size = load_model_from_ckpt(CKPT_PATH, device)
    sch = make_full_schedule(timesteps, device=device)

    # 1) Original + forward diffusion steps
    x0 = load_target(TARGET_IMAGE, image_size=image_size, device=device)
    ts_fwd = torch.linspace(0, timesteps - 1, N_FORWARD_STEPS).long().to(device)
    fwd = [x0.squeeze(0).cpu()]
    for t in ts_fwd:
        xt = q_sample(x0, torch.tensor([int(t.item())], device=device, dtype=torch.long), sch)
        fwd.append(xt.squeeze(0).cpu())
    fwd_grid = make_grid(torch.stack(fwd, 0), nrow=len(fwd), normalize=True, value_range=(-1, 1))
    save_image(fwd_grid, OUT_DIR / "forward_steps.png")

    # 2) Reconstruction with intermediate reverse steps (start from xT ~ q(xT|x0))
    tT = torch.full((1,), timesteps - 1, device=device, dtype=torch.long)
    xT = q_sample(x0, tT, sch)  # fully noised (random)

    capture_ts = set(torch.linspace(timesteps - 1, 0, N_REVERSE_STEPS).long().tolist())
    rev_snaps = []
    xt = xT
    for i in reversed(range(timesteps)):
        if i in capture_ts:
            rev_snaps.append(xt.squeeze(0).cpu())
        t = torch.full((1,), i, device=device, dtype=torch.long)
        xt = p_sample(model, xt, t, sch)
    x_recon = xt
    rev_snaps.append(x_recon.squeeze(0).cpu())

    rev_grid = make_grid(torch.stack(rev_snaps, 0), nrow=len(rev_snaps), normalize=True, value_range=(-1, 1))
    save_image(rev_grid, OUT_DIR / "reverse_steps.png")

    # Target vs Generated (reconstruction side-by-side)
    compare = make_grid(torch.cat([x0.cpu(), x_recon.cpu()], dim=0), nrow=2, normalize=True, value_range=(-1, 1))
    save_image(compare, OUT_DIR / "target_vs_reconstruction.png")

    # Quantitative evaluation
    psnr_val = psnr(x_recon, x0)
    ssim_val = ssim(x_recon, x0)
    with open(OUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"PSNR: {psnr_val:.4f}\n")
        f.write(f"SSIM: {ssim_val:.4f}\n")

    # 3) Generate images from pure noise
    gen = sample_from_noise(model, image_size=image_size, timesteps=timesteps, n=N_GENERATED_IMAGES, device=device)
    gen_grid = make_grid(gen.cpu(), nrow=N_GENERATED_IMAGES, normalize=True, value_range=(-1, 1))
    save_image(gen_grid, OUT_DIR / "generated_images.png")

    # 4) Training logs: Loss vs epochs plot
    losses = read_losses_csv(Path(train.OUT_DIR) / "losses.csv")
    if losses is None:
        plt.figure(figsize=(7, 4))
        plt.axis("off")
        plt.text(
            0.5,
            0.5,
            "losses.csv not found.\nRe-run training to generate checkpoints/losses.csv",
            ha="center",
            va="center",
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "loss_vs_epochs.png", dpi=150)
        plt.close()
    else:
        epochs, vals = losses
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, vals, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Avg MSE Loss")
        plt.title("Training Loss vs Epochs")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "loss_vs_epochs.png", dpi=150)
        plt.close()

    # One summary file with evidence pointers
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {CKPT_PATH}\n")
        f.write(f"timesteps={timesteps}\n")
        f.write(f"image_size={image_size}\n")
        f.write("Artifacts:\n")
        f.write("- forward_steps.png (original + >=5 forward steps)\n")
        f.write("- reverse_steps.png (>=5 reverse steps + final recon)\n")
        f.write("- generated_images.png (>=5 generated images)\n")
        f.write("- target_vs_reconstruction.png\n")
        f.write("- loss_vs_epochs.png\n")
        f.write("- metrics.txt (PSNR/SSIM)\n")

    print(f"Saved visualization outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()

