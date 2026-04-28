import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from torchvision.utils import make_grid
import numpy as np

import train
from visualize import (
    load_model_from_ckpt,
    make_full_schedule,
    q_sample,
    p_sample,
    to_01,
)

import io

def frames_to_gif(frames, duration_ms=300): #little helper to turn the steps into a temp gif
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    buf.seek(0)
    return buf


@st.cache_resource
def load_model_and_config():
    """Load model once and reuse across app sessions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(train.OUT_DIR) / "ckpt_latest.pt"

    if not ckpt_path.exists():
        st.error(f"Checkpoint not found at {ckpt_path}")
        st.info("Please ensure you have trained a model and saved the checkpoint.")
        st.stop()

    model, timesteps, image_size = load_model_from_ckpt(ckpt_path, device)
    sch = make_full_schedule(timesteps, device=device)

    return model, timesteps, image_size, device, sch


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """Convert tensor [-1, 1] to PIL Image [0, 255]."""
    x01 = to_01(x.squeeze(0))  # [-1, 1] -> [0, 1]
    x_np = (x01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(x_np)


def load_and_preprocess(uploaded_file, image_size: int, device: torch.device) -> torch.Tensor:
    """Load image from upload and preprocess."""
    img = Image.open(uploaded_file).convert("RGB")
    tfm = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),  # [0, 1]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
        ]
    )
    x = tfm(img).unsqueeze(0).to(device)
    return x


def get_denoising_frames(
    model: torch.nn.Module,
    x_input: torch.Tensor,
    timesteps: int,
    device: torch.device,
    sch,
    n_frames: int = 10,
) -> list[Image.Image]:
    # generate denoising steps.
    # fully noised (timestep T-1)
    t_end = torch.full((1,), timesteps - 1, device=device, dtype=torch.long)
    x_noisy = q_sample(x_input, t_end, sch)

    # timesteps to capture (even spread)
    capture_ts = set(torch.linspace(timesteps - 1, 0, n_frames).long().tolist())
    frames = []
    xt = x_noisy

    with torch.no_grad():
        for i in reversed(range(timesteps)):
            if i in capture_ts:
                frames.append(tensor_to_pil(xt.clone()))

            t = torch.full((1,), i, device=device, dtype=torch.long)
            xt = p_sample(model, xt, t, sch)

    # Add final frame
    frames.append(tensor_to_pil(xt))
    return frames

def main():
    st.set_page_config(page_title="CelebGen - Diffusion Model", layout="wide")
    st.title("CelebGen - Image Denoising with Diffusion")
    st.markdown(
        "Upload an image to see the DDPM denoising process in action. "
        "The model will progressively denoise from noise to final image."
    )

    try:
        model, timesteps, image_size, device, sch = load_model_and_config()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    with st.sidebar:
        st.header("Settings")
        n_frames = st.slider(
            "Number of denoising frames",
            min_value=3,
            max_value=30,
            value=10,
            help="More frames = smoother animation (but slower)",
        )
        animation_speed = st.slider(
            "Animation speed (ms per frame)",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
        )
        st.markdown("---")
        st.caption(f"Model: {train.OUT_DIR}/ckpt_latest.pt")
        st.caption(f"Image size: {image_size}×{image_size}")
        st.caption(f"Timesteps: {timesteps}")
        st.caption(f"Device: {device}")

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Recommended: Face images for best results",
        )

    with col2:
        st.subheader("Denoising Animation")
        if uploaded_file is None:
            st.info("Upload an image to start")
        else:
            try:
                # Load and preprocess
                with st.spinner("Loading image..."):
                    x_input = load_and_preprocess(uploaded_file, image_size, device)

                # Generate frames
                with st.spinner("Generating denoising frames..."):
                    frames = get_denoising_frames(
                        model, x_input, timesteps, device, sch, n_frames=n_frames
                    )

                # # Display animation
                # st.image(
                #     frames,
                #     caption=[f"Step {i}/{len(frames)-1}" for i in range(len(frames))],
                #     width=300,
                #     use_container_width=True,
                # )

                # Create GIF
                gif_bytes = frames_to_gif(frames, duration_ms=animation_speed)

                # Display animation
                st.image(gif_bytes, caption="Denoising Animation", use_container_width=True)

                # Show final result in detail
                st.subheader("Final Result")
                col_final_l, col_final_r = st.columns([1, 1])
                with col_final_l:
                    st.caption("Original Input")
                    orig_pil = tensor_to_pil(x_input)
                    st.image(orig_pil, use_container_width=True)
                with col_final_r:
                    st.caption("Denoised Output")
                    st.image(frames[-1], use_container_width=True)

                # Download final image
                final_img = frames[-1]
                buf = np.array(final_img)
                st.download_button(
                    label="⬇Download denoised image",
                    data=Image.fromarray(buf).tobytes(),
                    file_name="denoised_output.png",
                    mime="image/png",
                )

            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.exception(e)


if __name__ == "__main__":
    main()
