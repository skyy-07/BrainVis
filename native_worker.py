"""
Native CUDA Worker for BrainVis (ICASSP 2025)
Runs the authentic full-stack BrainVis pipeline:
- Strict (128, 440) EEG input contract
- Authentic BrainVis AlignNet (EEG -> 77x768 CLIP latent embedding)
- Stage 1: Latent Diffusion Model (PLMS Sampler conditioned on EEG-CLIP latent)
- Stage 2: Stable Diffusion Img2Img Cascaded Refinement

Run this on a machine with an NVIDIA GPU (local CUDA, RunPod, Modal, or Render GPU Instance).
Usage:
    python native_worker.py --port 8080 --device cuda
"""

import os
import io
import base64
import argparse
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from service.brainvis_native import BrainVisNativePipeline, BRAINVIS_CLASSES, EXPECTED_CHANNELS, EXPECTED_TIME_SAMPLES

app = FastAPI(
    title="BrainVis Native CUDA Worker",
    description="Dedicated GPU worker for authentic BrainVis cascaded diffusion reconstruction.",
    version="1.0.0"
)

# Global model state
pipeline = None
stage1_ldm = None
stage1_sampler = None
stage2_pipe = None
cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

class WorkerEEGRequest(BaseModel):
    eeg_data: List[List[float]] = Field(
        ...,
        description=f"Strict 2D array of EEG wave values: ({EXPECTED_CHANNELS}, {EXPECTED_TIME_SAMPLES}) or ({EXPECTED_TIME_SAMPLES}, {EXPECTED_CHANNELS})."
    )
    ddim_steps: int = Field(50, description="DDIM sampling steps for Stage 1 diffusion.")
    strength: float = Field(0.85, description="Img2Img refinement strength for Stage 2.")
    guidance_scale: float = Field(7.5, description="Classifier-free guidance scale.")

def load_diffusion_models(sd_ckpt_path="pretrained_model/v1-5-pruned-emaonly.ckpt", yaml_path="pretrained_model/config15.yaml"):
    global stage1_ldm, stage1_sampler, stage2_pipe
    if not os.path.isfile(sd_ckpt_path):
        print(f"[Worker] Stable Diffusion weights not found at {sd_ckpt_path}. Stage 1/2 diffusion disabled until weights mounted.")
        return False

    try:
        from omegaconf import OmegaConf
        from dc_ldm.util import instantiate_from_config
        from dc_ldm.models.diffusion.plms import PLMSSampler
        from diffusers import StableDiffusionImg2ImgPipeline

        # 1. Stage 1 LDM
        config = OmegaConf.load(yaml_path)
        stage1_ldm = instantiate_from_config(config.model)
        pl_sd = torch.load(sd_ckpt_path, map_location=cuda_device)["state_dict"]
        stage1_ldm.load_state_dict(pl_sd, strict=False)
        stage1_ldm.cond_stage_trainable = False
        stage1_ldm.re_init_ema()
        stage1_ldm = stage1_ldm.to(cuda_device)
        stage1_ldm.eval()
        stage1_sampler = PLMSSampler(stage1_ldm)

        # 2. Stage 2 Img2Img
        stage2_pipe = StableDiffusionImg2ImgPipeline.from_single_file(sd_ckpt_path).to(cuda_device)
        print("[Worker] Successfully initialized Cascaded Diffusion models (Stage 1 LDM + Stage 2 SD).")
        return True
    except Exception as e:
        print(f"[Worker] Failed loading diffusion models: {e}")
        return False

@app.on_event("startup")
def startup_event():
    global pipeline
    print(f"[Worker] Starting BrainVis Native Worker on device: {cuda_device}")
    pipeline = BrainVisNativePipeline(device=cuda_device)
    load_diffusion_models()

@app.get("/status")
def get_worker_status():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU (CPU only)"
    vram_free = None
    vram_total = None
    if torch.cuda.is_available():
        free_b, tot_b = torch.cuda.mem_get_info()
        vram_free = round(free_b / (1024 ** 3), 2)
        vram_total = round(tot_b / (1024 ** 3), 2)

    return {
        "status": "online",
        "device": cuda_device,
        "gpu_name": gpu_name,
        "vram_gb": {"free": vram_free, "total": vram_total},
        "brainvis_checkpoint_loaded": pipeline.is_checkpoint_loaded if pipeline else False,
        "checkpoint_source": pipeline.checkpoint_source if pipeline else None,
        "diffusion_stage1_ready": stage1_ldm is not None,
        "diffusion_stage2_ready": stage2_pipe is not None,
        "input_contract": f"({EXPECTED_CHANNELS}, {EXPECTED_TIME_SAMPLES}) float32 tensor"
    }

@app.post("/reconstruct")
def run_native_reconstruction(req: WorkerEEGRequest):
    """
    Runs the genuine BrainVis cascaded diffusion reconstruction from raw EEG.
    """
    eeg_np = np.array(req.eeg_data, dtype=np.float32)

    try:
        # 1. Forward BrainVis AlignNet
        forward_res = pipeline.forward(eeg_np)
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))

    top_pred = forward_res["predicted_stimulus"]
    prompt = top_pred["label"] if top_pred else "visual stimulus"

    # Verify if diffusion models are ready
    if stage1_ldm is None or stage2_pipe is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Diffusion pipeline not initialized. Make sure 'v1-5-pruned-emaonly.ckpt' is placed "
                "in pretrained_model/ and CUDA GPU is available."
            )
        )

    # 2. Stage 1: Latent Diffusion Sampling
    val_eeg = pipeline.validate_input_contract(eeg_np)
    tensor_in = torch.tensor(val_eeg, dtype=torch.float32, device=cuda_device)

    with torch.no_grad():
        clip_encoded = pipeline.align_net(tensor_in) # (1, 59136)
        latent = clip_encoded[0].reshape(1, 77, 768).to(cuda_device)

        # Stage 1 conditioning
        shape = (4, 64, 64) # Latent shape
        samples_ddim, _ = stage1_sampler.sample(
            S=req.ddim_steps,
            conditioning=latent,
            batch_size=1,
            shape=shape,
            verbose=False
        )
        x_samples_ddim = stage1_ldm.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        stage1_img_np = (255.0 * x_samples_ddim[0].detach().cpu().permute(1, 2, 0).numpy()).astype(np.uint8)
        stage1_pil = Image.fromarray(stage1_img_np)

        # 3. Stage 2: Stable Diffusion Img2Img Refinement
        stage2_out = stage2_pipe(
            prompt=[prompt],
            image=[stage1_pil],
            strength=req.strength,
            guidance_scale=req.guidance_scale,
            num_inference_steps=50
        ).images[0]

    # Convert to Base64
    buf = io.BytesIO()
    stage2_out.save(buf, format="PNG")
    b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    buf1 = io.BytesIO()
    stage1_pil.save(buf1, format="PNG")
    b64_stage1 = "data:image/png;base64," + base64.b64encode(buf1.getvalue()).decode("utf-8")

    return {
        "analysis": forward_res,
        "reconstruction": {
            "reconstructed_image_base64": b64_str,
            "stage1_coarse_base64": b64_stage1,
            "engine": "native-brainvis-cascaded-diffusion",
            "prompt": prompt,
            "resolution": f"{stage2_out.width}x{stage2_out.height}"
        }
    }

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("native_worker:app", host=args.host, port=args.port, reload=False)
