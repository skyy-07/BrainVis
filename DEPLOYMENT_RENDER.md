# BrainVis Deployment Guide: Authentic Architecture & Hybrid Worker Setup

This guide documents the authentic **BrainVis (ICASSP 2025)** system architecture, the strict input tensor contract, and how to operate the system either as a **Native CUDA Worker** (with real weights and cascaded diffusion) or as a **Cloud API Gateway** on Render.

---

## 🧠 Authentic BrainVis System Architecture

In the BrainVis paper (*"BrainVis: Exploring the Bridge between Brain and Visual Signals via Image Reconstruction"*), the system pipeline consists of three core stages:

```
Raw EEG Signal (128 Channels, 440 Time Samples @ 250 Hz)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │   BrainVis TimeFreqEncoder       │
        │   • TimeEncoder (Transformer,    │
        │     1024-d, 8 layers, 16 heads)  │
        │   • FreqEncoder (FFT + LSTM,     │
        │     128-d latent features)       │
        │   • 40-Class Stimulus Logits     │
        └─────────────────┬────────────────┘
                         │ (1152-d multi-domain representation)
                         ▼
        ┌──────────────────────────────────┐
        │        BrainVis AlignNet         │
        │   • Multi-layer Tanh Residuals   │
        │   • Maps EEG to CLIP space       │
        │   • Output: [1, 77, 768] Latent  │
        └─────────────────┬────────────────┘
                         │ (77×768 CLIP text-embedding conditioning)
                         ▼
        ┌──────────────────────────────────┐
        │       Cascaded Diffusion         │
        │   Stage 1: PLMS Latent Diffusion │
        │            (dc_ldm conditioned   │
        │             on EEG-CLIP latent)  │
        │   Stage 2: SD 1.5 Img2Img        │
        │            (prompt + coarse img  │
        │             refinement @ s=0.85) │
        └─────────────────┬────────────────┘
                         │
                         ▼
             Reconstructed Visual Stimulus
```

---

## 📐 Strict Input Tensor Contract

BrainVis **strictly enforces** the experimental EEG recording contract:
- **Number of Channels**: Exactly `128` channels in the standard 10-20 montage.
- **Time Duration**: Exactly `440` time samples (samples 20 to 460 from a 250 Hz recording, corresponding to 1.76 seconds of visual stimulus presentation).
- **Accepted Shapes**: `(128, 440)` or `(440, 128)` or `(batch, 128, 440)`.
- If an input tensor with any other channel count or duration is provided, the API strictly rejects it with an **HTTP 422 Unprocessable Entity**.

---

## 🔑 Obtaining Checkpoints

The authors of BrainVis (`fu2025brainvis`) do **not** distribute model weights directly inside the Git repository.

1. **BrainVis Alignment Weights (`clipfinetune_model.pkl`)**:
   - Request from the paper authors via email: `hfu006@e.ntu.edu.sg`
   - Place the received checkpoint at:
     ```
     exp/clipfinetune_model.pkl
     ```
   - When loaded, the service automatically detects the checkpoint and marks `checkpoint_loaded: true`.
2. **Stable Diffusion 1.5 Weights (`v1-5-pruned-emaonly.ckpt`)**:
   - Download the official Stable Diffusion v1.5 pruned checkpoint (4.27 GB):
     ```bash
     wget -O pretrained_model/v1-5-pruned-emaonly.ckpt https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
     ```

---

## 🖥️ Running the Native CUDA Worker (for MnemoSplat / GPU Inference)

For genuine cascaded diffusion and EEG-to-image reconstruction, run the dedicated CUDA worker on an NVIDIA GPU machine (e.g. local RTX 3090/4090/A100, RunPod, or Modal):

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn omegaconf einops diffusers transformers pillow numpy scipy

# 2. Start the native CUDA worker
python native_worker.py --host 0.0.0.0 --port 8080
```

The worker exposes:
- `POST /reconstruct`: Receives `(128, 440)` EEG tensor, runs `AlignNet` &rarr; `(77, 768)` CLIP latent &rarr; Stage 1 PLMSSampler &rarr; Stage 2 SD Img2Img &rarr; Returns true reconstructed image.
- `GET /status`: Reports GPU VRAM, device, and checkpoint status.

---

## ☁️ Deploying on Render (Hybrid API Gateway)

Render's standard/free tiers provide 512MB RAM and CPU only, which cannot host a 4.2GB diffusion model locally. Render serves as the **API Gateway**:

1. **Push your repository to GitHub**:
   ```bash
   git add .
   git commit -m "Update authentic BrainVis gateway and CUDA worker"
   git push origin main
   ```
2. **Deploy on Render**:
   - Connect your repo on [dashboard.render.com](https://dashboard.render.com).
   - Use the included [`render.yaml`](render.yaml) blueprint.
3. **Connect to your Native CUDA Worker (Hybrid Mode)**:
   - In Render Web Service settings, set the environment variable:
     ```
     CUDA_WORKER_URL=http://your-gpu-worker-ip:8080
     ```
   - When configured, Render validates the strict `(128, 440)` EEG input contract, runs BrainVis feature validation, delegates the cascaded diffusion to the GPU worker, and returns the authentic reconstruction to MnemoSplat or your frontend.

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/model-status` | `GET` | Reports loaded checkpoint path, input contract, and GPU worker status |
| `GET /health` | `GET` | Health check endpoint |
| `POST /api/pipeline` | `POST` | Unified pipeline: validates contract, extracts features, and reconstructs |
| `POST /api/analyze` | `POST` | Strict EEG forward pass through `TimeFreqEncoder` and `AlignNet` |
| `POST /api/upload-eeg` | `POST` | Validates and processes uploaded `.npy`, `.csv`, `.json`, or `.pth` EEG files |
