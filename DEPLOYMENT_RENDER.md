# Deploying BrainVis on Render (Lightweight Production Guide)

This guide provides step-by-step instructions for deploying **BrainVis** on [Render](https://render.com) as a lightweight, production-ready REST API and interactive web dashboard for **EEG wave analysis** and **visual stimulus image reconstruction**.

---

## 🚀 Key Lightweight Architecture Highlights

To satisfy low-memory limits (Render 512MB RAM free/starter tiers) and avoid multi-gigabyte build downloads:

1. **Lightweight CPU PyTorch**:
   Uses CPU-only PyTorch wheels (~200MB) instead of the default ~2.5GB CUDA packages.
2. **Zero Heavy Checkpoint Footprint**:
   The raw BrainVis pipeline relies on a 4.2GB Stable Diffusion v1.5 checkpoint (`v1-5-pruned-emaonly.ckpt`) and ~1.5GB CLIP weights, which cause instant Out-Of-Memory (OOM) crashes on standard cloud web services.
   - **Default Mode**: Built-in, ultra-fast neural visual synthesizer and cortical saliency generator running entirely in RAM (< 50MB) on CPU in milliseconds.
   - **Cloud Diffusion Mode (Optional)**: If you set the `HF_API_TOKEN` environment variable on Render, BrainVis offloads image generation to Hugging Face's serverless Stable Diffusion v1.5 API using the decoded EEG prompt—providing full photorealistic 512×512 diffusion generations with **0 MB local model weight download**.
3. **Interactive Web Dashboard**:
   Serves an interactive web interface at `/` with preset EEG waveforms (Panda, Mountain Bike, Pizza, Cat, Daisy, etc.), interactive trace charts, frequency band breakdowns, and instant image generation.
4. **Full REST API**:
   Automated interactive OpenAPI / Swagger documentation at `/docs`.

---

## 📋 Deployment Methods on Render

### Method 1: 1-Click Blueprint Deployment (Recommended)

1. **Push to GitHub**:
   Fork or push this repository to your GitHub account:
   ```bash
   git add .
   git commit -m "Add lightweight Render deployment setup"
   git push origin main
   ```
2. **Open Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com).
   - Click **New +** > **Blueprint**.
   - Connect your GitHub repository.
   - Render automatically detects `render.yaml` and sets up the web service.
3. **Configure Environment Variables (Optional)**:
   - `HF_API_TOKEN`: *(Optional)* Your Hugging Face user access token (free from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) if you want photorealistic Stable Diffusion generations.
4. Click **Apply**. Render will build and deploy your service.

---

### Method 2: Manual Web Service Setup

1. In Render Dashboard, click **New +** > **Web Service**.
2. Connect your GitHub repository.
3. Configure the settings:
   - **Name**: `brainvis-api` (or any name)
   - **Region**: Choose the closest region (e.g., Oregon, Frankfurt)
   - **Runtime**: `Python`
   - **Build Command**:
     ```bash
     pip install --no-cache-dir -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn app:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan**: `Free` or `Starter`
4. In **Advanced Settings**:
   - **Health Check Path**: `/health`
   - **Environment Variables**:
     - `PYTHON_VERSION`: `3.11.9`
     - `HF_API_TOKEN`: *(Optional)* `hf_xxxxxxxxxxxxxxxxxxxx`
5. Click **Create Web Service**.

---

### Method 3: Docker Deployment

If you prefer containerized deployment, Render also supports Docker:
1. Choose **Docker** as the environment.
2. Render will automatically build the included `Dockerfile`.

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Interactive Web Dashboard (UI) |
| `GET` | `/health` | Health check endpoint for uptime monitoring |
| `GET` | `/docs` | Interactive Swagger UI API documentation |
| `GET` | `/api/classes` | Returns all 40 ImageNet visual stimulus categories |
| `GET` | `/api/sample-eeg` | Returns synthetic sample EEG waveform data |
| `POST` | `/api/analyze` | Computes band powers, wave statistics, and predicted stimulus |
| `POST` | `/api/reconstruct` | Reconstructs image from label/category/EEG data |
| `POST` | `/api/pipeline` | Unified end-to-end: EEG waves &rarr; Analysis + Image Reconstruction |
| `POST` | `/api/upload-eeg` | File upload endpoint (`.npy`, `.csv`, `.json`, `.pth`) |

---

## 💻 Code Examples & Usage

### 1. Unified End-to-End Pipeline (Preset or Custom Waves)

```bash
curl -X POST "https://your-service-name.onrender.com/api/pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "preset": "panda",
    "top_k": 5,
    "prefer_cloud_diffusion": false
  }'
```

**Response Preview**:
```json
{
  "analysis": {
    "signal_shape": {"channels": 128, "time_samples": 440},
    "statistics": {
      "mean_amplitude_uv": -0.0412,
      "peak_to_peak_uv": 58.31,
      "rms_uv": 14.85,
      "spectral_entropy": 0.8124
    },
    "spectral_analysis": {
      "relative_powers_percent": {
        "delta": 18.4,
        "theta": 12.1,
        "alpha": 38.6,
        "beta": 24.2,
        "gamma": 6.7
      },
      "indices": {
        "alpha_theta_ratio": 3.19,
        "cognitive_workload_index": 0.48,
        "visual_attention_index": 0.61
      }
    },
    "predicted_stimulus": {
      "class_index": 8,
      "wnid": "n02510455",
      "label": "panda",
      "category": "animal",
      "confidence_percent": 94.2
    }
  },
  "reconstruction": {
    "reconstructed_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "saliency_map_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "engine": "built-in-neural-synth",
    "resolution": "512x512"
  }
}
```

---

### 2. Python Client Example

```python
import requests
import base64

API_URL = "https://your-service-name.onrender.com"

# 1. Check health
health = requests.get(f"{API_URL}/health").json()
print("Service status:", health["status"])

# 2. Run pipeline with EEG wave
payload = {
    "preset": "mountain bike",
    "top_k": 3
}
response = requests.post(f"{API_URL}/api/pipeline", json=payload)
data = response.json()

print("Decoded Visual Stimulus:", data["analysis"]["predicted_stimulus"]["label"])
print("Alpha Power:", data["analysis"]["spectral_analysis"]["relative_powers_percent"]["alpha"], "%")

# 3. Save the reconstructed image to disk
img_b64 = data["reconstruction"]["reconstructed_image_base64"].split(",")[1]
with open("reconstructed_stimulus.png", "wb") as f:
    f.write(base64.b64decode(img_b64))

print("Saved reconstructed image to reconstructed_stimulus.png")
```

---

### 3. Uploading Custom EEG Files

You can upload `.npy`, `.csv`, `.json`, or `.pth` files directly:

```bash
curl -X POST "https://your-service-name.onrender.com/api/upload-eeg" \
  -F "file=@sample_eeg.npy"
```
