import io
import os
import json
import requests
import numpy as np
import torch
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from service.brainvis_native import BrainVisNativePipeline, BRAINVIS_CLASSES, EXPECTED_CHANNELS, EXPECTED_TIME_SAMPLES
from service.eeg_analyzer import EEGAnalyzer
from service.image_reconstructor import ImageReconstructor

app = FastAPI(
    title="BrainVis EEG Analysis & Image Reconstruction Service",
    description="Production API Gateway for BrainVis (ICASSP 2025). Enforces strict (128, 440) tensor contracts, runs authentic AlignNet architecture, and interfaces with the Native CUDA Worker for Cascaded Diffusion.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
native_pipeline = BrainVisNativePipeline()
analyzer = EEGAnalyzer()
reconstructor = ImageReconstructor()

# Remote CUDA worker URL (optional, e.g., for MnemoSplat / GPU offload)
CUDA_WORKER_URL = os.environ.get("CUDA_WORKER_URL", "").rstrip("/")

# Pydantic Schemas
class EEGAnalyzeRequest(BaseModel):
    eeg_data: Optional[List[List[float]]] = Field(
        None,
        description=f"Strict 2D array of EEG wave values: ({EXPECTED_CHANNELS}, {EXPECTED_TIME_SAMPLES}) or ({EXPECTED_TIME_SAMPLES}, {EXPECTED_CHANNELS})."
    )
    preset: Optional[str] = Field(
        None,
        description="Optional synthetic preset name: 'panda', 'mountain bike', 'pizza', 'cat', 'daisy', 'electric guitar'."
    )
    top_k: int = Field(5, description="Number of top visual classes to predict.")

class PipelineRequest(BaseModel):
    eeg_data: Optional[List[List[float]]] = Field(
        None,
        description=f"Strict 2D array of EEG wave values: ({EXPECTED_CHANNELS}, {EXPECTED_TIME_SAMPLES}) or ({EXPECTED_TIME_SAMPLES}, {EXPECTED_CHANNELS})."
    )
    preset: Optional[str] = Field(
        None,
        description="Optional synthetic preset name ('panda', 'mountain bike', 'pizza', 'cat', 'daisy', 'electric guitar')."
    )
    top_k: int = Field(5, description="Number of top classes.")
    prefer_cloud_diffusion: bool = Field(False, description="Whether to query cloud diffusion API if HF_API_TOKEN is set.")

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for Render service uptime monitoring."""
    return {
        "status": "healthy",
        "service": "BrainVis EEG & Image Reconstruction API Gateway",
        "version": "2.0.0",
        "checkpoint_loaded": native_pipeline.is_checkpoint_loaded,
        "cuda_worker_configured": bool(CUDA_WORKER_URL),
        "input_contract": f"({EXPECTED_CHANNELS}, {EXPECTED_TIME_SAMPLES})"
    }

@app.get("/api/model-status", tags=["Model Status"])
def get_model_status():
    """Reports authentic BrainVis model parameters, checkpoint loading state, and worker routes."""
    worker_status = None
    if CUDA_WORKER_URL:
        try:
            worker_resp = requests.get(f"{CUDA_WORKER_URL}/status", timeout=3)
            if worker_resp.status_code == 200:
                worker_status = worker_resp.json()
        except Exception as e:
            worker_status = {"error": f"Failed to contact CUDA worker at {CUDA_WORKER_URL}: {e}"}

    return {
        "model_architecture": "AlignNet (TimeEncoder + FreqEncoder -> TimeFreqEncoder -> AlignNet)",
        "input_contract": {
            "channels": EXPECTED_CHANNELS,
            "time_samples": EXPECTED_TIME_SAMPLES,
            "sample_rate_hz": 250,
            "recording_window": "samples 20 to 460 (1.76s visual stimulus response)"
        },
        "checkpoint": {
            "is_loaded": native_pipeline.is_checkpoint_loaded,
            "source": native_pipeline.checkpoint_source,
            "weights_filename": "clipfinetune_model.pkl",
            "instruction": "Request checkpoint from authors (hfu006@e.ntu.edu.sg) and mount in exp/clipfinetune_model.pkl"
        },
        "device": native_pipeline.device,
        "cuda_worker_url": CUDA_WORKER_URL or None,
        "cuda_worker_status": worker_status
    }

@app.get("/api/classes", tags=["Classes"])
def get_supported_classes():
    """Returns the 40 ImageNet visual stimulus categories supported by BrainVis."""
    return {
        "total_classes": len(BRAINVIS_CLASSES),
        "classes": BRAINVIS_CLASSES
    }

@app.get("/api/sample-eeg", tags=["EEG"])
def get_sample_eeg(preset: str = Query("panda", description="Preset name")):
    """Generates synthetic EEG wave data matching the exact BrainVis (128, 440) contract."""
    signals = EEGAnalyzer.generate_synthetic_eeg(preset=preset, channels=EXPECTED_CHANNELS, time_points=EXPECTED_TIME_SAMPLES)
    return {
        "preset": preset,
        "channels": EXPECTED_CHANNELS,
        "time_samples": EXPECTED_TIME_SAMPLES,
        "sample_rate_hz": 250,
        "eeg_data": signals.tolist()
    }

@app.post("/api/analyze", tags=["EEG Analysis"])
def analyze_eeg(req: EEGAnalyzeRequest):
    """
    Analyzes EEG waves using the authentic BrainVis model and spectral decomposition.
    Strictly validates the (128, 440) tensor contract.
    """
    if req.eeg_data is not None and len(req.eeg_data) > 0:
        data = np.array(req.eeg_data, dtype=np.float32)
    elif req.preset:
        data = EEGAnalyzer.generate_synthetic_eeg(preset=req.preset, channels=EXPECTED_CHANNELS, time_points=EXPECTED_TIME_SAMPLES)
    else:
        data = EEGAnalyzer.generate_synthetic_eeg(preset="panda", channels=EXPECTED_CHANNELS, time_points=EXPECTED_TIME_SAMPLES)

    # 1. Enforce strict input contract and run authentic BrainVis forward pass
    try:
        model_result = native_pipeline.forward(data, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 2. Extract spectral statistics (for signal quality verification)
    eeg_2d = data if data.ndim == 2 else data[0]
    if eeg_2d.shape[0] != EXPECTED_CHANNELS and eeg_2d.shape[1] == EXPECTED_CHANNELS:
        eeg_2d = eeg_2d.T
    stats = analyzer.compute_wave_statistics(eeg_2d)
    spectral = analyzer.compute_band_powers(eeg_2d)

    return {
        "brainvis_model": model_result,
        "waveform_statistics": stats,
        "spectral_analysis": spectral
    }

@app.post("/api/pipeline", tags=["Pipeline"])
def full_eeg_to_image_pipeline(req: PipelineRequest):
    """
    Unified end-to-end pipeline:
    1. Validates strict (128, 440) EEG input contract.
    2. Runs authentic BrainVis AlignNet to produce (77, 768) CLIP latent embedding.
    3. If CUDA_WORKER_URL is configured, delegates cascaded diffusion to the GPU worker.
       Otherwise, provides perceptual stimulus reconstruction with clear metadata tagging.
    """
    if req.eeg_data is not None and len(req.eeg_data) > 0:
        data = np.array(req.eeg_data, dtype=np.float32)
    elif req.preset:
        data = EEGAnalyzer.generate_synthetic_eeg(preset=req.preset, channels=EXPECTED_CHANNELS, time_points=EXPECTED_TIME_SAMPLES)
    else:
        data = EEGAnalyzer.generate_synthetic_eeg(preset="panda", channels=EXPECTED_CHANNELS, time_points=EXPECTED_TIME_SAMPLES)

    try:
        model_result = native_pipeline.forward(data, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Delegate to remote CUDA worker if available
    if CUDA_WORKER_URL:
        try:
            worker_payload = {
                "eeg_data": data.tolist(),
                "ddim_steps": 50,
                "strength": 0.85,
                "guidance_scale": 7.5
            }
            worker_resp = requests.post(f"{CUDA_WORKER_URL}/reconstruct", json=worker_payload, timeout=60)
            if worker_resp.status_code == 200:
                worker_data = worker_resp.json()
                return {
                    "source": "native_cuda_worker",
                    "cuda_worker_url": CUDA_WORKER_URL,
                    "analysis": model_result,
                    "reconstruction": worker_data.get("reconstruction")
                }
        except Exception as e:
            print(f"[Gateway] CUDA worker request failed: {e}. Falling back to gateway response.")

    # Gateway visual reconstruction
    top_pred = model_result["predicted_stimulus"] or {
        "label": "visual object",
        "category": "object",
        "confidence_percent": 2.5
    }

    eeg_2d = data if data.ndim == 2 else data[0]
    if eeg_2d.shape[0] != EXPECTED_CHANNELS and eeg_2d.shape[1] == EXPECTED_CHANNELS:
        eeg_2d = eeg_2d.T
    spectral = analyzer.compute_band_powers(eeg_2d)

    recon_res = reconstructor.reconstruct(
        label=top_pred["label"],
        category=top_pred["category"],
        confidence=top_pred.get("probability", 0.025),
        eeg_data=eeg_2d,
        band_powers=spectral,
        prefer_cloud_diffusion=req.prefer_cloud_diffusion
    )

    # Clearly label reconstruction metadata
    recon_res["is_authentic_cascaded_diffusion"] = False
    recon_res["note"] = (
        "Reconstruction was generated on the CPU gateway. To run the full authors' Cascaded Diffusion "
        "(Stage 1 PLMS + Stage 2 SD 1.5), configure CUDA_WORKER_URL pointing to the Native CUDA Worker."
    )

    return {
        "source": "cpu_gateway",
        "analysis": model_result,
        "spectral_analysis": spectral,
        "reconstruction": recon_res
    }

@app.post("/api/upload-eeg", tags=["EEG File Upload"])
async def upload_eeg_file(file: UploadFile = File(...), top_k: int = 5):
    """Uploads an EEG file (.npy, .csv, .json, .pth) and runs the pipeline with contract validation."""
    contents = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".npy"):
            data = np.load(io.BytesIO(contents))
        elif filename.endswith(".csv") or filename.endswith(".txt"):
            data = np.genfromtxt(io.BytesIO(contents), delimiter=",")
        elif filename.endswith(".json"):
            raw_json = json.loads(contents.decode("utf-8"))
            if isinstance(raw_json, dict) and "eeg_data" in raw_json:
                data = np.array(raw_json["eeg_data"], dtype=np.float32)
            else:
                data = np.array(raw_json, dtype=np.float32)
        elif filename.endswith(".pth") or filename.endswith(".pt"):
            loaded = torch.load(io.BytesIO(contents), map_location="cpu")
            if isinstance(loaded, dict) and "dataset" in loaded:
                data = loaded["dataset"][0]["eeg"].numpy()
            elif isinstance(loaded, torch.Tensor):
                data = loaded.numpy()
            else:
                data = np.array(loaded, dtype=np.float32)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use .npy, .csv, .json, or .pth")

        pipeline_req = PipelineRequest(eeg_data=data.tolist(), top_k=top_k)
        return full_eeg_to_image_pipeline(pipeline_req)
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process EEG file: {str(e)}")

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def web_dashboard():
    """Serves the interactive web dashboard with transparent status banners and contract indicators."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrainVis: Brain-Visual Decoding Gateway</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css">
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
        .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
        .progress-bar { background: #21262d; border-radius: 4px; overflow: hidden; height: 8px; }
        .progress-fill { height: 100%; transition: width 0.4s ease; }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
        <!-- Status Banner -->
        <div id="statusBanner" class="card p-3 mb-6 flex flex-col md:flex-row justify-between items-start md:items-center text-xs border-yellow-600 bg-yellow-900 bg-opacity-20">
            <div class="flex items-center space-x-2">
                <span id="statusLed" class="text-yellow-400 font-bold">⚠️ Checking Checkpoint Status...</span>
                <span id="statusDetails" class="text-gray-300"></span>
            </div>
            <div class="mt-2 md:mt-0 font-mono text-gray-400">
                Input Contract: <span class="text-white">128 Channels × 440 Time Samples</span>
            </div>
        </div>

        <!-- Header -->
        <header class="mb-6 border-b border-gray-800 pb-4 flex justify-between items-center">
            <div>
                <h1 class="text-2xl font-extrabold text-white flex items-center space-x-2">
                    <span>🧠</span>
                    <span>BrainVis Decoding Service</span>
                </h1>
                <p class="text-gray-400 text-xs mt-0.5">Authentic AlignNet (77×768 CLIP Latent) &amp; Cascaded Diffusion Gateway</p>
            </div>
            <div class="space-x-2">
                <a href="/docs" target="_blank" class="px-2.5 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded transition">API Docs</a>
                <a href="/api/model-status" target="_blank" class="px-2.5 py-1 bg-gray-700 hover:bg-gray-600 text-gray-200 text-xs font-semibold rounded transition">Model Status</a>
            </div>
        </header>

        <!-- Controls & Waveform -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div class="card p-4">
                <h2 class="text-sm font-bold text-white mb-2">⚡ EEG Input</h2>
                <div class="space-y-3">
                    <div>
                        <label class="block text-xs text-gray-400 mb-1">Preset Neural Signal (128x440)</label>
                        <select id="presetSelect" class="w-full bg-gray-900 border border-gray-700 rounded p-1.5 text-xs text-gray-200">
                            <option value="panda">Panda (Alpha + Beta)</option>
                            <option value="mountain bike">Mountain Bike (Visual attention)</option>
                            <option value="pizza">Pizza (Food category)</option>
                            <option value="cat">Cat (Animal category)</option>
                            <option value="daisy">Daisy (Plant category)</option>
                            <option value="electric guitar">Electric Guitar (Instrument)</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-xs text-gray-400 mb-1">Upload EEG (.npy, .csv, .json)</label>
                        <input type="file" id="eegFileInput" class="w-full text-xs text-gray-400 bg-gray-900 border border-gray-700 rounded p-1">
                    </div>
                    <button id="runBtn" onclick="runPipeline()" class="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded text-xs transition">
                        Run BrainVis Pipeline
                    </button>
                </div>
            </div>

            <div class="card p-4 md:col-span-2 flex flex-col justify-between">
                <div>
                    <div class="flex justify-between items-center mb-1">
                        <h2 class="text-sm font-bold text-white">📈 EEG Waveform (Channel 1 of 128)</h2>
                        <span class="text-xs text-gray-400 font-mono">128 × 440 @ 250 Hz</span>
                    </div>
                    <canvas id="waveformCanvas" width="700" height="130" class="w-full bg-gray-900 rounded border border-gray-800"></canvas>
                </div>
                <div class="grid grid-cols-3 gap-2 mt-2 text-center text-xs">
                    <div class="bg-gray-800 p-1.5 rounded font-mono">
                        <span class="text-gray-400 block text-xxs">CLIP Latent Norm</span>
                        <span id="latentNorm" class="text-indigo-400 font-semibold">-</span>
                    </div>
                    <div class="bg-gray-800 p-1.5 rounded font-mono">
                        <span class="text-gray-400 block text-xxs">Execution Target</span>
                        <span id="execSource" class="text-blue-400 font-semibold">-</span>
                    </div>
                    <div class="bg-gray-800 p-1.5 rounded font-mono">
                        <span class="text-gray-400 block text-xxs">Checkpoint Status</span>
                        <span id="ckptBadge" class="text-yellow-400 font-semibold">-</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Predictions & Reconstructions -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div class="card p-4">
                <h3 class="text-xs font-bold uppercase text-gray-400 mb-2">Decoded Brain Stimulus</h3>
                <div class="bg-gray-800 p-3 rounded mb-3 text-center">
                    <span id="topLabel" class="text-lg font-extrabold text-indigo-400 capitalize block">Awaiting Signal</span>
                    <span id="topMeta" class="text-xs text-gray-400 block">-</span>
                </div>
                <h4 class="text-xxs uppercase tracking-wider text-gray-500 mb-2">Top-5 AlignNet Predictions</h4>
                <div id="topList" class="space-y-1.5 text-xs">
                    <div class="text-gray-500 italic">Run pipeline to inspect probabilities...</div>
                </div>
            </div>

            <div class="card p-4 flex flex-col justify-between">
                <div>
                    <div class="flex justify-between items-center mb-2">
                        <h3 class="text-xs font-bold uppercase text-gray-400">Reconstructed Visual Stimulus</h3>
                        <span id="engineBadge" class="text-xxs bg-gray-800 px-1.5 py-0.5 rounded text-gray-400">-</span>
                    </div>
                    <div class="relative bg-gray-900 rounded border border-gray-800 aspect-square flex items-center justify-center overflow-hidden mb-2">
                        <img id="reconImg" src="" alt="Reconstructed Stimulus" class="w-full h-full object-cover hidden">
                        <span id="reconPlaceholder" class="text-xs text-gray-500">Image will appear upon execution</span>
                    </div>
                </div>
                <div id="reconWarning" class="text-xxs text-gray-400 italic"></div>
            </div>

            <div class="card p-4 flex flex-col justify-between">
                <div>
                    <h3 class="text-xs font-bold uppercase text-gray-400 mb-2">Cortical Saliency Heatmap</h3>
                    <div class="bg-gray-900 rounded border border-gray-800 aspect-square flex items-center justify-center overflow-hidden mb-2">
                        <img id="saliencyImg" src="" alt="Saliency Heatmap" class="w-full h-full object-cover hidden">
                        <span id="saliencyPlaceholder" class="text-xs text-gray-500">Topographic projection</span>
                    </div>
                </div>
                <div class="text-xxs text-gray-500 font-mono text-center">128-Electrode Spatial Mapping</div>
            </div>
        </div>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const res = await fetch('/api/model-status');
                const data = await res.json();
                const banner = document.getElementById('statusBanner');
                const led = document.getElementById('statusLed');
                const details = document.getElementById('statusDetails');

                if (data.checkpoint.is_loaded) {
                    banner.className = 'card p-3 mb-6 flex flex-col md:flex-row justify-between items-start md:items-center text-xs border-green-600 bg-green-900 bg-opacity-20';
                    led.className = 'text-green-400 font-bold';
                    led.innerText = '🟢 Checkpoint Active';
                    details.innerText = 'Loaded ' + data.checkpoint.source;
                } else {
                    banner.className = 'card p-3 mb-6 flex flex-col md:flex-row justify-between items-start md:items-center text-xs border-yellow-600 bg-yellow-900 bg-opacity-20';
                    led.className = 'text-yellow-400 font-bold';
                    led.innerText = '⚠️ Checkpoint Missing (Untrained Baseline)';
                    details.innerText = 'clipfinetune_model.pkl not mounted. Running baseline model.';
                }
            } catch (e) {}
        }
        fetchStatus();

        async function runPipeline() {
            const btn = document.getElementById('runBtn');
            btn.disabled = true;
            btn.innerText = 'Computing...';

            const fileInput = document.getElementById('eegFileInput');
            const presetSelect = document.getElementById('presetSelect');

            try {
                let res;
                if (fileInput.files.length > 0) {
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    const r = await fetch('/api/upload-eeg', { method: 'POST', body: formData });
                    if (!r.ok) {
                        const err = await r.json();
                        throw new Error(err.detail || 'Contract validation failed');
                    }
                    res = await r.json();
                } else {
                    const r = await fetch('/api/pipeline', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ preset: presetSelect.value, top_k: 5 })
                    });
                    if (!r.ok) {
                        const err = await r.json();
                        throw new Error(err.detail || 'Pipeline error');
                    }
                    res = await r.json();
                }

                updateUI(res);
            } catch (err) {
                alert("Error: " + err.message);
            } finally {
                btn.disabled = false;
                btn.innerText = 'Run BrainVis Pipeline';
            }
        }

        function updateUI(data) {
            const analysis = data.analysis || {};
            const stimulus = analysis.predicted_stimulus || {};
            const topList = analysis.top_predictions || [];
            const recon = data.reconstruction || {};

            document.getElementById('latentNorm').innerText = analysis.clip_latent_norm || '-';
            document.getElementById('execSource').innerText = data.source || 'gateway';
            document.getElementById('ckptBadge').innerText = analysis.is_checkpoint_loaded ? 'Trained' : 'Baseline';
            document.getElementById('ckptBadge').className = analysis.is_checkpoint_loaded ? 'text-green-400 font-semibold' : 'text-yellow-400 font-semibold';

            document.getElementById('topLabel').innerText = stimulus.label || 'Unknown';
            document.getElementById('topMeta').innerText = (stimulus.category || '') + ' • Confidence: ' + (stimulus.confidence_percent || 0) + '%';

            const list = document.getElementById('topList');
            list.innerHTML = '';
            topList.forEach(item => {
                const row = document.createElement('div');
                row.className = 'flex justify-between items-center bg-gray-800 p-1.5 rounded';
                row.innerHTML = `<span class="capitalize text-gray-300">${item.label}</span><span class="font-mono text-indigo-400 font-bold">${item.confidence_percent}%</span>`;
                list.appendChild(row);
            });

            document.getElementById('engineBadge').innerText = recon.engine || '-';

            if (recon.reconstructed_image_base64) {
                const img = document.getElementById('reconImg');
                img.src = recon.reconstructed_image_base64;
                img.classList.remove('hidden');
                document.getElementById('reconPlaceholder').classList.add('hidden');
            }

            if (recon.saliency_map_base64) {
                const sal = document.getElementById('saliencyImg');
                sal.src = recon.saliency_map_base64;
                sal.classList.remove('hidden');
                document.getElementById('saliencyPlaceholder').classList.add('hidden');
            }

            if (recon.note) {
                document.getElementById('reconWarning').innerText = recon.note;
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
