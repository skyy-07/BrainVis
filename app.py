import io
import json
import numpy as np
import torch
from typing import List, Optional, Union
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from service.eeg_analyzer import EEGAnalyzer, BRAINVIS_CLASSES
from service.image_reconstructor import ImageReconstructor

app = FastAPI(
    title="BrainVis EEG Analysis & Image Reconstruction API",
    description="Lightweight production endpoint for EEG signal analysis and visual stimulus reconstruction based on the BrainVis framework (ICASSP 2025).",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = EEGAnalyzer()
reconstructor = ImageReconstructor()

# Pydantic Schemas
class EEGAnalyzeRequest(BaseModel):
    eeg_data: Optional[List[List[float]]] = Field(
        None,
        description="2D array of EEG wave values [channels, time_points] (e.g. 128x440) or [time_points]."
    )
    preset: Optional[str] = Field(
        None,
        description="Optional synthetic preset name: 'panda', 'mountain bike', 'pizza', 'cat', 'daisy', 'electric guitar'."
    )
    top_k: int = Field(5, description="Number of top visual classes to predict (default 5).")

class ReconstructRequest(BaseModel):
    label: str = Field(..., description="Target visual stimulus label (e.g. 'panda', 'pizza', 'cat').")
    category: str = Field("object", description="Category of stimulus (e.g. 'animal', 'vehicle', 'food').")
    confidence: float = Field(0.95, description="Prediction confidence score between 0 and 1.")
    prefer_cloud_diffusion: bool = Field(False, description="Whether to query cloud Stable Diffusion if HF_API_TOKEN is set.")

class PipelineRequest(BaseModel):
    eeg_data: Optional[List[List[float]]] = Field(
        None,
        description="2D array of EEG wave values [channels, time_points] or 1D list."
    )
    preset: Optional[str] = Field(
        None,
        description="Optional synthetic preset name ('panda', 'mountain bike', 'pizza', 'cat', 'daisy', 'electric guitar')."
    )
    top_k: int = Field(5, description="Number of top classes.")
    prefer_cloud_diffusion: bool = Field(False, description="Whether to invoke cloud diffusion API if token available.")

@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint for Render service uptime monitoring."""
    return {
        "status": "healthy",
        "service": "BrainVis EEG & Image Reconstruction API",
        "version": "1.0.0",
        "device": "cpu",
        "lightweight_mode": True
    }

@app.get("/api/classes", tags=["Classes"])
def get_supported_classes():
    """Returns the 40 ImageNet visual stimulus categories supported by BrainVis."""
    return {
        "total_classes": len(BRAINVIS_CLASSES),
        "classes": BRAINVIS_CLASSES
    }

@app.get("/api/sample-eeg", tags=["EEG"])
def get_sample_eeg(preset: str = Query("panda", description="Sample preset name")):
    """Generates and returns synthetic sample EEG wave data."""
    preset = preset.lower()
    signals = EEGAnalyzer.generate_synthetic_eeg(preset=preset, channels=8, time_points=200)
    return {
        "preset": preset,
        "channels": signals.shape[0],
        "time_samples": signals.shape[1],
        "sample_rate_hz": 250,
        "eeg_data": signals.tolist()
    }

@app.post("/api/analyze", tags=["EEG Analysis"])
def analyze_eeg(req: EEGAnalyzeRequest):
    """
    Analyze EEG wave data:
    - Power spectral density across Delta, Theta, Alpha, Beta, Gamma bands
    - Waveform statistics (mean, variance, RMS, peak-to-peak amplitude, entropy)
    - Brain state indices (Alpha/Theta ratio, cognitive workload, visual attention)
    - Neural stimulus prediction across BrainVis visual categories
    """
    if req.eeg_data is not None and len(req.eeg_data) > 0:
        data = np.array(req.eeg_data, dtype=np.float32)
    elif req.preset:
        data = EEGAnalyzer.generate_synthetic_eeg(preset=req.preset, channels=128, time_points=440)
    else:
        # Default to panda preset
        data = EEGAnalyzer.generate_synthetic_eeg(preset="panda", channels=128, time_points=440)

    result = analyzer.analyze(data, top_k=req.top_k)
    return result

@app.post("/api/reconstruct", tags=["Image Reconstruction"])
def reconstruct_image(req: ReconstructRequest):
    """
    Reconstruct visual stimulus image from decoded stimulus label/category.
    Returns base64 encoded reconstruction image and metadata.
    """
    res = reconstructor.reconstruct(
        label=req.label,
        category=req.category,
        confidence=req.confidence,
        prefer_cloud_diffusion=req.prefer_cloud_diffusion
    )
    return res

@app.post("/api/pipeline", tags=["Pipeline"])
def full_eeg_to_image_pipeline(req: PipelineRequest):
    """
    Unified end-to-end endpoint:
    Input: Multi-channel EEG waves.
    Output: Comprehensive wave analysis + Decoded Visual Class + Reconstructed Image & Cortical Saliency Map.
    """
    if req.eeg_data is not None and len(req.eeg_data) > 0:
        eeg_np = np.array(req.eeg_data, dtype=np.float32)
    elif req.preset:
        eeg_np = EEGAnalyzer.generate_synthetic_eeg(preset=req.preset, channels=128, time_points=440)
    else:
        eeg_np = EEGAnalyzer.generate_synthetic_eeg(preset="panda", channels=128, time_points=440)

    # 1. Analyze EEG wave
    analysis = analyzer.analyze(eeg_np, top_k=req.top_k)
    top_pred = analysis["predicted_stimulus"] or {
        "label": "visual object",
        "category": "object",
        "confidence_percent": 85.0
    }

    # 2. Reconstruct image
    recon_res = reconstructor.reconstruct(
        label=top_pred["label"],
        category=top_pred["category"],
        confidence=top_pred.get("probability", 0.9),
        eeg_data=eeg_np,
        band_powers=analysis["spectral_analysis"],
        prefer_cloud_diffusion=req.prefer_cloud_diffusion
    )

    return {
        "analysis": analysis,
        "reconstruction": recon_res
    }

@app.post("/api/upload-eeg", tags=["EEG File Upload"])
async def upload_eeg_file(file: UploadFile = File(...), top_k: int = 5):
    """
    Upload an EEG file (.npy, .csv, .json, or .pth) and run the full BrainVis pipeline.
    """
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
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .npy, .csv, .json, or .pth")

        if data.ndim == 1:
            data = data.reshape(1, -1)

        analysis = analyzer.analyze(data, top_k=top_k)
        top_pred = analysis["predicted_stimulus"]

        recon = reconstructor.reconstruct(
            label=top_pred["label"],
            category=top_pred["category"],
            confidence=top_pred.get("probability", 0.9),
            eeg_data=data,
            band_powers=analysis["spectral_analysis"]
        )

        return {
            "filename": file.filename,
            "shape": list(data.shape),
            "analysis": analysis,
            "reconstruction": recon
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process EEG file: {str(e)}")

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def web_dashboard():
    """Serves the interactive web UI for testing EEG wave analysis and visual reconstruction."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BrainVis: EEG Wave Analysis & Image Reconstruction</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css">
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
        .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
        .accent-blue { color: #58a6ff; }
        .accent-green { color: #3fb950; }
        .accent-purple { color: #bc8cff; }
        .badge { background: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 2px 8px; font-size: 0.8rem; }
        .progress-bar { background: #21262d; border-radius: 4px; overflow: hidden; height: 8px; }
        .progress-fill { height: 100%; transition: width 0.4s ease; }
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
        <!-- Header -->
        <header class="mb-8 border-b border-gray-800 pb-6 flex flex-col md:flex-row justify-between items-start md:items-center">
            <div>
                <h1 class="text-3xl font-extrabold text-white flex items-center space-x-3">
                    <span class="text-indigo-400">🧠</span>
                    <span>BrainVis Cloud Endpoint</span>
                </h1>
                <p class="text-gray-400 text-sm mt-1">
                    Multi-modal Brain-Visual Signal Decoding &amp; Perceptual Image Reconstruction
                </p>
            </div>
            <div class="mt-4 md:mt-0 flex space-x-2">
                <a href="/docs" target="_blank" class="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-md transition shadow">
                    API Docs (Swagger)
                </a>
                <a href="/health" target="_blank" class="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-200 text-xs font-semibold rounded-md transition">
                    Health Check
                </a>
            </div>
        </header>

        <!-- Controls -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="card p-5 md:col-span-1">
                <h2 class="text-lg font-bold text-white mb-3 flex items-center space-x-2">
                    <span>⚡ EEG Signal Source</span>
                </h2>
                <div class="space-y-4">
                    <div>
                        <label class="block text-xs font-semibold text-gray-400 mb-1">Preset Neural Waveform</label>
                        <select id="presetSelect" class="w-full bg-gray-900 border border-gray-700 rounded-md p-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500">
                            <option value="panda">Panda (Alpha + Beta sync)</option>
                            <option value="mountain bike">Mountain Bike (Visual attention)</option>
                            <option value="pizza">Pizza (Food stimulus)</option>
                            <option value="cat">Cat (Animal category)</option>
                            <option value="daisy">Daisy (Floral stimulus)</option>
                            <option value="electric guitar">Electric Guitar (Instrument)</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-xs font-semibold text-gray-400 mb-1">Or Upload EEG Data (.npy, .csv, .json)</label>
                        <input type="file" id="eegFileInput" class="w-full text-xs text-gray-400 bg-gray-900 border border-gray-700 rounded-md p-1.5">
                    </div>

                    <button id="runBtn" onclick="runAnalysis()" class="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-md transition flex justify-center items-center space-x-2">
                        <span id="btnSpinner" class="hidden animate-spin">⏳</span>
                        <span>Run BrainVis Pipeline</span>
                    </button>
                </div>
            </div>

            <!-- Waveform Preview Canvas -->
            <div class="card p-5 md:col-span-2 flex flex-col justify-between">
                <div>
                    <div class="flex justify-between items-center mb-2">
                        <h2 class="text-lg font-bold text-white">📈 EEG Waveform Trace</h2>
                        <span id="signalMeta" class="text-xs text-gray-400">128 Channels | 440 Samples | 250 Hz</span>
                    </div>
                    <canvas id="waveformCanvas" width="700" height="150" class="w-full bg-gray-900 rounded border border-gray-800"></canvas>
                </div>
                <div class="grid grid-cols-4 gap-2 mt-4 text-center">
                    <div class="bg-gray-800 p-2 rounded">
                        <span class="block text-xs text-gray-400">Mean Amplitude</span>
                        <span id="statMean" class="font-mono text-sm text-blue-400">-</span>
                    </div>
                    <div class="bg-gray-800 p-2 rounded">
                        <span class="block text-xs text-gray-400">Peak-to-Peak</span>
                        <span id="statPtp" class="font-mono text-sm text-green-400">-</span>
                    </div>
                    <div class="bg-gray-800 p-2 rounded">
                        <span class="block text-xs text-gray-400">RMS Value</span>
                        <span id="statRms" class="font-mono text-sm text-yellow-400">-</span>
                    </div>
                    <div class="bg-gray-800 p-2 rounded">
                        <span class="block text-xs text-gray-400">Spectral Entropy</span>
                        <span id="statEntropy" class="font-mono text-sm text-purple-400">-</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Results Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <!-- Frequency Bands -->
            <div class="card p-5">
                <h3 class="text-md font-bold text-white mb-4">📊 EEG Frequency Bands</h3>
                <div class="space-y-3" id="bandsContainer">
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span>Delta (0.5 - 4 Hz)</span>
                            <span id="bandDelta" class="font-mono text-gray-400">-</span>
                        </div>
                        <div class="progress-bar"><div id="barDelta" class="progress-fill bg-blue-500" style="width: 0%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span>Theta (4 - 8 Hz)</span>
                            <span id="bandTheta" class="font-mono text-gray-400">-</span>
                        </div>
                        <div class="progress-bar"><div id="barTheta" class="progress-fill bg-green-500" style="width: 0%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span>Alpha (8 - 13 Hz)</span>
                            <span id="bandAlpha" class="font-mono text-gray-400">-</span>
                        </div>
                        <div class="progress-bar"><div id="barAlpha" class="progress-fill bg-yellow-500" style="width: 0%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span>Beta (13 - 30 Hz)</span>
                            <span id="bandBeta" class="font-mono text-gray-400">-</span>
                        </div>
                        <div class="progress-bar"><div id="barBeta" class="progress-fill bg-red-500" style="width: 0%"></div></div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span>Gamma (30 - 50 Hz)</span>
                            <span id="bandGamma" class="font-mono text-gray-400">-</span>
                        </div>
                        <div class="progress-bar"><div id="barGamma" class="progress-fill bg-purple-500" style="width: 0%"></div></div>
                    </div>
                </div>

                <div class="mt-6 border-t border-gray-800 pt-4 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Alpha / Theta Ratio:</span>
                        <span id="idxAlphaTheta" class="font-mono text-white">-</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Visual Attention Index:</span>
                        <span id="idxVisualAttn" class="font-mono text-white">-</span>
                    </div>
                </div>
            </div>

            <!-- Decoded Stimulus & Top Predictions -->
            <div class="card p-5">
                <h3 class="text-md font-bold text-white mb-3">🎯 Decoded Visual Category</h3>
                <div class="bg-gray-800 p-3 rounded mb-4 text-center">
                    <span class="text-xs text-gray-400 uppercase tracking-wider block">Predicted Visual Stimulus</span>
                    <span id="topLabel" class="text-xl font-extrabold text-indigo-400 capitalize block mt-1">Awaiting Signal</span>
                    <span id="topMeta" class="text-xs text-gray-400 block mt-0.5">-</span>
                </div>

                <h4 class="text-xs font-semibold text-gray-400 mb-2 uppercase">Top-5 Probabilities</h4>
                <div id="topList" class="space-y-2 text-xs">
                    <div class="text-gray-500 italic">Run pipeline to see probabilities...</div>
                </div>
            </div>

            <!-- Reconstructed Image & Cortical Saliency -->
            <div class="card p-5 flex flex-col justify-between">
                <div>
                    <h3 class="text-md font-bold text-white mb-3">🖼️ Reconstructed Image</h3>
                    <div class="relative bg-gray-900 rounded border border-gray-800 aspect-square flex items-center justify-center overflow-hidden mb-3">
                        <img id="reconImage" src="" alt="Reconstructed Stimulus" class="w-full h-full object-cover hidden">
                        <div id="reconPlaceholder" class="text-center p-4">
                            <span class="text-3xl block mb-2">👁️</span>
                            <span class="text-xs text-gray-500">Image will appear upon pipeline execution</span>
                        </div>
                    </div>
                </div>

                <div>
                    <h4 class="text-xs font-semibold text-gray-400 mb-1">Cortical Saliency Map</h4>
                    <div class="h-16 bg-gray-900 rounded border border-gray-800 overflow-hidden flex items-center justify-center">
                        <img id="saliencyImage" src="" alt="Saliency Map" class="w-full h-full object-cover hidden">
                        <span id="saliencyPlaceholder" class="text-xs text-gray-500">Signal topography</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Draw synthetic EEG waveform trace on load
        function drawDummyWave() {
            const canvas = document.getElementById('waveformCanvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = '#58a6ff';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            const width = canvas.width;
            const height = canvas.height;
            const mid = height / 2;

            for (let x = 0; x < width; x++) {
                const t = x / 40;
                const y = mid + Math.sin(t * 1.5) * 25 + Math.sin(t * 3.7) * 15 + (Math.random() - 0.5) * 8;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
        drawDummyWave();

        async function runAnalysis() {
            const btn = document.getElementById('runBtn');
            const spinner = document.getElementById('btnSpinner');
            btn.disabled = true;
            spinner.classList.remove('hidden');

            const fileInput = document.getElementById('eegFileInput');
            const presetSelect = document.getElementById('presetSelect');

            try {
                let res;
                if (fileInput.files.length > 0) {
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    const response = await fetch('/api/upload-eeg', { method: 'POST', body: formData });
                    res = await response.json();
                } else {
                    const preset = presetSelect.value;
                    const response = await fetch('/api/pipeline', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ preset: preset, top_k: 5 })
                    });
                    res = await response.json();
                }

                updateDashboard(res);
            } catch (err) {
                alert("Error running BrainVis pipeline: " + err.message);
            } finally {
                btn.disabled = false;
                spinner.classList.add('hidden');
            }
        }

        function updateDashboard(data) {
            const analysis = data.analysis || {};
            const stats = analysis.statistics || {};
            const spectral = analysis.spectral_analysis || {};
            const relPowers = spectral.relative_powers_percent || {};
            const indices = spectral.indices || {};
            const stimulus = analysis.predicted_stimulus || {};
            const topList = analysis.top_predictions || [];
            const recon = data.reconstruction || {};

            // Waveform stats
            document.getElementById('statMean').innerText = stats.mean_amplitude_uv ?? '-';
            document.getElementById('statPtp').innerText = stats.peak_to_peak_uv ?? '-';
            document.getElementById('statRms').innerText = stats.rms_uv ?? '-';
            document.getElementById('statEntropy').innerText = stats.spectral_entropy ?? '-';

            // Bands
            const bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'];
            bands.forEach(b => {
                const val = relPowers[b.toLowerCase()] || 0;
                document.getElementById('band' + b).innerText = val + '%';
                document.getElementById('bar' + b).style.width = Math.min(100, val * 2.5) + '%';
            });

            document.getElementById('idxAlphaTheta').innerText = indices.alpha_theta_ratio ?? '-';
            document.getElementById('idxVisualAttn').innerText = indices.visual_attention_index ?? '-';

            // Top Stimulus
            document.getElementById('topLabel').innerText = stimulus.label || 'Unknown';
            document.getElementById('topMeta').innerText = (stimulus.category || '') + ' • Confidence: ' + (stimulus.confidence_percent || 0) + '%';

            // Top-5 List
            const listContainer = document.getElementById('topList');
            listContainer.innerHTML = '';
            topList.forEach(item => {
                const row = document.createElement('div');
                row.className = 'flex justify-between items-center bg-gray-800 p-1.5 rounded';
                row.innerHTML = `
                    <span class="capitalize text-gray-300 font-medium">${item.label}</span>
                    <span class="font-mono text-indigo-400 font-semibold">${item.confidence_percent}%</span>
                `;
                listContainer.appendChild(row);
            });

            // Images
            if (recon.reconstructed_image_base64) {
                const img = document.getElementById('reconImage');
                img.src = recon.reconstructed_image_base64;
                img.classList.remove('hidden');
                document.getElementById('reconPlaceholder').classList.add('hidden');
            }

            if (recon.saliency_map_base64) {
                const salImg = document.getElementById('saliencyImage');
                salImg.src = recon.saliency_map_base64;
                salImg.classList.remove('hidden');
                document.getElementById('saliencyPlaceholder').classList.add('hidden');
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
