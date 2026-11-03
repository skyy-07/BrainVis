import numpy as np
from scipy import signal, integrate
import torch
import torch.nn as nn
import torch.nn.functional as F

# The 40 ImageNet visual stimulus categories from BrainVis
BRAINVIS_CLASSES = {
    0: {"id": "n02389026", "name": "sorrel horse", "category": "animal"},
    1: {"id": "n03888257", "name": "parachute", "category": "object"},
    2: {"id": "n03584829", "name": "smoothing iron", "category": "appliance"},
    3: {"id": "n02607072", "name": "anemone fish", "category": "animal"},
    4: {"id": "n03297495", "name": "espresso maker", "category": "appliance"},
    5: {"id": "n03063599", "name": "coffee mug", "category": "object"},
    6: {"id": "n03792782", "name": "mountain bike, all-terrain bike", "category": "vehicle"},
    7: {"id": "n04086273", "name": "revolver, six-shooter", "category": "object"},
    8: {"id": "n02510455", "name": "panda", "category": "animal"},
    9: {"id": "n11939491", "name": "daisy", "category": "plant"},
    10: {"id": "n02951358", "name": "canoe or kayak", "category": "vehicle"},
    11: {"id": "n02281787", "name": "lycaenid butterfly", "category": "animal"},
    12: {"id": "n02106662", "name": "german shepherd dog", "category": "animal"},
    13: {"id": "n04120489", "name": "running shoe", "category": "object"},
    14: {"id": "n03590841", "name": "Orange jack-o’-lantern", "category": "object"},
    15: {"id": "n02992529", "name": "cellular telephone", "category": "electronic"},
    16: {"id": "n03445777", "name": "golf ball", "category": "object"},
    17: {"id": "n03180011", "name": "desktop computer", "category": "electronic"},
    18: {"id": "n02906734", "name": "broom", "category": "object"},
    19: {"id": "n07873807", "name": "pizza", "category": "food"},
    20: {"id": "n03773504", "name": "missile", "category": "vehicle"},
    21: {"id": "n02492035", "name": "Cebus capucinus", "category": "animal"},
    22: {"id": "n03982430", "name": "pool table, billiard table", "category": "object"},
    23: {"id": "n03709823", "name": "mailbag", "category": "object"},
    24: {"id": "n03100240", "name": "old convertible", "category": "vehicle"},
    25: {"id": "n03376595", "name": "folding chair", "category": "furniture"},
    26: {"id": "n03877472", "name": "pajama", "category": "clothing"},
    27: {"id": "n03775071", "name": "mitten, glove", "category": "clothing"},
    28: {"id": "n03272010", "name": "electric guitar", "category": "instrument"},
    29: {"id": "n04069434", "name": "reflex camera", "category": "electronic"},
    30: {"id": "n03452741", "name": "grand piano", "category": "instrument"},
    31: {"id": "n03792972", "name": "mountain tent", "category": "object"},
    32: {"id": "n07753592", "name": "banana", "category": "food"},
    33: {"id": "n13054560", "name": "bolete", "category": "plant"},
    34: {"id": "n03197337", "name": "digital watch", "category": "electronic"},
    35: {"id": "n02504458", "name": "African elephant", "category": "animal"},
    36: {"id": "n02690373", "name": "airliner", "category": "vehicle"},
    37: {"id": "n03272562", "name": "electric locomotive", "category": "vehicle"},
    38: {"id": "n04044716", "name": "radio telescope", "category": "electronic"},
    39: {"id": "n02124075", "name": "cat", "category": "animal"}
}

# Standard EEG frequency bands (Hz)
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0)
}

class CPUFreqEncoder(nn.Module):
    """
    CPU-safe implementation of BrainVis FreqEncoder
    Transforms EEG frequency domain features using FFT + LSTM.
    """
    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
        super().__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size, 40)

    def forward(self, x):
        # x expected shape: [batch_size, time_points, channels] or [batch_size, channels, time_points]
        batch_size = x.size(0)
        if x.size(1) != self.input_size and x.size(2) == self.input_size:
            # [batch, time, channels] -> [batch, channels, time]
            x = x.permute(0, 2, 1)

        channels = x.size(1)
        time_points = x.size(2)

        # FFT amplitude spectrum along time axis
        x_np = x.detach().cpu().numpy()
        fourier_transform = np.fft.fft(x_np, axis=2)
        half_len = time_points // 2
        half_spectrum = np.abs(fourier_transform[:, :, 1:half_len + 1])

        # Permute to [batch, freq_bins, channels]
        amp_tensor = torch.tensor(half_spectrum, dtype=torch.float32).permute(0, 2, 1)

        # Ensure freq_bins is at least 1
        if amp_tensor.size(1) == 0:
            amp_tensor = torch.zeros((batch_size, 10, channels), dtype=torch.float32)

        # Adapt channels to match LSTM input_size (128)
        cur_channels = amp_tensor.size(-1)
        if cur_channels < self.input_size:
            pad_amount = self.input_size - cur_channels
            amp_tensor = F.pad(amp_tensor, (0, pad_amount), mode='constant', value=0.0)
        elif cur_channels > self.input_size:
            amp_tensor = amp_tensor[:, :, :self.input_size]

        lstm_out, _ = self.lstm(amp_tensor)
        reps = lstm_out[:, -1, :]
        features = F.relu(self.output(reps))
        logits = self.classifier(features)
        return logits, features

class EEGAnalyzer:
    """
    Comprehensive EEG Wave Analysis engine for BrainVis.
    Supports signal decomposition, frequency band powers, waveform statistics,
    and visual stimulus classification.
    """
    def __init__(self, sample_rate=1000):
        self.sample_rate = sample_rate
        self.freq_encoder = CPUFreqEncoder()
        self.freq_encoder.eval()

    def normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalize array to [0, 1] range."""
        min_v = np.min(data)
        max_v = np.max(data)
        if max_v - min_v > 1e-8:
            return (data - min_v) / (max_v - min_v)
        return data - min_v

    def compute_band_powers(self, eeg_data: np.ndarray, fs: float = 250.0) -> dict:
        """
        Compute absolute and relative band power using Welch's method.
        eeg_data: shape (channels, time) or (time,)
        """
        if eeg_data.ndim == 1:
            eeg_data = eeg_data[np.newaxis, :]

        n_channels, n_times = eeg_data.shape
        nperseg = min(n_times, int(fs * 2)) if n_times >= 64 else n_times

        freqs, psd = signal.welch(eeg_data, fs=fs, nperseg=nperseg, axis=-1)
        mean_psd = np.mean(psd, axis=0) # Average across channels

        band_powers = {}
        total_power = 0.0

        for band, (low, high) in FREQ_BANDS.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx):
                power = float(integrate.trapezoid(mean_psd[idx], freqs[idx]))
            else:
                power = 0.0
            band_powers[band] = power
            total_power += power

        # Relative power (%)
        relative_powers = {}
        for band, power in band_powers.items():
            relative_powers[band] = round((power / total_power * 100.0) if total_power > 0 else 0.0, 2)

        # Brain indices
        alpha = band_powers.get("alpha", 1e-5)
        theta = band_powers.get("theta", 1e-5)
        beta = band_powers.get("beta", 1e-5)
        gamma = band_powers.get("gamma", 1e-5)

        indices = {
            "alpha_theta_ratio": round(alpha / (theta + 1e-6), 3),
            "cognitive_workload_index": round(beta / (alpha + theta + 1e-6), 3),
            "visual_attention_index": round((beta + gamma) / (alpha + theta + 1e-6), 3)
        }

        return {
            "absolute_powers": {k: float(f"{v:.4e}") for k, v in band_powers.items()},
            "relative_powers_percent": relative_powers,
            "indices": indices
        }

    def compute_wave_statistics(self, eeg_data: np.ndarray) -> dict:
        """Compute morphological statistical features of EEG wave."""
        mean_amp = float(np.mean(eeg_data))
        std_amp = float(np.std(eeg_data))
        variance = float(np.var(eeg_data))
        ptp = float(np.ptp(eeg_data)) # Peak-to-peak
        rms = float(np.sqrt(np.mean(eeg_data ** 2)))

        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(eeg_data - mean_amp) != 0, axis=-1))
        zcr = float(zero_crossings / (eeg_data.size + 1e-6))

        # Spectral entropy
        norm_fft = np.abs(np.fft.rfft(eeg_data, axis=-1))
        norm_fft = norm_fft / (np.sum(norm_fft, axis=-1, keepdims=True) + 1e-12)
        spectral_entropy = float(-np.sum(norm_fft * np.log2(norm_fft + 1e-12)) / np.log2(norm_fft.shape[-1]))

        return {
            "mean_amplitude_uv": round(mean_amp, 4),
            "std_deviation_uv": round(std_amp, 4),
            "variance": round(variance, 4),
            "peak_to_peak_uv": round(ptp, 4),
            "rms_uv": round(rms, 4),
            "zero_crossing_rate": round(zcr, 4),
            "spectral_entropy": round(spectral_entropy, 4)
        }

    def predict_stimulus_class(self, eeg_tensor: torch.Tensor, top_k: int = 5) -> list:
        """
        Classify visual stimulus from EEG signals across the 40 BrainVis classes.
        eeg_tensor: shape [batch, channels, time] or [channels, time]
        """
        if eeg_tensor.dim() == 2:
            eeg_tensor = eeg_tensor.unsqueeze(0)

        with torch.no_grad():
            logits, features = self.freq_encoder(eeg_tensor)
            probs = F.softmax(logits, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, min(top_k, 40))

        predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cls_info = BRAINVIS_CLASSES.get(idx, {"id": f"c_{idx}", "name": "unknown", "category": "general"})
            predictions.append({
                "class_index": idx,
                "wnid": cls_info["id"],
                "label": cls_info["name"],
                "category": cls_info["category"],
                "probability": round(prob, 4),
                "confidence_percent": round(prob * 100.0, 2)
            })

        return predictions

    def analyze(self, eeg_array: np.ndarray, top_k: int = 5) -> dict:
        """
        Full analysis pipeline:
        - Validates dimensions
        - Computes frequency band powers
        - Computes wave statistics
        - Predicts visual stimulus category
        """
        # Ensure 2D (channels, time)
        if eeg_array.ndim == 1:
            # Single channel input -> expand to (1, time)
            eeg_array = eeg_array.reshape(1, -1)
        elif eeg_array.ndim == 3:
            # Batch dimension -> take first sample
            eeg_array = eeg_array[0]

        n_channels, n_time = eeg_array.shape

        # Signal stats
        stats = self.compute_wave_statistics(eeg_array)

        # Spectral Band powers
        spectral = self.compute_band_powers(eeg_array)

        # Neural stimulus predictions
        eeg_torch = torch.tensor(eeg_array, dtype=torch.float32)
        top_predictions = self.predict_stimulus_class(eeg_torch, top_k=top_k)

        # Primary decoded stimulus
        top_prediction = top_predictions[0] if top_predictions else None

        return {
            "signal_shape": {"channels": n_channels, "time_samples": n_time},
            "statistics": stats,
            "spectral_analysis": spectral,
            "predicted_stimulus": top_prediction,
            "top_predictions": top_predictions
        }

    @staticmethod
    def generate_synthetic_eeg(preset: str = "panda", channels: int = 128, time_points: int = 440) -> np.ndarray:
        """
        Generate synthetic EEG wave data matching BrainVis experimental parameters
        (128 channels, 440 time samples, 250 Hz sampling).
        Preset determines dominant neural frequencies and signal profile.
        """
        t = np.linspace(0, time_points / 250.0, time_points)
        signals = np.zeros((channels, time_points))

        # Base 1/f neural pink noise
        noise = np.random.randn(channels, time_points)
        b, a = signal.butter(2, [0.5 / 125.0, 45.0 / 125.0], btype='band')
        filtered_noise = signal.filtfilt(b, a, noise, axis=-1)

        preset_freq_map = {
            "panda": (10.5, 22.0, 8),       # Alpha dominant (10.5Hz) + Beta (22Hz), class 8
            "mountain bike": (12.0, 26.0, 6), # High visual attention, class 6
            "pizza": (9.0, 18.0, 19),         # class 19
            "cat": (11.0, 20.0, 39),          # class 39
            "daisy": (8.5, 16.0, 9),          # class 9
            "electric guitar": (13.5, 28.0, 28) # class 28
        }

        f1, f2, class_idx = preset_freq_map.get(preset.lower(), (10.0, 20.0, 0))

        for ch in range(channels):
            phase_offset = (ch / channels) * np.pi
            # Visual evoked potential simulation (P100, N170, P300 components)
            vep = np.exp(-((t - 0.17) ** 2) / 0.005) * -15.0 + np.exp(-((t - 0.3) ** 2) / 0.01) * 20.0
            wave = (
                15.0 * np.sin(2 * np.pi * f1 * t + phase_offset) +
                8.0 * np.sin(2 * np.pi * f2 * t + 2 * phase_offset) +
                vep +
                3.0 * filtered_noise[ch]
            )
            signals[ch] = wave

        return signals
