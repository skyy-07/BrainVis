import os
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.BrainVisModels import AlignNet, TimeFreqEncoder, TimeEncoder, FreqEncoder

# 40 Visual Stimulus Categories from BrainVis (ICASSP 2025)
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

EXPECTED_CHANNELS = 128
EXPECTED_TIME_SAMPLES = 440

class BrainVisNativePipeline:
    """
    Authentic BrainVis model pipeline based on RomGai/BrainVis (ICASSP 2025).
    - Enforces strict (128, 440) EEG input contract.
    - Uses exact BrainVis architecture: TimeEncoder + FreqEncoder -> TimeFreqEncoder -> AlignNet.
    - Explicitly verifies and loads checkpoint ('clipfinetune_model.pkl').
    - Reports clear status: whether real weights are active or missing.
    - Aligns EEG features to (77, 768) CLIP latent space.
    """
    def __init__(self, device: str = None, checkpoint_path: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path or os.environ.get("BRAINVIS_CHECKPOINT_PATH")
        self.is_checkpoint_loaded = False
        self.checkpoint_source = None

        # Build official BrainVis arguments
        self.args = types.SimpleNamespace(
            d_model=1024,
            dropout=0.2,
            attn_heads=16,
            enable_res_parameter=1,
            layers=8,
            momentum=0.99,
            vocab_size=660,
            wave_length=4,
            mask_ratio=0.75,
            reg_layers=4,
            num_class=40,
            data_shape=(EXPECTED_TIME_SAMPLES, EXPECTED_CHANNELS),
            device=self.device
        )

        # Initialize real authors' model architecture
        self.time_model = TimeEncoder(self.args)
        self.freq_model = FreqEncoder()
        self.timefreq_model = TimeFreqEncoder(self.time_model, self.freq_model, self.args)
        self.align_net = AlignNet(
            input_size=1024,
            freq_size=128,
            output_size=77 * 768,
            pretrained_model=self.timefreq_model
        )

        # Move to device and set eval mode
        self.align_net = self.align_net.to(self.device)
        self.align_net.eval()

        # Attempt to load genuine checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Locates and loads the author-trained 'clipfinetune_model.pkl' checkpoint."""
        candidate_paths = [
            self.checkpoint_path,
            "exp/clipfinetune_model.pkl",
            "exp/epilepsy/test/clipfinetune_model.pkl",
            "exp/epilepsy/test/exp/epilepsy/test/clipfinetune_model.pkl",
            "pretrained_model/clipfinetune_model.pkl"
        ]

        found_path = None
        for p in candidate_paths:
            if p and os.path.isfile(p):
                found_path = p
                break

        if found_path:
            try:
                state_dict = torch.load(found_path, map_location=self.device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                self.align_net.load_state_dict(state_dict, strict=False)
                self.is_checkpoint_loaded = True
                self.checkpoint_source = found_path
                print(f"[BrainVis] Successfully loaded authentic checkpoint from: {found_path}")
            except Exception as e:
                print(f"[BrainVis] Failed loading checkpoint from {found_path}: {e}")
                self.is_checkpoint_loaded = False
        else:
            self.is_checkpoint_loaded = False
            self.checkpoint_source = None
            print("[BrainVis] WARNING: No checkpoint found! Model running with untrained weights.")

    def validate_input_contract(self, eeg_array: np.ndarray) -> np.ndarray:
        """
        Strictly enforces the BrainVis contract:
        - Must be exactly 128 channels in 10-20 layout.
        - Must be exactly 440 time samples (samples 20 to 460 at 250Hz).
        - Returns normalized shape (1, 440, 128) for the Transformer time encoder.
        """
        if eeg_array.ndim == 2:
            # Can be (128, 440) or (440, 128)
            if eeg_array.shape == (EXPECTED_CHANNELS, EXPECTED_TIME_SAMPLES):
                # Transpose to (440, 128)
                eeg_array = eeg_array.T
            elif eeg_array.shape == (EXPECTED_TIME_SAMPLES, EXPECTED_CHANNELS):
                pass
            else:
                raise ValueError(
                    f"Invalid EEG tensor shape {list(eeg_array.shape)}. "
                    f"BrainVis strictly requires exactly {EXPECTED_CHANNELS} channels "
                    f"and {EXPECTED_TIME_SAMPLES} time samples: shape ({EXPECTED_CHANNELS}, {EXPECTED_TIME_SAMPLES}) "
                    f"or ({EXPECTED_TIME_SAMPLES}, {EXPECTED_CHANNELS})."
                )
            eeg_array = np.expand_dims(eeg_array, axis=0) # (1, 440, 128)

        elif eeg_array.ndim == 3:
            # (batch, 128, 440) or (batch, 440, 128)
            b = eeg_array.shape[0]
            if eeg_array.shape[1:] == (EXPECTED_CHANNELS, EXPECTED_TIME_SAMPLES):
                eeg_array = np.transpose(eeg_array, (0, 2, 1))
            elif eeg_array.shape[1:] == (EXPECTED_TIME_SAMPLES, EXPECTED_CHANNELS):
                pass
            else:
                raise ValueError(
                    f"Invalid EEG batch tensor shape {list(eeg_array.shape)}. "
                    f"BrainVis strictly requires (batch, {EXPECTED_CHANNELS}, {EXPECTED_TIME_SAMPLES}) "
                    f"or (batch, {EXPECTED_TIME_SAMPLES}, {EXPECTED_CHANNELS})."
                )
        else:
            raise ValueError(
                f"Invalid EEG array dimensions: expected 2D or 3D tensor, got {eeg_array.ndim}D array."
            )

        return eeg_array.astype(np.float32)

    def forward(self, eeg_array: np.ndarray, top_k: int = 5) -> dict:
        """
        Executes the genuine BrainVis forward pipeline:
        1. Validates strict (128, 440) input contract.
        2. Feeds EEG into TimeFreqEncoder to obtain 40-class classification scores.
        3. Feeds into AlignNet to obtain the (77, 768) CLIP latent embedding.
        4. Returns top-k predictions, CLIP latent tensor, and checkpoint status.
        """
        validated_array = self.validate_input_contract(eeg_array)
        tensor_in = torch.tensor(validated_array, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # 1. Classification scores from TimeFreqEncoder
            _, _, scores = self.align_net.pretrained_model(tensor_in)
            probs = F.softmax(scores, dim=-1)[0]
            top_probs, top_indices = torch.topk(probs, min(top_k, 40))

            # 2. CLIP latent embedding from AlignNet
            clip_flat = self.align_net(tensor_in) # shape [1, 59136]
            clip_latent = clip_flat.reshape(-1, 77, 768) # shape [1, 77, 768]

        top_predictions = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            cls_info = BRAINVIS_CLASSES.get(idx, {"id": f"c_{idx}", "name": "unknown", "category": "general"})
            top_predictions.append({
                "class_index": idx,
                "wnid": cls_info["id"],
                "label": cls_info["name"],
                "category": cls_info["category"],
                "probability": round(prob, 4),
                "confidence_percent": round(prob * 100.0, 2)
            })

        primary_stimulus = top_predictions[0] if top_predictions else None

        result = {
            "is_checkpoint_loaded": self.is_checkpoint_loaded,
            "checkpoint_source": self.checkpoint_source,
            "device": self.device,
            "input_contract_verified": True,
            "input_shape": list(validated_array.shape),
            "clip_latent_shape": list(clip_latent.shape),
            "clip_latent_norm": round(float(torch.norm(clip_latent).cpu().item()), 4),
            "predicted_stimulus": primary_stimulus,
            "top_predictions": top_predictions
        }

        if not self.is_checkpoint_loaded:
            result["warning"] = (
                "AUTHENTIC_CHECKPOINT_MISSING: The BrainVis weights ('clipfinetune_model.pkl') "
                "were not found. Output classification is from an uninitialized baseline (~2.5% chance). "
                "To run true visual decoding, request checkpoints from the authors (hfu006@e.ntu.edu.sg) "
                "and mount to 'exp/clipfinetune_model.pkl'."
            )

        return result
