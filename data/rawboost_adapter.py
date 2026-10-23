from __future__ import annotations
import importlib
import numpy as np

# Defaults widely used with the RawBoost anti-spoofing implementation. Override
# individual settings via the `rawboost` section of config.yaml.
DEFAULTS = {
    "nBands": 5, "minF": 20, "maxF": 8000, "minBW": 100,
    "maxBW": 1000, "minCoeff": 10, "maxCoeff": 100,
    "minG": 0, "maxG": 0, "minBiasLinNonLin": 5,
    "maxBiasLinNonLin": 20, "N_f": 5, "P": 10, "g_sd": 2,
    "SNRmin": 10, "SNRmax": 40,
}

def apply_rawboost(audio, config=None, sample_rate=16000):
    """Apply a selected composition of unchanged RawBoost functions.

    algo=0 disabled; 1 LnL; 2 ISD; 3 SSI; 4 LnL->ISD->SSI;
    5 LnL->ISD; 6 LnL->SSI; 7 ISD->SSI; 8 parallel LnL+ISD.
    Other algorithm IDs deliberately raise rather than silently changing data.
    """
    config = config or {}
    algo = int(config.get("algo", 0))
    if algo == 0:
        return np.asarray(audio, dtype=np.float32)
    if algo not in range(1, 9):
        raise ValueError(f"Unsupported rawboost.algo={algo}; supported: 0..8")
    try:
        raw = importlib.import_module("RawBoost")
    except ImportError as exc:
        raise ImportError(
            "RawBoost augmentation is enabled: place your original RawBoost.py "
            "beside main.py (it will not be modified)."
        ) from exc
    p = {**DEFAULTS, **{k: v for k, v in config.items() if k in DEFAULTS}}
    x = np.asarray(audio, dtype=np.float64).reshape(-1).copy()
    if not x.size:
        raise ValueError("Cannot augment an empty waveform")
    fs = int(sample_rate)
    if algo == 8:
        a = raw.LnL_convolutive_noise(x.copy(), p['N_f'], p['nBands'], p['minF'],
            min(p['maxF'], fs/2 - 1), p['minBW'], p['maxBW'], p['minCoeff'],
            p['maxCoeff'], p['minG'], p['maxG'], p['minBiasLinNonLin'],
            p['maxBiasLinNonLin'], fs)
        b = raw.ISD_additive_noise(x.copy(), p['P'], p['g_sd'])
        x = raw.normWav(a + b, 0)
    if algo in (1, 4, 5, 6):
        x = raw.LnL_convolutive_noise(x, p['N_f'], p['nBands'], p['minF'],
              min(p['maxF'], fs/2 - 1), p['minBW'], p['maxBW'], p['minCoeff'],
              p['maxCoeff'], p['minG'], p['maxG'], p['minBiasLinNonLin'],
              p['maxBiasLinNonLin'], fs)
    if algo in (2, 4, 5, 7):
        x = raw.ISD_additive_noise(x, p['P'], p['g_sd'])
    if algo in (3, 4, 6, 7):
        x = raw.SSI_additive_noise(x, p['SNRmin'], p['SNRmax'], p['nBands'],
              p['minF'], min(p['maxF'], fs/2 - 1), p['minBW'], p['maxBW'],
              p['minCoeff'], p['maxCoeff'], p['minG'], p['maxG'], fs)
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("RawBoost produced non-finite samples")
    return x
