import numpy as np

def create_context_windows(reps: np.ndarray, window_size: int = 51, step: int = 5) -> np.ndarray:
    """Generate context windows for each frame.

    Args:
        reps: Array of shape (N, D) representing per-frame features at 50 Hz.
        window_size: Number of context steps (default 51 -> 5 s with 10 Hz sampling).
        step: Stride in frames between context steps (default 5 -> 100 ms steps).

    Returns:
        Array of shape (N, window_size, D) containing context around each frame.
    """
    N, D = reps.shape
    half = window_size // 2
    pad = half * step
    padded = np.pad(reps, ((pad, pad), (0, 0)), mode="edge")
    offsets = np.arange(-half, half + 1) * step
    idx = np.arange(N)[:, None] + offsets + pad
    windows = padded[idx]
    return windows.astype(reps.dtype)
