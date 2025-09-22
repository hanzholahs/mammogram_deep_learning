import numpy as np
from numpy.typing import NDArray


def to_uint8(arr: NDArray) -> NDArray[np.uint8]:
    """Min-max rescale any dtype/range to 0-255 uint8."""
    arr = np.asarray(arr, dtype=np.float32)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    # to prevent division by zero
    if vmax == vmin:
        return np.zeros_like(arr, dtype=np.uint8)

    # scale the array
    scaled = 255 * (arr - vmin) / (vmax - vmin)
    return scaled.astype(np.uint8)
