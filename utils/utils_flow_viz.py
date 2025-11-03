import os, cv2, numpy as np
from typing import Optional
from .utils_io import ensure_dir

def save_flow_outputs(flow: np.ndarray, out_prefix: str, mask_for_norm: Optional[np.ndarray],
                      write_png: bool=True) -> float:
    ensure_dir(os.path.dirname(out_prefix))
    np.save(out_prefix + "_flow_vec.npy", flow.astype(np.float32))
    mag = np.linalg.norm(flow, axis=2)

    vals = mag[mask_for_norm] if (mask_for_norm is not None) else mag.reshape(-1)
    p99 = np.percentile(vals, 99) if vals.size else 1.0
    den = max(p99, 1e-8)
    mag01 = np.clip(mag / den, 0, 1)

    if write_png:
        mag_vis = (mag01 * 255).astype(np.uint8)
        if mask_for_norm is not None: mag_vis[~mask_for_norm] = 0
        cv2.imwrite(out_prefix + "_flow_mag.png", mag_vis)

        ang = np.arctan2(flow[...,1], flow[...,0])
        ang = (ang + np.pi) / (2*np.pi)
        hsv = np.zeros((*mag.shape,3), np.uint8)
        hsv[...,0] = (ang * 179).astype(np.uint8)
        hsv[...,1] = 255
        hsv[...,2] = (mag01 * 255).astype(np.uint8)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if mask_for_norm is not None: color[~mask_for_norm] = 0
        cv2.imwrite(out_prefix + "_flow_color.png", color)

    return float(np.percentile(vals, 95)) if vals.size else 0.0
