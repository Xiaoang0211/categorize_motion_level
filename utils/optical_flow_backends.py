import cv2
import numpy as np

def _to_gray(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im

def farneback(I0: np.ndarray, I1: np.ndarray, **kw) -> np.ndarray:
    g0, g1 = _to_gray(I0), _to_gray(I1)
    params = dict(
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    params.update(kw or {})
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, **params)
    return flow.astype(np.float32)

def tvl1(I0: np.ndarray, I1: np.ndarray, **kw) -> np.ndarray:
    # OpenCV Dual TV-L1
    g0, g1 = _to_gray(I0), _to_gray(I1)
    r = cv2.optflow.DualTVL1OpticalFlow_create()
    # 常用参数（可在 config 覆盖）
    r.setTau(kw.get("tau", 0.25))
    r.setLambda(kw.get("lambda_", 0.15))
    r.setTheta(kw.get("theta", 0.3))
    r.setEpsilon(kw.get("epsilon", 0.01))
    r.setScalesNumber(kw.get("scales", 5))
    r.setWarpingsNumber(kw.get("warpings", 5))
    r.setInnerIterations(kw.get("inner_iter", 30))
    r.setOuterIterations(kw.get("outer_iter", 10))
    flow = r.calc(g0, g1, None)
    return flow.astype(np.float32)

BACKENDS = {
    "farneback": farneback,
    "tvl1": tvl1,
}
