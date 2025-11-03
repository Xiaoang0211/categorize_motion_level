import os, sys, argparse
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2, numpy as np
from typing import Optional, Tuple, List
from utils.utils_io import load_json, ensure_dir, fmt_template, list_images_dir, resolve_sequences, categorize_motion
from utils.utils_flow_viz import save_flow_outputs
from utils import optical_flow_backends as F

def compute_sequence_metric(frames_dir: str,
                            masks_dir: Optional[str],
                            out_dir: str,
                            exts: List[str],
                            flow_backend: str = "farneback",
                            flow_params: dict = None,
                            stride: int = 1,
                            single_mask: bool = False,
                            morph_kernel: int = 3,
                            inpaint_radius: int = 3) -> Tuple[int, float, float, float]:
    frames = list_images_dir(frames_dir, exts)
    if len(frames) < stride + 1:
        raise AssertionError(f"Not enough frames in {frames_dir}")

    have_masks = bool(masks_dir) and os.path.isdir(masks_dir)
    masks = list_images_dir(masks_dir, exts) if have_masks else []
    if have_masks and len(masks) < len(frames):
        raise AssertionError(f"Mask count mismatch in {masks_dir}")

    ensure_dir(out_dir)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    trend = []

    flow_fn = F.BACKENDS.get(flow_backend)
    if flow_fn is None:
        raise ValueError(f"Unknown flow backend: {flow_backend}")

    for i in range(len(frames) - stride):
        j = i + stride
        I0 = cv2.imread(frames[i], cv2.IMREAD_COLOR)
        I1 = cv2.imread(frames[j], cv2.IMREAD_COLOR)
        if I0 is None or I1 is None:
            print(f"[Warn] skip {i}->{j} (read failed)"); continue

        if have_masks:
            M0 = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
            M1 = cv2.imread(masks[j], cv2.IMREAD_GRAYSCALE)
            if M0 is None or M1 is None:
                print(f"[Warn] skip {i}->{j} (mask read failed)"); continue
            dyn = (M0 > 0).astype(np.uint8) if single_mask else ((M0 > 0) | (M1 > 0)).astype(np.uint8)
            dynamic = (dyn * 255).astype(np.uint8)
            dynamic = cv2.dilate(dynamic, ker, iterations=1)
            static = cv2.erode(cv2.bitwise_not(dynamic), ker, iterations=1)
            smask = static > 0
            I0_inp = cv2.inpaint(I0, dynamic, inpaint_radius, cv2.INPAINT_TELEA)
            I1_inp = cv2.inpaint(I1, dynamic, inpaint_radius, cv2.INPAINT_TELEA)
        else:
            smask = None
            I0_inp, I1_inp = I0, I1

        flow = flow_fn(I0_inp, I1_inp, **(flow_params or {}))
        base_i = os.path.splitext(os.path.basename(frames[i]))[0]
        base_j = os.path.splitext(os.path.basename(frames[j]))[0]
        out_prefix = os.path.join(out_dir, f"{base_i}_to_{base_j}")
        severity = save_flow_outputs(flow, out_prefix, mask_for_norm=smask, write_png=True)
        trend.append(severity)

    np.savetxt(os.path.join(out_dir, "trend_masked_p95.txt"), trend, fmt="%.6f")
    med, std = (float(np.median(trend)), float(np.std(trend))) if trend else (0.0, 0.0)
    score = med + 0.5 * std
    with open(os.path.join(out_dir, "global_metric.txt"), "w") as f:
        f.write(f"median={med}\nstd={std}\nmedian+0.5*std={score}\n")
    return len(trend), med, std, score


def main():
    ap = argparse.ArgumentParser(description="Categorize motion level by optical flow (config-driven).")
    ap.add_argument("--config", required=True, help="JSON config path")
    ap.add_argument("--root", type=str, help="Override cfg.root")
    ap.add_argument("--out", type=str, help="Override cfg.out")
    ap.add_argument("--split", type=str, help="Override cfg.split")
    args = ap.parse_args()

    cfg = load_json(args.config)
    root  = args.root  or cfg.get("root")
    out_root = args.out or cfg.get("out", "out_flow_all")
    split = (args.split or cfg.get("split", "val"))
    res   = cfg.get("res", "")
    exts  = cfg.get("image_exts", [".jpg",".jpeg",".png",".bmp"])

    stride        = int(cfg.get("stride", 1))
    single_mask   = bool(cfg.get("single_mask", False))
    morph_kernel  = int(cfg.get("morph_kernel", 3))
    inpaint_radius= int(cfg.get("inpaint_radius", 3))
    flow_backend  = cfg.get("flow_backend", "farneback")
    flow_params   = cfg.get("flow_params", {})

    frames_tpl = cfg["frames_dir_template"]
    masks_tpl  = cfg.get("masks_dir_template")
    split_out_root = os.path.join(out_root, str(split))
    ensure_dir(split_out_root)

    seqs = resolve_sequences({
        **cfg,
        "root": root,
        "res": res,
        "split": split
    })
    print(f"[info] split={split}, #seq={len(seqs)}, backend={flow_backend}")

    metrics_rows = []
    for seq in seqs:
        frames_dir = fmt_template(frames_tpl, root=root, res=res, split=split, seq=seq)
        masks_dir  = fmt_template(masks_tpl,  root=root, res=res, split=split, seq=seq) if masks_tpl else None
        if not os.path.isdir(frames_dir):
            raise AssertionError(f"Frames dir missing: {frames_dir}")
        if masks_dir and (not os.path.isdir(masks_dir)):
            print(f"[warn] masks dir missing -> no-mask: {masks_dir}")
            masks_dir = None

        seq_out_dir = os.path.join(split_out_root, seq)
        print(f"[proc] {seq} -> {seq_out_dir}")
        pairs, med, std, score = compute_sequence_metric(
            frames_dir, masks_dir, seq_out_dir, exts,
            flow_backend=flow_backend, flow_params=flow_params,
            stride=stride, single_mask=single_mask,
            morph_kernel=morph_kernel, inpaint_radius=inpaint_radius
        )
        metrics_rows.append((seq, pairs, med, std, score))

    # 汇总
    metrics_tsv = os.path.join(split_out_root, "split_metrics.tsv")
    with open(metrics_tsv, "w") as f:
        f.write("sequence\tpairs\tmedian\tstd\tscore\n")
        for seq, pairs, med, std, score in metrics_rows:
            f.write(f"{seq}\t{pairs}\t{med:.6f}\t{std:.6f}\t{score:.6f}\n")

    ranked = sorted(metrics_rows, key=lambda x: x[4])  # score升序
    ranking_txt = os.path.join(split_out_root, "split_ranking.txt")
    with open(ranking_txt, "w") as f:
        f.write("# rank\tsequence\tscore(median+0.5*std)\tcategory\n")
        for r, (seq, _, _, _, score) in enumerate(ranked, start=1):
            cat = categorize_motion(score)
            f.write(f"{r}\t{seq}\t{score:.6f}\t{cat}\n")

    print(f"[done] metrics -> {metrics_tsv}")
    print(f"[done] ranking -> {ranking_txt}")
    print(f"[done] outputs -> {split_out_root}")

if __name__ == "__main__":
    main()
