import os, glob, json
from typing import List, Optional

# ---------------------- basic I/O ----------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def fmt_template(tpl: str, **kw) -> str:
    return tpl.format(**kw)

def _fmt_from_cfg(s: Optional[str], cfg: dict) -> Optional[str]:
    if s is None:
        return None
    return s.format(**{
        "root":  cfg.get("root",""),
        "res":   cfg.get("res",""),
        "split": cfg.get("split",""),
        "seq":   cfg.get("seq",""),  # 不一定用到
    })

def read_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]

def list_subdirs(path: str) -> List[str]:
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

def list_images_dir(d: str, exts: List[str]) -> List[str]:
    ps=[]
    for e in exts:
        ps.extend(glob.glob(os.path.join(d, f"*{e}")))
    return sorted(ps)

# ---------------------- split utilities ----------------------
def resolve_sequences(cfg: dict) -> List[str]:
    """
    优先级：
      1) split_list_file （支持 {root}/{res}/{split} 占位符）
      2) scan_sequences_from （同样支持占位符；扫描子目录名）
    """
    split_list_file = _fmt_from_cfg(cfg.get("split_list_file"), cfg)
    if split_list_file:
        return read_lines(split_list_file)

    scan_base = _fmt_from_cfg(cfg.get("scan_sequences_from"), cfg)
    if scan_base:
        return list_subdirs(scan_base)

    raise RuntimeError("No sequences: provide 'split_list_file' or 'scan_sequences_from' in config.")

def write_split_list(base: str, out_txt: str):
    seqs = list_subdirs(base)
    with open(out_txt, "w") as f:
        for s in seqs:
            f.write(s + "\n")
    print(f"[done] {len(seqs)} sequences -> {out_txt}")

def write_split_list_from_config(cfg: dict, out_txt: str):
    scan_base = _fmt_from_cfg(cfg.get("scan_sequences_from"), cfg)
    if not scan_base:
        raise RuntimeError("write_split_list_from_config requires 'scan_sequences_from' in cfg.")
    write_split_list(scan_base, out_txt)
    
def categorize_motion(score):
    if score < 5: return "slow"
    elif score < 15: return "medium"
    else: return "fast"

# ---------------------- optional CLI ----------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Scan a base dir and write a split list file.")
    ap.add_argument("--base", type=str, help="Base dir to scan")
    ap.add_argument("--out", type=str, help="Output txt path")
    ap.add_argument("--config", type=str, help="Optional JSON config: use 'scan_sequences_from' with templates")
    args = ap.parse_args()

    if args.config:
        cfg = load_json(args.config)
        if not args.out:
            raise SystemExit("--out is required when using --config")
        write_split_list_from_config(cfg, args.out)
    else:
        if not args.base or not args.out:
            raise SystemExit("--base and --out are required")
        write_split_list(args.base, args.out)