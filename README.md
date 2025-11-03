# Categorize Motion Level

# Categorize Motion Level

A toolkit to estimate and rank camera motion intensity in image/video sequences using optical flow.  
Dynamic foreground regions are masked out using provided segmentation masks before flow computation to ensure that only camera motion is evaluated.  
We use the conda environment from [TTT3R](https://github.com/Inception3D/TTT3R).



---

## Usage

```bash
# Activate environment
conda activate ttt3r   # verified working with the conda env "ttt3r"
cd /home/xiaoang/FF_Reconstruction/categorize_motion_level

# Run for DAVIS train / val splits
python scripts/run_split.py --config configs/davis_train.json
python scripts/run_split.py --config configs/davis_val.json
```

Each run produces:

```bash
output_flow/<split>/<sequence>/
  *_flow_mag.png
  *_flow_color.png
  trend_masked_p95.txt
  global_metric.txt

output_flow/<split>/split_metrics.tsv
output_flow/<split>/split_ranking.txt
```

## Motion Categories
| Category | Score Range | Description |
|-----------|--------------|-------------|
| slow      | < 5          | minimal camera motion |
| medium    | 5–15         | moderate motion |
| fast      | ≥ 15         | strong or fast motion |

## .gitignore Suggestion
```
__pycache__/
output_*/
out_*/
outputs_*/
```