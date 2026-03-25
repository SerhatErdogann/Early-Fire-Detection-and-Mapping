import pandas as pd

INP = "outputs/video_predictions_scored.csv"
OUT = "outputs/top_frames.csv"

K = 50  # en riskli 50 frame

df = pd.read_csv(INP)

top = df.sort_values("risk_score", ascending=False).head(K).copy()
top.to_csv(OUT, index=False)

print("✅ yazıldı:", OUT)
print(top[["frame_idx", "prob_fire", "intensity_top10", "area_heat_gt_0_6", "risk_score", "heatmap_path"]].head(20))