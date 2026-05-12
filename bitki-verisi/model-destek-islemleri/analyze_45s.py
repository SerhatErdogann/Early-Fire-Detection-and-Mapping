import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'outputs', 'video_predictions_scored.csv'))

# Video 30fps ile ilk 45 saniye = 0-1350 frame
ilk45 = df[df['frame_idx'] <= 1350].copy()

prob_col = 'decision_prob' if 'decision_prob' in df.columns else 'prob_fire'

print("=" * 65)
print(" ILK 45 SANIYE ANALIZI (0-1350 frame arasi)")
print("=" * 65)
toplam = len(ilk45)
yangin_var = int(ilk45['pred_fire'].sum())
yangin_yok = toplam - yangin_var
print(f"Toplam okunan frame: {toplam}")
print(f"Yangin dedigi (pred_fire=1): {yangin_var}")
print(f"Yangin demedigi (pred_fire=0): {yangin_yok}")
print()
print("DETAYLI TABLO:")
print("-" * 65)
print(f"{'Frame':>8} {'Yangin?':>10} {'Yuzdesi':>10} {'Alarm':>12} {'Threshold':>10}")
print("-" * 65)
for _, r in ilk45.iterrows():
    fire = "YANGIN" if r['pred_fire'] == 1 else "-"
    alarm = str(r.get('alarm_state', ''))
    thr = r.get('threshold_used', 0.5)
    prob = r[prob_col] * 100
    print(f"{int(r['frame_idx']):>8} {fire:>10} {prob:>9.1f}% {alarm:>12} {thr:>10.3f}")
