import argparse
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Antalya Düzlerçamı Ormanı (Termessos bölgesi) yaklaşık koordinatları
START_LAT = 36.621777
START_LON = 30.517945



def get_video_duration_s(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Video suresi okunamadi: {video_path}")

    return float(frame_count / fps)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    parser = argparse.ArgumentParser(description="Antalya test rotasi icin video suresine gore GPS CSV uretir.")
    parser.add_argument(
        "--video",
        default=str(project_root / "data" / "videos" / "test_rgb.mp4"),
        help="Suresi okunacak video yolu",
    )
    parser.add_argument(
        "--out",
        default=str(script_dir / "Antalya_Orman_Ucusu.csv"),
        help="Uretilecek telemetry CSV yolu",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=10.0,
        help="Saniye basina GPS noktasi",
    )
    return parser.parse_args()


args = parse_args()
video_path = Path(args.video).resolve()
out_csv = Path(args.out).resolve()

TOTAL_SECONDS = get_video_duration_s(video_path)
HZ = float(args.hz)
total_points = max(2, int(round(TOTAL_SECONDS * HZ)))

# Zaman dizisi: videonun gercek suresine otomatik yayilir.
time_s = np.linspace(0, TOTAL_SECONDS, total_points)

# Gerçekçi bir uçuş rotası için (kıvrımlı bir yol)
# Drone yavaşça kuzeydoğuya doğru ilerlesin ve zigzag yapsın
latitudes = []
longitudes = []

current_lat = START_LAT
current_lon = START_LON

# Drone hızı (derece cinsinden çok küçük değerler)
# Yaklaşık 10m/s hıza denk gelmesi için
lat_speed = 0.00002
lon_speed = 0.00003

for t in time_s:
    # Biraz gürültü (rüzgar/titreme)
    noise_lat = np.random.normal(0, 0.000002)
    noise_lon = np.random.normal(0, 0.000002)
    
    # Zigzag hareketi için sinüs dalgası
    zigzag = np.sin(t * 0.2) * 0.00005
    
    current_lat += lat_speed + noise_lat
    current_lon += lon_speed + zigzag + noise_lon
    
    latitudes.append(current_lat)
    longitudes.append(current_lon)

# DataFrame oluştur
df = pd.DataFrame({
    'OSD.flyTime [s]': time_s,
    'CUSTOM.date [local]': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * total_points,
    'OSD.latitude': latitudes,
    'OSD.longitude': longitudes,
    'OSD.height [ft]': [150.0] * total_points,
    'OSD.altitude [ft]': [150.0] * total_points,
    'OSD.yaw [360]': [0.0] * total_points,
    'GIMBAL.pitch': [-90.0] * total_points,
    'GIMBAL.yaw [360]': [0.0] * total_points
})

os.makedirs(out_csv.parent, exist_ok=True)
df.to_csv(out_csv, index=False)

print(f"Basariyla olusturuldu: {out_csv}")
print(f"Video: {video_path}")
print(f"Video suresi: {TOTAL_SECONDS:.2f} saniye")
print(f"GPS nokta sayisi: {total_points}")
print(f"Baslangic: {START_LAT}, {START_LON}")
print(f"Bitis: {current_lat}, {current_lon}")
