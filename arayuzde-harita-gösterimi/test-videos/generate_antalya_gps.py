import pandas as pd
import numpy as np
import os
from datetime import datetime

# Antalya Düzlerçamı Ormanı (Termessos bölgesi) yaklaşık koordinatları
START_LAT = 36.621777
START_LON = 30.517945



# Toplam saniye (Video yaklaşık 65 saniye, biz 100 saniye yapalım garanti olsun)
TOTAL_SECONDS = 100
# Saniyede kaç GPS verisi olacak (DJI loglarında genelde 10 Hz falandır, 1 Hz yapalım yeterli)
HZ = 10
total_points = TOTAL_SECONDS * HZ

# Zaman dizisi
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
    'OSD.longitude': longitudes
})

# Çıktı dizini
out_dir = r"c:\Users\SERHAT\Desktop\Projeler\bitirme_proje\bitki-verisi\test-videos"
out_csv = os.path.join(out_dir, "Antalya_Orman_Ucusu.csv")

# DJI CSV formatı gereği ilk satırda boş/başka bilgiler olabiliyor ama pandas genelde direkt 1. satırdan okur.
# process_video_to_db.py skiprows=1 yapıyor.
# Biz de ilk satıra dummy bir başlık ekleyelim, 2. satıra gerçek başlıkları koyalım.

with open(out_csv, 'w') as f:
    f.write("DUMMY_HEADER_FOR_SKIPROWS\n")
    
df.to_csv(out_csv, index=False, mode='a')

print(f"Başarıyla oluşturuldu: {out_csv}")
print(f"Başlangıç: {START_LAT}, {START_LON}")
print(f"Bitiş: {current_lat}, {current_lon}")
