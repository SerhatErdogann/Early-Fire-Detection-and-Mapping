import laspy
import numpy as np

laz_path = "#14) RGB Pointcloud.laz"

print("LAZ dosyasi okunuyor...")
las = laspy.read(laz_path)

print(f"Toplam nokta: {las.header.point_count:,}")

# Numpy array'e cevir
x = np.array(las.x)  # Boylam
y = np.array(las.y)  # Enlem
z = np.array(las.z)  # Yukseklik

print("\n" + "=" * 60)
print(" FLAME 2 DATASET - YANGIN BOLGE KONUMU")
print("=" * 60)

lat_center = float(np.mean(y))
lon_center = float(np.mean(x))

print(f"\nMerkez Koordinat:")
print(f"  Enlem (Latitude):  {lat_center:.6f}")
print(f"  Boylam (Longitude): {lon_center:.6f}")
print(f"  Yukseklik: {float(np.mean(z)):.0f} metre")

print(f"\nSinir Kutusu:")
print(f"  Guneybati: {float(np.min(y)):.6f}, {float(np.min(x)):.6f}")
print(f"  Kuzeydogu: {float(np.max(y)):.6f}, {float(np.max(x)):.6f}")

print(f"\nGoogle Maps:")
print(f"  https://www.google.com/maps?q={lat_center},{lon_center}&z=17")

print(f"\nKonum: Kaibab Ulusal Ormani, Kuzey Arizona, ABD")
print(f"Olay: Kasim 2021 Kontrollu Yanma (Prescribed Burn)")
