"""
Orman Yangını Tespit ve Konum Takip Sistemi
Video'dan her saniye için yangın tespiti yapar ve GPS koordinatlarını takip eder.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from datetime import datetime
import math
import os


class FireDetectionWithCoordinates:
    def __init__(self, video_path, model_path, start_lat, start_lon, altitude_meters=150):
        """
        Yangın tespit ve koordinat takip sistemi
        
        Args:
            video_path: Video dosya yolu
            model_path: Yangın tespit modeli yolu (.h5)
            start_lat: Başlangıç enlem (latitude)
            start_lon: Başlangıç boylam (longitude)
            altitude_meters: Drone yüksekliği (metre)
        """
        self.video_path = video_path
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.altitude = altitude_meters
        
        # Video açma
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        # Video özellikleri
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f"\n{'='*80}")
        print("🔥 ORMAN YANGINI TESPİT VE KONUMLAMA SİSTEMİ")
        print(f"{'='*80}")
        print(f"\n📹 Video Bilgileri:")
        print(f"   Dosya: {video_path}")
        print(f"   Çözünürlük: {self.width}x{self.height}")
        print(f"   FPS: {self.fps}")
        print(f"   Süre: {self.duration:.2f} saniye")
        
        # Modeli yükle
        print(f"\n🤖 Model yükleniyor: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("   ✅ Model yüklendi!")
        
        # Koordinat sistemi parametreleri
        self.current_lat = start_lat
        self.current_lon = start_lon
        
        # Yerdeki görüntü alanı hesaplama
        self.fov_horizontal = 84  # derece
        self.fov_vertical = 53    # derece
        
        self.ground_width = 2 * self.altitude * math.tan(math.radians(self.fov_horizontal / 2))
        self.ground_height = 2 * self.altitude * math.tan(math.radians(self.fov_vertical / 2))
        
        self.meters_per_pixel_x = self.ground_width / self.width
        self.meters_per_pixel_y = self.ground_height / self.height
        
        print(f"\n📍 Koordinat Sistemi:")
        print(f"   Başlangıç: {self.start_lat}, {self.start_lon}")
        print(f"   Yükseklik: {self.altitude}m")
        print(f"   Görüş alanı: {self.ground_width:.2f}m x {self.ground_height:.2f}m")
        
        # Optical flow parametreleri
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # İlk frame'i oku
        ret, self.old_frame = self.cap.read()
        if not ret:
            raise ValueError("İlk frame okunamadı!")
        
        self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        
        # Sonuçları sakla
        self.results = []
        self.frame_count = 0
        self.total_displacement_x = 0
        self.total_displacement_y = 0
        
        print(f"\n{'='*80}")
        print("⏳ İşlem başlıyor...")
        print(f"{'='*80}\n")

    def meters_to_lat_lon(self, dx_meters, dy_meters):
        """Metre cinsinden yer değiştirmeyi GPS koordinatlarına çevir"""
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(self.current_lat))
        
        delta_lat = dy_meters / meters_per_degree_lat
        delta_lon = dx_meters / meters_per_degree_lon
        
        return delta_lat, delta_lon

    def calculate_movement(self, p0, p1):
        """Optical flow'dan hareket vektörünü hesapla"""
        if p0 is None or p1 is None or len(p0) == 0 or len(p1) == 0:
            return 0, 0
        
        movements = p1 - p0
        if movements.ndim == 3:
            movements = movements.reshape(-1, 2)
        
        median_x = np.median(movements[:, 0])
        median_y = np.median(movements[:, 1])
        
        std_x = np.std(movements[:, 0])
        std_y = np.std(movements[:, 1])
        
        threshold = 2.0
        mask = (np.abs(movements[:, 0] - median_x) < threshold * std_x) & \
               (np.abs(movements[:, 1] - median_y) < threshold * std_y)
        
        if np.sum(mask) == 0:
            return 0, 0
        
        filtered_movements = movements[mask]
        avg_dx = np.mean(filtered_movements[:, 0])
        avg_dy = np.mean(filtered_movements[:, 1])
        
        return avg_dx, avg_dy

    def detect_fire(self, frame):
        """
        Frame'de yangın tespiti yap
        
        Returns:
            (result, confidence): ('FIRE' veya 'NO FIRE', güven yüzdesi)
        """
        try:
            # Görüntüyü model için hazırla
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Tahmin yap
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # Model: fire=0, nofire=1 (inverse)
            if prediction > 0.5:
                result = "NO FIRE"
                confidence = prediction * 100
            else:
                result = "FIRE"
                confidence = (1 - prediction) * 100
            
            return result, confidence
        except Exception as e:
            print(f"   ⚠️ Yangın tespit hatası: {e}")
            return "UNKNOWN", 0

    def process_video(self, output_file="fire_detection_results.txt"):
        """
        Videoyu işle ve sonuçları kaydet
        
        Args:
            output_file: Sonuçların kaydedileceği dosya
        """
        print(f"\n📊 İşleniyor: Her saniye için yangın tespiti yapılıyor...")
        print(f"   (Toplam ~{int(self.duration)} saniye işlenecek)\n")
        
        # Her saniye bir frame işle
        frames_to_skip = int(self.fps)
        
        # Sonuç dosyası
        result_file = open(output_file, "w", encoding="utf-8")
        result_file.write("Saniye,Frame,Latitude,Longitude,Yangin_Durumu,Güven (%)\n")
        
        fire_count = 0
        nofire_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Her saniye bir frame
                if self.frame_count % frames_to_skip != 0 and self.frame_count != 1:
                    continue
                
                # Video zamanı (saniye)
                video_time = self.frame_count / self.fps
                
                # Yangın tespiti
                result, confidence = self.detect_fire(frame)
                
                # Koordinat güncelleme (optical flow)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if self.p0 is not None and len(self.p0) > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        self.old_gray, frame_gray, self.p0, None, **self.lk_params
                    )
                    
                    if p1 is not None and st is not None:
                        good_new = p1[st == 1]
                        good_old = self.p0[st == 1]
                        
                        if len(good_new) > 5:
                            avg_dx_pixel, avg_dy_pixel = self.calculate_movement(good_old, good_new)
                            
                            # Hareket yönünü ters çevir (drone hareket ediyor, görüntü ters hareket ediyor)
                            dx_meters = -avg_dx_pixel * self.meters_per_pixel_x
                            dy_meters = -avg_dy_pixel * self.meters_per_pixel_y
                            
                            self.total_displacement_x += dx_meters
                            self.total_displacement_y += dy_meters
                            
                            # GPS koordinatlarını güncelle
                            delta_lat, delta_lon = self.meters_to_lat_lon(dx_meters, dy_meters)
                            self.current_lat += delta_lat
                            self.current_lon += delta_lon
                
                # Yeni özellik noktaları bul
                self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
                self.old_gray = frame_gray.copy()
                
                # Sonuçları kaydet
                self.results.append({
                    'time': video_time,
                    'frame': self.frame_count,
                    'lat': self.current_lat,
                    'lon': self.current_lon,
                    'result': result,
                    'confidence': confidence
                })
                
                # İstatistik
                if result == "FIRE":
                    fire_count += 1
                    emoji = "🔥"
                else:
                    nofire_count += 1
                    emoji = "✅"
                
                # Konsol çıktısı
                print(f"   [{emoji}] {video_time:6.2f}s | Frame: {self.frame_count:5d} | "
                      f"GPS: {self.current_lat:.8f}, {self.current_lon:.8f} | "
                      f"{result:8s} (%{confidence:5.2f})")
                
                # Dosyaya yaz
                result_file.write(f"{video_time:.2f},{self.frame_count},"
                                f"{self.current_lat:.10f},{self.current_lon:.10f},"
                                f"{result},{confidence:.2f}\n")
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Kullanıcı tarafından durduruldu!")
        
        finally:
            result_file.close()
            self.cap.release()
            
            # Özet rapor
            self.print_summary(fire_count, nofire_count, output_file)

    def print_summary(self, fire_count, nofire_count, output_file):
        """İşlem özetini yazdır"""
        total = fire_count + nofire_count
        
        print(f"\n{'='*80}")
        print("📋 İŞLEM TAMAMLANDI")
        print(f"{'='*80}")
        print(f"\n📊 İstatistikler:")
        print(f"   Toplam işlenen saniye: {total}")
        print(f"   🔥 Yangın tespit edilen: {fire_count} saniye")
        print(f"   ✅ Yangın yok: {nofire_count} saniye")
        
        if fire_count > 0:
            print(f"\n⚠️  DİKKAT: {fire_count} farklı konumda YANGIN TESPİT EDİLDİ!")
            print(f"   Acil müdahale gerekebilir!")
        
        print(f"\n💾 Sonuç dosyası: {output_file}")
        print(f"📍 Başlangıç koordinatı: {self.start_lat:.10f}, {self.start_lon:.10f}")
        print(f"📍 Bitiş koordinatı:    {self.current_lat:.10f}, {self.current_lon:.10f}")
        print(f"📏 Toplam yer değiştirme: {self.total_displacement_x:.2f}m (X), {self.total_displacement_y:.2f}m (Y)")
        print(f"\n{'='*80}\n")


def main():
    """Ana program"""
    # Yapılandırma
    VIDEO_PATH = "output_20s.mp4"  # videodan_konum_alma klasöründeki video
    MODEL_PATH = "forest_fire_model_withrgb.h5"  # Ana klasördeki model
    
    # Başlangıç koordinatları (videodan_konum_alma/coordinate_log.txt'den alındı)
    START_LAT = 39.971122237497404
    START_LON = 32.81855850529272
    ALTITUDE = 150  # metre
    
    # Çalışma dizimi
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Tam yollar
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, VIDEO_PATH)
    model_path = os.path.join(script_dir, MODEL_PATH)
    
    # Alternatif model yolları dene
    possible_model_paths = [
        model_path,
        os.path.join(script_dir, "forest_fire_model_withrgb.h5"),
        os.path.join(os.path.dirname(script_dir), "forest_fire_model_withrgb.h5"),
    ]
    
    model_found = False
    for mp in possible_model_paths:
        if os.path.exists(mp):
            model_path = mp
            model_found = True
            break
    
    if not model_found:
        print(f"❌ Model dosyası bulunamadı!")
        print(f"   Aranan yollar:")
        for mp in possible_model_paths:
            print(f"   - {mp}")
        return
    
    print(f"\n📁 Çalışma dizini: {script_dir}")
    print(f"📹 Video: {video_path}")
    print(f"🤖 Model: {model_path}")
    
    try:
        # Sistemi başlat
        system = FireDetectionWithCoordinates(
            video_path=video_path,
            model_path=model_path,
            start_lat=START_LAT,
            start_lon=START_LON,
            altitude_meters=ALTITUDE
        )
        
        # Videoyu işle
        system.process_video(output_file="fire_detection_results.txt")
        
    except FileNotFoundError as e:
        print(f"\n❌ Hata: {e}")
        print("\nÇözüm:")
        print("1. Video dosyasının doğru klasörde olduğundan emin olun")
        print("2. Model dosyasının mevcut olduğunu kontrol edin")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
