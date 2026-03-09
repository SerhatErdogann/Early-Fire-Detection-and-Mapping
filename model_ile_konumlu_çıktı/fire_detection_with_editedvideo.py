import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import math
import os

class FireDetectionWithGPSVideo:
    def __init__(self, video_path, model_path, csv_path):

        self.video_path = video_path
        self.csv_path = csv_path
        
        # Video açma
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}")
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}")
        
        # Video özellikleri
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or math.isnan(self.fps):
            self.fps = 30.0
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
        print(f"\n{'='*80}")
        print("ORMAN YANGINI TESPİT VE CSV KONUMLAMA SİSTEMİ")
        print(f"{'='*80}")
        print(f"\n Video Bilgileri:")
        print(f"   Dosya: {video_path}")
        print(f"   Çözünürlük: {self.width}x{self.height}")
        print(f"   FPS: {self.fps}")
        print(f"   Süre: {self.duration:.2f} saniye")
        
        # Modeli yükle
        print(f"\n Model yükleniyor: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        print("   Model yüklendi!")
        
        # CSV Oku
        print(f"\n GPS verileri yükleniyor: {csv_path}")
        try:
            # 'sep=,' ilk satırda olabiliyor, bu nedenle ondan kaçınmak için ya 2. satırdan başlayacağız ya da sep ayarı yapacağız.
            # İlk satırdaki 'sep=,' kısmını atla (skiprows=1)
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if 'sep=' in first_line.lower():
                    self.gps_data = pd.read_csv(csv_path, skiprows=1, low_memory=False)
                else:
                    self.gps_data = pd.read_csv(csv_path, low_memory=False)
                    
            if 'OSD.flyTime [s]' in self.gps_data.columns:
                self.time_col = 'OSD.flyTime [s]'
                self.lat_col = 'OSD.latitude'
                self.lon_col = 'OSD.longitude'
            else:
                raise ValueError("Gerekli kolonlar (OSD.flyTime [s], OSD.latitude, OSD.longitude) CSV'de bulunamadı!")
                
            self.gps_data[self.time_col] = pd.to_numeric(self.gps_data[self.time_col], errors='coerce')
            self.gps_data[self.lat_col] = pd.to_numeric(self.gps_data[self.lat_col], errors='coerce')
            self.gps_data[self.lon_col] = pd.to_numeric(self.gps_data[self.lon_col], errors='coerce')
            self.gps_data = self.gps_data.dropna(subset=[self.time_col, self.lat_col, self.lon_col])
            
            self.start_gps_time = self.gps_data.iloc[0][self.time_col]
            print(f"   {len(self.gps_data)} GPS kaydı başarıyla yüklendi.")
            print(f"   Log başlangıç zamanı: {self.start_gps_time}s")
        except Exception as e:
            print(f" GPS verisi okunurken kritik hata: {e}")
            raise
            
        self.frame_count = 0
        self.current_lat = 0.0
        self.current_lon = 0.0

    def get_gps_for_time(self, video_time_sec):
        """Video zamanına karşılık gelen en yakın GPS verisini bulur"""
        # Videonun başlangıcının log'un başlangıcına denk geldiğini varsayıyoruz
        target_time = self.start_gps_time + video_time_sec
        # En yakın zamanı bul
        idx = (np.abs(self.gps_data[self.time_col] - target_time)).argmin()
        row = self.gps_data.iloc[idx]
        return row[self.lat_col], row[self.lon_col]

    def detect_fire(self, frame):
        """Frame'de yangın tespiti yap"""
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            if prediction > 0.5:
                result = "NO FIRE"
                confidence = prediction * 100
            else:
                result = "FIRE"
                confidence = (1 - prediction) * 100
            
            return result, confidence
        except Exception as e:
            print(f"   Yangın tespit hatası: {e}")
            return "UNKNOWN", 0

    def process_video(self, output_file="fire_detection_results.txt"):
        """Videoyu işle ve sonuçları kaydet"""
        print(f"\n İşleniyor: Her saniye için yangın tespiti yapılıyor...")
        print(f"   (Toplam ~{int(self.duration)} saniye işlenecek)\n")
        
        frames_to_skip = int(self.fps)
        result_file = open(output_file, "w", encoding="utf-8")
        result_file.write("Saniye,Frame,Latitude,Longitude,Yangin_Durumu,Guven\n")
        
        fire_count = 0
        nofire_count = 0
        
        last_result = "BEKLENIYOR"
        last_confidence = 0.0
        
        cv2.namedWindow("Orman Yangini Tespiti", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Orman Yangini Tespiti", 800, 600)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                current_video_time = self.frame_count / self.fps
                
                # GPS Verisini Al
                self.current_lat, self.current_lon = self.get_gps_for_time(current_video_time)
                
                # --- Ekranda Görüntüleme Bölümü ---
                display_frame = frame.copy()
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
                
                textColor = (0, 0, 255) if last_result == "FIRE" else (0, 255, 0)
                
                cv2.putText(display_frame, f"Durum: {last_result} (%{last_confidence:.1f})", 
                            (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, textColor, 2)
                cv2.putText(display_frame, f"Enlem:  {self.current_lat:.6f}", 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Boylam: {self.current_lon:.6f}", 
                            (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Sure: {current_video_time:.1f}s / {self.duration:.1f}s", 
                            (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
                cv2.imshow("Orman Yangini Tespiti", display_frame)
                
                # Videoyu kendi hızında oynat (yaklaşık)
                wait_time = max(1, int(1000 / self.fps))
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    print("\nKullanici tarafindan 'q' tusu ile durduruldu!")
                    break
                
                # Saniyede 1 kere analiz yap
                if self.frame_count % frames_to_skip != 0 and self.frame_count != 1:
                    continue
                
                # Yangın tespiti
                result, confidence = self.detect_fire(frame)
                last_result = result
                last_confidence = confidence
                
                # İstatistik
                if result == "FIRE":
                    fire_count += 1
                else:
                    nofire_count += 1
                
                # Konsol çıktısı
                print(f"{current_video_time:6.2f}s | Frame: {self.frame_count:5d} | "
                      f"GPS: {self.current_lat:.8f}, {self.current_lon:.8f} | "
                      f"{result:8s}")
                
                # Dosyaya yaz
                result_file.write(f"{current_video_time:.2f},{self.frame_count},"
                                f"{self.current_lat:.10f},{self.current_lon:.10f},"
                                f"{result},{confidence:.2f}\n")
        
        except KeyboardInterrupt:
            print("\n\nKullanıcı tarafından durduruldu!")
        
        finally:
            result_file.close()
            self.cap.release()
            cv2.destroyAllWindows()
            
            # Özet rapor
            self.print_summary(fire_count, nofire_count, output_file)

    def print_summary(self, fire_count, nofire_count, output_file):
        """İşlem özetini yazdır"""
        total = fire_count + nofire_count
        
        print(f"\n{'='*80}")
        print("İŞLEM TAMAMLANDI")
        print(f"{'='*80}")
        print(f"\nİstatistikler:")
        print(f"   Toplam işlenen saniye: {total}")
        print(f"  Yangın tespit edilen: {fire_count} saniye")
        print(f"  Yangın yok: {nofire_count} saniye")
        
        if fire_count > 0:
            print(f"\n DİKKAT: {fire_count} farklı konumda YANGIN TESPİT EDİLDİ!")
        
        print(f"\n Sonuç dosyası: {output_file}")
        print(f"\n{'='*80}\n")


def main():
    """Ana program"""
    # Dosya yolları ayarlama
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    VIDEO_PATH = "video.mp4"
    CSV_PATH = "DJIFlightRecord_2019-01-09_[13-16-53].csv"
    MODEL_PATH = "forest_fire_model_withrgb.h5"
    
    video_path = os.path.join(script_dir, VIDEO_PATH)
    csv_path = os.path.join(script_dir, CSV_PATH)
    model_path = os.path.join(script_dir, MODEL_PATH)
    
    # Alternatif model yolları dene
    possible_model_paths = [
        model_path,
        os.path.join(os.path.dirname(script_dir), "forest_fire_model_withrgb.h5"),
    ]
    
    model_found = False
    for mp in possible_model_paths:
        if os.path.exists(mp):
            model_path = mp
            model_found = True
            break
            
    if not model_found:
        print("Model dosyası bulunamadı!")
        return

    try:
        system = FireDetectionWithGPSVideo(
            video_path=video_path,
            model_path=model_path,
            csv_path=csv_path
        )
        system.process_video(output_file="fire_detection_results.txt")
        
    except Exception as e:
        print(f"\n HATA: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
