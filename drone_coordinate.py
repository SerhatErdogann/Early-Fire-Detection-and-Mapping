import cv2
import numpy as np
from datetime import datetime
import math

class DroneCoordinateTracker:
    def __init__(self, video_path, start_lat, start_lon, altitude_meters=150):
        """
        Drone video koordinat takip sistemi
        
        Args:
            video_path: Video dosya yolu
            start_lat: Başlangıç enlem (latitude)
            start_lon: Başlangıç boylam (longitude)
            altitude_meters: Drone yüksekliği (metre)
        """
        import os
        
        # Video dosyasının varlığını kontrol et
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video dosyası bulunamadı: {video_path}\n"
                                   f"Mevcut klasör: {os.getcwd()}\n"
                                   f"Lütfen video dosyasının tam yolunu kontrol edin.")
        
        self.video_path = video_path
        self.current_lat = start_lat
        self.current_lon = start_lon
        self.altitude = altitude_meters
        
        # Video açma
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Video açılamadı: {video_path}\n"
                           f"Dosya var ama OpenCV açamıyor. Video formatını kontrol edin.\n"
                           f"Desteklenen formatlar: .mp4, .avi, .mov, .mkv")
        
        # Video özellikleri
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Bilgileri:")
        print(f"  Çözünürlük: {self.width}x{self.height}")
        print(f"  FPS: {self.fps}")
        print(f"  Toplam Kare: {self.total_frames}")
        print(f"  Süre: {self.total_frames/self.fps:.2f} saniye")
        print(f"  Drone Yüksekliği: {self.altitude}m")
        print(f"  Başlangıç Koordinatı: {self.current_lat}, {self.current_lon}")
        
        # Kamera açısı ve görüş alanı hesaplamaları
        # 150m yükseklikten tipik drone kamera için tahmini değerler
        # Varsayılan: 84° FOV (Field of View)
        self.fov_horizontal = 84  # derece
        self.fov_vertical = 53    # derece (16:9 aspect ratio için)
        
        # Yerdeki görüntü alanı hesaplama
        self.ground_width = 2 * self.altitude * math.tan(math.radians(self.fov_horizontal / 2))
        self.ground_height = 2 * self.altitude * math.tan(math.radians(self.fov_vertical / 2))
        
        print(f"\nYerdeki Görüntü Alanı:")
        print(f"  Genişlik: {self.ground_width:.2f}m")
        print(f"  Yükseklik: {self.ground_height:.2f}m")
        
        # Piksel başına metre hesaplama
        self.meters_per_pixel_x = self.ground_width / self.width
        self.meters_per_pixel_y = self.ground_height / self.height
        
        print(f"\nPiksel Çözünürlüğü:")
        print(f"  X: {self.meters_per_pixel_x:.4f} m/pixel")
        print(f"  Y: {self.meters_per_pixel_y:.4f} m/pixel")
        
        # Optical flow için parametreler
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # İlk frame'i oku
        ret, self.old_frame = self.cap.read()
        if not ret:
            raise ValueError("İlk frame okunamadı!")
        
        self.old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        
        # Takip için özellik noktaları bul
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        
        # Log dosyası
        self.log_file = open("coordinate_log.txt", "w", encoding="utf-8")
        self.log_file.write("Zaman,Frame,Video_Zamani(s),Latitude,Longitude,Delta_X(m),Delta_Y(m),Delta_Lat,Delta_Lon,Toplam_X(m),Toplam_Y(m)\n")
        
        self.frame_count = 0
        self.total_displacement_x = 0
        self.total_displacement_y = 0

    def meters_to_lat_lon(self, dx_meters, dy_meters):
        """
        Metre cinsinden yer değiştirmeyi GPS koordinatlarına çevir
        
        Args:
            dx_meters: Doğu-Batı yönünde hareket (metre)
            dy_meters: Kuzey-Güney yönünde hareket (metre)
            
        Returns:
            (delta_lat, delta_lon): Koordinat değişimi
        """
        # Dünya'nın ekvator çapı yaklaşık 40,075 km
        # 1 derece enlem ≈ 111,320 metre (sabit)
        # 1 derece boylam ≈ 111,320 * cos(latitude) metre (enlemle değişir)
        
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(self.current_lat))
        
        delta_lat = dy_meters / meters_per_degree_lat
        delta_lon = dx_meters / meters_per_degree_lon
        
        return delta_lat, delta_lon

    def calculate_movement(self, p0, p1):
        """
        Optical flow'dan hareket vektörünü hesapla
        
        Args:
            p0: Eski nokta pozisyonları
            p1: Yeni nokta pozisyonları
            
        Returns:
            (avg_dx, avg_dy): Ortalama piksel hareketi
        """
        if p0 is None or p1 is None or len(p0) == 0 or len(p1) == 0:
            return 0, 0
        
        # Hareket vektörlerini hesapla
        movements = p1 - p0
        
        # Array shape'ini kontrol et ve düzelt
        if movements.ndim == 3:
            # Shape: (N, 1, 2) -> (N, 2)
            movements = movements.reshape(-1, 2)
        
        # RANSAC benzeri filtreleme - outlier'ları çıkar
        median_x = np.median(movements[:, 0])
        median_y = np.median(movements[:, 1])
        
        std_x = np.std(movements[:, 0])
        std_y = np.std(movements[:, 1])
        
        # Median'dan çok uzak olan noktaları filtrele
        threshold = 2.0  # standart sapma katı
        mask = (np.abs(movements[:, 0] - median_x) < threshold * std_x) & \
               (np.abs(movements[:, 1] - median_y) < threshold * std_y)
        
        if np.sum(mask) == 0:
            return 0, 0
        
        filtered_movements = movements[mask]
        
        avg_dx = np.mean(filtered_movements[:, 0])
        avg_dy = np.mean(filtered_movements[:, 1])
        
        return avg_dx, avg_dy

    def process_video(self, display=True, update_interval=1):
        """
        Videoyu işle ve koordinatları güncelle
        
        Args:
            display: Videoyu ekranda göster (True/False)
            update_interval: Kaç saniyede bir frame işle (1 = saniyede 1 frame)
        """
        print(f"\n{'='*80}")
        print("VIDEO İŞLEME BAŞLIYOR...")
        print(f"İşleme modu: Saniyede {update_interval} frame")
        print(f"{'='*80}\n")
        
        # Windows'ta GUI desteği yoksa display'i kapat
        try:
            if display:
                cv2.namedWindow("test", cv2.WINDOW_NORMAL)
                cv2.destroyWindow("test")
        except:
            print("UYARI: OpenCV GUI desteği bulunamadı. Sadece konsol çıktısı verilecek.")
            display = False
        
        color = np.random.randint(0, 255, (100, 3))
        mask = np.zeros_like(self.old_frame)
        
        # Saniyede kaç frame işleneceğini hesapla
        frames_to_skip = int(self.fps / update_interval)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Sadece belirlenen aralıkta frame'leri işle
                if self.frame_count % frames_to_skip != 0 and self.frame_count != 1:
                    continue
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Optical flow ile hareket takibi
                if self.p0 is not None and len(self.p0) > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, 
                                                           self.p0, None, **self.lk_params)
                    
                    if p1 is not None and st is not None:
                        # İyi takip edilen noktaları seç
                        good_new = p1[st == 1]
                        good_old = self.p0[st == 1]
                        
                        if len(good_new) > 5:  # En az 5 nokta takip edilebiliyorsa
                            # Hareket hesapla
                            avg_dx_pixel, avg_dy_pixel = self.calculate_movement(good_old, good_new)
                            
                            # Pikseli metreye çevir (frames_to_skip kadar frame atlandığı için çarp)
                            dx_meters = -avg_dx_pixel * self.meters_per_pixel_x * frames_to_skip
                            dy_meters = -avg_dy_pixel * self.meters_per_pixel_y * frames_to_skip
                            
                            # Toplam yer değiştirmeyi güncelle
                            self.total_displacement_x += dx_meters
                            self.total_displacement_y += dy_meters
                            
                            # GPS koordinatlarını güncelle
                            delta_lat, delta_lon = self.meters_to_lat_lon(dx_meters, dy_meters)
                            self.current_lat += delta_lat
                            self.current_lon += delta_lon
                            
                            # Konsola çıktı ver
                            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                            video_time = self.frame_count / self.fps
                            
                            print(f"[{timestamp}] Frame: {self.frame_count:5d}/{self.total_frames} | "
                                  f"Zaman: {video_time:6.2f}s | "
                                  f"GPS: {self.current_lat:.10f}, {self.current_lon:.10f} | "
                                  f"Δ: X={dx_meters:+.3f}m Y={dy_meters:+.3f}m | "
                                  f"Toplam: X={self.total_displacement_x:+.2f}m Y={self.total_displacement_y:+.2f}m")
                            
                            # Log dosyasına yaz
                            self.log_file.write(f"{timestamp},{self.frame_count},{video_time:.2f},"
                                               f"{self.current_lat:.10f},{self.current_lon:.10f},"
                                               f"{dx_meters:.4f},{dy_meters:.4f},"
                                               f"{delta_lat:.12f},{delta_lon:.12f},"
                                               f"{self.total_displacement_x:.4f},{self.total_displacement_y:.4f}\n")
                            self.log_file.flush()
                            
                            # Görselleştirme için çizimler
                            if display:
                                try:
                                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                                        a, b = new.ravel()
                                        c, d = old.ravel()
                                        a, b, c, d = int(a), int(b), int(c), int(d)
                                        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                                        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                                    
                                    img = cv2.add(frame, mask)
                                    
                                    # Bilgi metni ekle
                                    cv2.putText(img, f"GPS: {self.current_lat:.6f}, {self.current_lon:.6f}", 
                                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    cv2.putText(img, f"Frame: {self.frame_count}/{self.total_frames}", 
                                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    cv2.putText(img, f"Movement: ({dx_meters:+.2f}m, {dy_meters:+.2f}m)", 
                                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    cv2.imshow('Drone Coordinate Tracker', img)
                                    
                                    # 'q' tuşu ile çıkış
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        print("\nKullanıcı tarafından durduruldu!")
                                        break
                                except:
                                    display = False  # Görselleştirme hatası olursa kapat
                    
                    # Yeni özellik noktaları bul
                    self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
                    if display:
                        mask = np.zeros_like(frame)
                
                self.old_gray = frame_gray.copy()
        
        except KeyboardInterrupt:
            print("\n\nProgram kullanıcı tarafından durduruldu!")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Kaynakları temizle"""
        print(f"\n{'='*80}")
        print("ÖZET RAPOR")
        print(f"{'='*80}")
        print(f"Başlangıç Koordinatı: 39.971122, 32.818559")
        print(f"Bitiş Koordinatı:     {self.current_lat:.6f}, {self.current_lon:.6f}")
        print(f"Toplam Yer Değiştirme: ({self.total_displacement_x:.2f}m, {self.total_displacement_y:.2f}m)")
        print(f"İşlenen Frame Sayısı: {self.frame_count}")
        print(f"\nLog dosyası 'coordinate_log.txt' olarak kaydedildi.")
        print(f"{'='*80}\n")
        
        self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass  # GUI desteği yoksa hata vermesin
        self.log_file.close()


def main():
    """Ana program"""
    import os
    

    video_path = "video.mp4"  
    
    # Başlangıç koordinatları
    start_latitude = 39.971122237497404
    start_longitude = 32.81855850529272
    altitude = 150  # metre
    
    print("="*80)
    print("DRONE VIDEO GPS KOORDİNAT TAKİP SİSTEMİ")
    print("="*80)
    print(f"\nMevcut klasör: {os.getcwd()}")
    print(f"Aranan video: {video_path}")
    
    # Video dosyasının tam yolunu kontrol et
    if not os.path.isabs(video_path):
        full_path = os.path.join(os.getcwd(), video_path)
        print(f"Tam yol: {full_path}")
    
    # Klasördeki video dosyalarını listele
    print(f"\nMevcut klasördeki video dosyaları:")
    video_extensions = ['.mp4']
    found_videos = []
    for file in os.listdir(os.getcwd()):
        if any(file.endswith(ext) for ext in video_extensions):
            found_videos.append(file)
            print(f"  - {file}")
    
    if not found_videos:
        print("  (Video dosyası bulunamadı)")
    print()
    
    try:
        # Tracker oluştur
        tracker = DroneCoordinateTracker(
            video_path=video_path,
            start_lat=start_latitude,
            start_lon=start_longitude,
            altitude_meters=altitude
        )
        
        # Videoyu işle
        # display=True: Videoyu ekranda göster
        # update_interval=1: Her frame'de koordinat güncelle ve göster
        tracker.process_video(display=True, update_interval=1)
        
    except FileNotFoundError as e:
        print(f"\n{'='*80}")
        print("HATA: Video dosyası bulunamadı!")
        print(f"{'='*80}")
        print(str(e))
        print(f"\nÇözüm:")
        print(f"1. Video dosyasının adını kontrol edin")
        print(f"2. Video dosyasını script ile aynı klasöre koyun")
        print(f"3. veya tam yolu yazın: video_path = 'C:/Users/.../video.mp4'")
        if found_videos:
            print(f"\nKlasörde bulunan videolar: {', '.join(found_videos)}")
            print(f"Bunlardan birini kullanmak için video_path değişkenini güncelleyin.")
    except Exception as e:
        print(f"\nHATA: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()