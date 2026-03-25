# Proje Raporu – Genel Özet (Rapor İçin Eklenebilir Metin)

Bu metin, bitirme/tez raporunda "Yapılan İşler", "Proje Özeti" veya "Geliştirme Aşamaları" bölümüne eklenebilecek şekilde yazılmıştır.

---

## 1. Amaç ve Kapsam

Projenin amacı, **İHA (drone) ile çekilmiş RGB ve termal görüntüleri kullanarak yangın tespiti (ateş var / ateş yok)** yapan bir sistem geliştirmektir. Sistem:

- Sadece RGB, sadece termal veya **RGB+termal birleşik (fusion)** modda çalışabilir.
- Eğitimden video çıkarımına, risk skorlamasına ve isteğe bağlı kullanıcı arayüzüne kadar uçtan uca bir pipeline sunar.
- FLAME veri seti ve ek binary (RGB+termal çift) verileriyle eğitilir; gerçek İHA videolarında denenebilir.

Hedef kitle: Orman/alan yangınlarının erken tespiti için İHA görüntülerini değerlendiren operatörler veya otomatik izleme sistemleri.

---

## 2. Nelerden Faydalanıldı?

- **FLAME veri seti:** Aerial (havadan) RGB ve termal görüntü çiftleri; Fire / No Fire sınıfları. IEEE Dataport üzerinden erişilebilir.
- **Binary (RGB+termal) veri:** `data/flame3/binary` altında train/Fire ve train/No_Fire yapısında ek çiftler; path listesi `rgbt_multimodal_data.csv` ile verilmiştir.
- **Mimari ve eğitim fikirleri:** Fire-Detection-UAV (konfigürasyon, U-Net benzeri yapı), wildfire-detection (YOLO dayanıklılığı), Wildfire-Smoke-Detection (augmentation ve eğitim ayarları) projelerinden esinlenilmiştir.
- **Teknolojiler:** Python, PyTorch, ResNet (backbone), mixed-precision (AMP), Focal loss, temperature scaling, Streamlit/Gradio arayüzleri.

---

## 3. Nasıl Bu Aşamaya Gelindi? (Geliştirme Aşamaları)

1. **Veri ve indeks yapısı**  
   FLAME klasör yapısı (Fire / No Fire, RGB “Corrected FOV”, Thermal “Celsius TIFF”) ve isteğe bağlı “extra” no-fire verisi tek bir indeks CSV’de toplandı. Bu CSV’de her satır bir RGB–termal çiftini ve etiketini (ateş var=1, yok=0) temsil eder.

2. **Binary verinin entegrasyonu**  
   Binary veri seti (`data/flame3/binary` ve `rgbt_multimodal_data.csv`) projeye eklendi. CSV’deki path’ler okunup RGB–termal eşleştirmesi path kurallarına göre yapıldı (_rgb/_thermal veya _w/_t); etiket path’te “No_Fire” geçip geçmemesine göre atandı. Böylece hem FLAME hem binary kaynaklı çiftler aynı indekste birleştirildi.

3. **Disk taraması ile eksik çiftlerin tamamlanması**  
   CSV’de listelenmeyen ancak diskte bulunan binary Fire/No Fire klasörleri taranarak, RGB–termal çiftleri (aynı frame adına göre) indekse eklendi. Bu sayede No Fire örnek sayısı mümkün olduğunca artırıldı.

4. **Eğitim pipeline’ı**  
   İndeks CSV’ye göre veri seti oluşturuldu; RGB (3 kanal), termal (1 kanal) ve fusion (4 kanal) modları için ayrı modeller eğitildi. Sınıf dengesizliğine karşı Focal loss ve ağırlıklı örnekleyici (weighted sampler) kullanıldı; mixed-precision ve early stopping ile eğitim süresi ve overfitting kontrol altına alındı. Val setinde F1’e göre eşik seçilip checkpoint’e yazıldı; temperature scaling ile olasılık kalibrasyonu yapıldı.

5. **Video çıkarımı ve kararlılık**  
   Videodan frame okuma, aynı ön işleme (prep_rgb / prep_thermal) ve model yükleme modüler hale getirildi. EMA yumuşatma ve isteğe bağlı TTA ile kareler arası titreme azaltıldı; çıktı CSV ve heatmap’ler üretildi.

6. **Risk skoru ve arayüz**  
   Video çıkarım çıktısına ağırlıklı risk skoru (olasılık, yoğunluk, alan) eklendi; ardışık yüksek olasılık ile “yangın olayı” bayrağı üretildi. Streamlit/Gradio ile görüntü/video denemesi ve eşik kalibrasyonu için adımlar tanımlandı.

Bu adımlar sonucunda proje: **veri toplama–indeksleme → eğitim (RGB/termal/fusion) → video çıkarımı → risk skoru → arayüz ve eşik ayarı** akışına ulaşmıştır.

---

## 4. Ne Yapıldı? (Özet Liste)

- **Veri:** FLAME + binary verilerinin tek indeks CSV’de birleştirilmesi; path tabanlı RGB–termal eşleştirme ve etiketleme; disk taraması ile eksik çiftlerin eklenmesi.
- **Eğitim:** RGB, termal ve fusion modları; ResNet backbone; Focal loss + balanced sampler; AMP, temperature scaling, early stopping; val F1’e göre eşik kaydı.
- **Çıkarım:** Video üzerinde frame bazlı tahmin; EMA ve TTA ile kararlı çıktı; heatmap (Grad-CAM) üretimi.
- **Risk ve arayüz:** Risk skoru ve yangın olayı bayrağı; Streamlit/Gradio ile deneme; manuel etiket ve eşik kalibrasyonu (08_calibrate_threshold) ile eşiğin iyileştirilmesi.

---

## 5. Sınırlılıklar ve Notlar

- Veri setinde No Fire örneği sayısı Fire’a göre daha az olabilir (özellikle binary CSV’de); bu dağılım veri kaynaklıdır. Disk taraması ve balanced sampler / Focal loss ile bu dengesizlik kısmen giderilmeye çalışılmıştır.
- Termal veri yoksa yalnızca RGB modeli kullanılır; termal varken false positive fazlaysa fusion yerine RGB modu önerilir (DOC.md).
- Eşik değeri varsayılan olarak checkpoint’te saklanır; gerçek videolarda kalibrasyon (manuel etiket + 08_calibrate_threshold) ile iyileştirilebilir.

---

## 6. Raporda Kullanım Önerisi

- **Amaç:** Bölüm 1’i “Amaç ve Kapsam” veya “Projenin Hedefi” altında kullanabilirsin.
- **Kaynaklar:** Bölüm 2’yi “Kullanılan Veri Setleri ve Referanslar” veya “İlgili Çalışmalar / Araçlar” altında kullanabilirsin.
- **Yöntem / Geliştirme:** Bölüm 3’ü “Geliştirme Aşamaları”, “Yöntem” veya “Sistem Tasarımı” içinde adım adım anlatım olarak kullanabilirsin.
- **Sonuçlar / Özet:** Bölüm 4’ü “Yapılan İşler” veya “Sonuçlar” bölümünde madde madde özet olarak kullanabilirsin.
- **Tartışma:** Bölüm 5’i “Sınırlılıklar” veya “Tartışma” kısmında kısaca belirtebilirsin.

İstersen bu metni doğrudan raporuna kopyalayıp, bölüm numaralarını ve başlıkları kendi rapor yapına göre düzenleyebilirsin.
