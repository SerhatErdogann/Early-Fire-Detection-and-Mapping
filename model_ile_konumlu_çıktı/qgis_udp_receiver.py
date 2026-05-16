import json
import socket
import base64
import os
import tempfile
from PyQt5.QtCore import QThread, pyqtSignal, QVariant
from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsField,
    QgsRuleBasedRenderer,
    QgsSymbol,
    QgsMarkerSymbol
)

# ==========================================
# 1. AYARLAR
# ==========================================
UDP_IP = "0.0.0.0"  # Tüm ağlardan gelen veriyi dinle
UDP_PORT = 5005     # Python kodundaki port ile aynı olmalı

# QGIS Map Tip html'inin resimleri geçici olarak okuyacağı klasör
TEMP_IMG_DIR = os.path.join(tempfile.gettempdir(), "qgis_fire_imgs")
if not os.path.exists(TEMP_IMG_DIR):
    os.makedirs(TEMP_IMG_DIR)

# ==========================================
# 2. UDP DİNLEYİCİ İŞ PARÇACIĞI (THREAD)
# ==========================================
class UdpListener(QThread):
    # Yeni veri geldiğinde emit edeceğimiz sinyal
    data_received = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.settimeout(1.0) # Arayüzü dondurmamak için timeout

    def run(self):
        print(f"📡 ZZZ... UDP {UDP_PORT} portu dinleniyor...")
        while self.running:
            try:
                data, addr = self.sock.recvfrom(65535) 
                json_data = json.loads(data.decode('utf-8'))
                
                # Resim base64 ise, onu fiziksel dosyaya çevir
                if "image" in json_data and json_data["image"]:
                    img_data = base64.b64decode(json_data["image"])
                    img_filename = f"fire_{json_data['time']}.jpg"
                    img_path = os.path.join(TEMP_IMG_DIR, img_filename)
                    
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    
                    # C:\Users\... yolunu C:/Users/... yapar (QGIS HTML Motoru için ZORUNLUDUR)
                    img_path_qgis = img_path.replace('\\', '/')
                    
                    json_data["image_path"] = img_path_qgis
                    del json_data["image"]
                else:
                    json_data["image_path"] = ""
                    
                # Arayüze sinyal gönder
                self.data_received.emit(json_data)
                
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Hata: {e}")

    def stop(self):
        self.running = False
        self.sock.close()

# ==========================================
# 3. HARİTA (LAYER) VE STİL HAZIRLIĞI
# ==========================================
layer_name = "Canlı Yangın Takibi"

existing_layers = QgsProject.instance().mapLayersByName(layer_name)
if existing_layers:
    # Katman zaten varsa, içindeki eski noktaları sil (Katmanı komple SİLME)
    vl = existing_layers[0]
    pr = vl.dataProvider()
    pr.deleteFeatures([f.id() for f in vl.getFeatures()])
    
    # Yeni bir Katman eklemediğimiz için Map Tip (HTML) ayarlarınız kalıcı kalır.
else:
    # Katman yoksa yeni oluştur
    vl = QgsVectorLayer("Point?crs=epsg:4326", layer_name, "memory")
    pr = vl.dataProvider()
    
    pr.addAttributes([
        QgsField("time", QVariant.Double),
        QgsField("status", QVariant.String),
        QgsField("confidence", QVariant.Double),
        QgsField("image_path", QVariant.String)
    ])
    vl.updateFields()

# --- STİL (SYMBOLOGY) AYARLAMASI (Kırmızı/Yeşil Noktalar) ---
    symbol_fire = QgsMarkerSymbol.createSimple({'name': 'circle', 'color': '255,0,0', 'size': '4', 'outline_color': 'black'})
    symbol_nofire = QgsMarkerSymbol.createSimple({'name': 'circle', 'color': '0,255,0', 'size': '3', 'outline_color': 'black'})
    
    rules = (
        ('FIRE', '"status" = \'FIRE\'', symbol_fire),
        ('NO FIRE', '"status" = \'NO FIRE\'', symbol_nofire)
    )
    
    root_rule = QgsRuleBasedRenderer.Rule(None)
    for label, expression, symbol in rules:
        rule = QgsRuleBasedRenderer.Rule(symbol)
        rule.setLabel(label)
        rule.setFilterExpression(expression)
        root_rule.appendChild(rule)
    
    renderer = QgsRuleBasedRenderer(root_rule)
    vl.setRenderer(renderer)
    
    # --- MAP TIPS (HARİTA İPUÇLARI / HOVER) HTML AYARI ---
    html_code = """
    <h1>YANGIN DURUMU: [% status %]</h1>
    <b>Güven:</b> [% confidence %]% <br>
    <b>Zaman:</b> [% time %]s <br>
    <img src="file:///[% image_path %]" width="320">
    """
    vl.setDisplayExpression(html_code)
    
    # --- ÖNEMLİ: Katmanı QGIS haritasına ekle ---
    QgsProject.instance().addMapLayer(vl)

# ==========================================
# 4. GELEN VERİYİ HARİTAYA İŞLEME FONKSİYONU
# ==========================================
def process_data(data):
    feat = QgsFeature()
    feat.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(data["lon"], data["lat"])))
    
    feat.setAttributes([
        data.get("time", 0),
        data.get("status", "UNKNOWN"),
        data.get("confidence", 0),
        data.get("image_path", "")
    ])
    
    pr.addFeature(feat)
    vl.updateExtents()
    vl.triggerRepaint()

# ==========================================
# 5. SİSTEMİ BAŞLAT
# ==========================================
if 'udp_thread' in globals():
    try:
        udp_thread.stop()
        udp_thread.wait(2000)
    except:
        pass

udp_thread = UdpListener()
udp_thread.data_received.connect(process_data)
udp_thread.start()

print("1. Menüden (Harita İpuçları) basılı olduğundan emin olun.")
print("2. Farenizle kırmızı noktaların üzerine GELİN VE 1 SANİYE BEKLEYİN.")
print("3. Kapatmak istediğinizde konsola `udp_thread.stop()` yazabilirsiniz.")
