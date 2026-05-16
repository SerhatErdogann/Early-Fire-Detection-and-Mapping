import pandas as pd
import os

def merge():
    print("Veri setleri birlestiriliyor...")
    
    # 1. Yeni (Temporal) Veri Setini Yukle
    if not os.path.exists('temporal_ml_dataset.csv'):
        print("Hata: temporal_ml_dataset.csv bulunamadi!")
        return
    df_temporal = pd.read_csv('temporal_ml_dataset.csv')
    
    # 2. Eski (Spatial) Veri Setini Yukle
    if not os.path.exists('final_ml_dataset.csv'):
        print("Hata: final_ml_dataset.csv bulunamadi!")
        return
    df_old = pd.read_csv('final_ml_dataset.csv')
    
    # 3. Eski veri setinden "Yanmaz" alanlari sec (Label=0 ve LC=0,6,7,8)
    # Su (0), Sehir (6), Ciplak Toprak (7), Kar (8)
    df_extra_negatives = df_old[(df_old['label'] == 0) & (df_old['land_cover'].isin([0, 6, 7, 8]))]
    
    print(f"Eski veri setinden {len(df_extra_negatives)} adet yanmaz alan (Su, Sehir, Toprak) ayiklandi.")
    
    # 4. Birlestir
    df_final = pd.concat([df_temporal, df_extra_negatives], ignore_index=True)
    
    # 5. Kaydet
    output_name = 'final_merged_dataset.csv'
    df_final.to_csv(output_name, index=False)
    
    print(f"Birlestirme tamamlandi! Toplam satir sayisi: {len(df_final)}")
    print(f"Dosya kaydedildi: {output_name}")

if __name__ == "__main__":
    merge()
