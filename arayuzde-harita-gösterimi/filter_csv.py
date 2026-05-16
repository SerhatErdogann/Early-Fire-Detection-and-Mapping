import pandas as pd

def filtrele():
    print("Veri yukleniyor, lutfen bekleyin...")
    # 1. Veriyi Oku
    df = pd.read_csv('fire_archive_SV-C2_745116.csv')

    # 2. Sadece dogal bitki/orman yanginlari (type == 0)
    df_filtered = df[df['type'] == 0]

    # 3. Confidence filtrelemesi
    df_h = df_filtered[df_filtered['confidence'] == 'h']
    df_n = df_filtered[df_filtered['confidence'] == 'n']

    if len(df_h) >= 2000:
        # Eger 2000'den fazla kesin (h) yangin varsa, sadece onlardan 2000 sec
        final_df = df_h.sample(n=2000, random_state=42)
    else:
        # Eger 'h' ler 2000'den azsa, hepsini al
        # Geri kalani 'n' lerden (normal guvenilirlik) rastgele secip tamamla
        kalan_ihtiyac = 2000 - len(df_h)
        # N leri de secerken hata olmamasi icin sinir kontrolu
        kalan_ihtiyac = min(kalan_ihtiyac, len(df_n))
        secilen_n = df_n.sample(n=kalan_ihtiyac, random_state=42)
        final_df = pd.concat([df_h, secilen_n])

    # 4. Veriyi karistir (kutuplasmayi onlemek icin)
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Sadece isimize yarayacak sutunlari tutalim (islem hizi icin)
    final_df = final_df[['latitude', 'longitude', 'acq_date', 'confidence']]

    # 6. Kaydet
    final_df.to_csv('filtered_fires_2k.csv', index=False)

    print("\n--- FILTRELEME SONUCLARI ---")
    print(f"Orijinal Toplam Satir: {len(df)}")
    print(f"Doğal Yangın (Type=0) Satir: {len(df_filtered)}")
    print(f"High (h) Confidence Satir: {len(df_h)}")
    print(f"Normal (n) Confidence Satir: {len(df_n)}")
    print("-" * 28)
    print(f"SECILEN TOPLAM SATIR: {len(final_df)}")
    print("Yeni dosya 'filtered_fires_2k.csv' olarak basariyla kaydedildi.")

if __name__ == "__main__":
    filtrele()
