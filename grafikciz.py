import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# AYARLAR
# ==============================================================================
# İndirdiğin csv dosyasının adı veya tam yolu
# Eğer kodla aynı klasördeyse sadece ismini yazman yeterli
DOSYA_YOLU = "results.csv"  

# ==============================================================================
# ANALİZ VE ÇİZİM MOTORU
# ==============================================================================
def performans_raporu_olustur():
    if not os.path.exists(DOSYA_YOLU):
        print(f"❌ HATA: '{DOSYA_YOLU}' bulunamadı! İndirdiğinden emin misin?")
        return

    # 1. Veriyi Oku
    try:
        df = pd.read_csv(DOSYA_YOLU)
        # Sütun isimlerindeki boşlukları temizle (YOLO genelde boşluklu kaydeder)
        df.columns = [c.strip() for c in df.columns]
    except Exception as e:
        print(f"CSV Okuma Hatası: {e}")
        return

    # 2. F1 Score Hesapla (YOLO csv'sinde direkt yazmaz, biz hesaplarız)
    # Formül: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    eps = 1e-7 # Sıfıra bölünme hatasını önlemek için
    df['F1'] = 2 * (df['metrics/precision(B)'] * df['metrics/recall(B)']) / \
               (df['metrics/precision(B)'] + df['metrics/recall(B)'] + eps)

    # 3. En İyi Değerleri Bul
    best_epoch = df['metrics/mAP50(B)'].idxmax()
    
    print("\n" + "="*40)
    print(f"🏆 MODEL KARNESİ (En İyi Epoch: {best_epoch + 1})")
    print("="*40)
    print(f"🎯 Accuracy (mAP@50):  %{df['metrics/mAP50(B)'].max()*100:.2f}")
    print(f"⚖️ F1 Score (Denge):    %{df['F1'].max()*100:.2f}")
    print(f"✅ Precision (Kesinlik):%{df['metrics/precision(B)'].iloc[best_epoch]*100:.2f}")
    print(f"📡 Recall (Duyarlılık): %{df['metrics/recall(B)'].iloc[best_epoch]*100:.2f}")
    print("="*40 + "\n")

    # 4. Grafikleri Çiz (4'lü Panel)
    plt.style.use('ggplot') # Profesyonel görünüm
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Eğitim Sonuçları Analizi (Toplam {len(df)} Epoch)', fontsize=16, fontweight='bold')

    # Grafik 1: mAP (Accuracy) - Ne kadar doğru bildi?
    axs[0, 0].plot(df['metrics/mAP50(B)'], label='mAP@50 (Genel Doğruluk)', color='blue', linewidth=2)
    axs[0, 0].plot(df['metrics/mAP50-95(B)'], label='mAP@50-95 (Hassas Doğruluk)', color='cyan', linestyle='--')
    axs[0, 0].set_title('Model Başarısı (Yükselmeli)')
    axs[0, 0].set_ylabel('Skor (0-1.0)')
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid(True)

    # Grafik 2: F1 Score - Model dengeli mi?
    axs[0, 1].plot(df['F1'], label='F1 Score', color='purple', linewidth=2)
    axs[0, 1].set_title('F1 Score (Denge Puanı)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Grafik 3: Hata Oranları (Loss) - Hata azalıyor mu?
    axs[1, 0].plot(df['train/box_loss'], label='Eğitim Box Loss', color='red')
    axs[1, 0].plot(df['val/box_loss'], label='Test Box Loss', color='orange')
    axs[1, 0].set_title('Kutu Çizme Hatası (Düşmeli)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Grafik 4: Precision & Recall - Yanlış alarm vs Kaçırma
    axs[1, 1].plot(df['metrics/precision(B)'], label='Precision (Kesinlik)', color='green')
    axs[1, 1].plot(df['metrics/recall(B)'], label='Recall (Duyarlılık)', color='teal')
    axs[1, 1].set_title('Kesinlik ve Duyarlılık')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    
    # Resmi Kaydet
    plt.savefig("Performans_Grafigi.png", dpi=300)
    print("✅ Grafik 'Performans_Grafigi.png' olarak kaydedildi.")
    
    # Ekranda Göster
    plt.show()

if __name__ == "__main__":
    # Gerekli kütüphaneler yoksa uyar
    try:
        import pandas
        import matplotlib
        performans_raporu_olustur()
    except ImportError:
        print("Eksik kütüphane var! Terminale şunu yazıp enter'a bas:")
        print("pip install pandas matplotlib")