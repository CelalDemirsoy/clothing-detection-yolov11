import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==============================================================================
# 1. AYARLAR
# ==============================================================================
# YAML dosyanın tam yolu
YAML_DOSYASI = r"C:\Users\Celal Demirsoy\Desktop\giysi_ai_dataset\data.yaml"

# Sonuçların kaydedileceği klasörler
PROJE_ADI = "Moda_Projesi"
DENEY_ADI = "YOLO11_Egitimi"

# ==============================================================================
# 2. GRAFİK ÇİZME MOTORU (Eğitimden Sonra Çalışır)
# ==============================================================================
def grafikleri_olustur(kayit_klasoru):
    print(f"\n📊 Grafikler hazırlanıyor... Klasör: {kayit_klasoru}")
    
    csv_yolu = os.path.join(kayit_klasoru, "results.csv")
    
    if not os.path.exists(csv_yolu):
        print("❌ HATA: 'results.csv' bulunamadı! Eğitim hiç başlamamış olabilir.")
        return

    # Grafiklerin kaydedileceği klasör
    grafik_klasoru = os.path.join(kayit_klasoru, "Performans_Grafikleri")
    os.makedirs(grafik_klasoru, exist_ok=True)

    try:
        df = pd.read_csv(csv_yolu)
        df.columns = [c.strip() for c in df.columns] # Boşluk temizle
    except:
        print("⚠️ CSV dosyası okunamadı.")
        return

    # F1 Score Hesapla
    eps = 1e-7
    df['F1'] = 2 * (df['metrics/precision(B)'] * df['metrics/recall(B)']) / \
               (df['metrics/precision(B)'] + df['metrics/recall(B)'] + eps)

    epochs = range(1, len(df) + 1)

    # --- Grafik Çizme Yardımcı Fonksiyonu ---
    def tek_grafik(x, y_list, etiket_list, renk_list, baslik, dosya_adi):
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')
        for i in range(len(y_list)):
            plt.plot(x, y_list[i], label=etiket_list[i], color=renk_list[i], linewidth=2)
        plt.title(baslik, fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Değer')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(grafik_klasoru, dosya_adi), dpi=150)
        plt.close()
        print(f"   ✅ Kaydedildi: {dosya_adi}")

    # 1. Doğruluk (mAP)
    tek_grafik(epochs, [df['metrics/mAP50(B)'], df['metrics/mAP50-95(B)']], 
               ['mAP@50 (Genel)', 'mAP@50-95 (Hassas)'], ['blue', 'cyan'],
               'Model Doğruluğu (mAP)', '1_Dogruluk.png')

    # 2. F1 Score
    tek_grafik(epochs, [df['F1']], ['F1 Score'], ['purple'],
               'F1 Score (Denge)', '2_F1_Score.png')

    # 3. Precision
    tek_grafik(epochs, [df['metrics/precision(B)']], ['Precision'], ['green'],
               'Precision (Kesinlik)', '3_Precision.png')

    # 4. Recall
    tek_grafik(epochs, [df['metrics/recall(B)']], ['Recall'], ['orange'],
               'Recall (Duyarlılık)', '4_Recall.png')

    # 5. Box Loss
    tek_grafik(epochs, [df['train/box_loss'], df['val/box_loss']], 
               ['Eğitim', 'Test'], ['red', 'blue'],
               'Box Loss (Kutu Hatası)', '5_Box_Loss.png')

    # 6. Class Loss
    tek_grafik(epochs, [df['train/cls_loss'], df['val/cls_loss']], 
               ['Eğitim', 'Test'], ['red', 'blue'],
               'Class Loss (Sınıflandırma Hatası)', '6_Class_Loss.png')

    print(f"\n🎉 TÜM GRAFİKLER HAZIR! Klasöre bakabilirsin:\n📂 {grafik_klasoru}")

# ==============================================================================
# 3. ANA ÇALIŞMA MANTIĞI (Train + Auto-Graph)
# ==============================================================================
def sistemi_baslat():
    print(f"🚀 SİSTEM BAŞLATILIYOR... Model: YOLO11s")
    
    # Modeli Yükle
    model = YOLO("yolo11s.pt")
    
    # Kayıt klasörünü önceden belirle (YOLO'nun nereye kaydedeceğini bilmek için)
    # Eğer aynı isimde klasör varsa YOLO sonuna 2, 3 ekler. Bunu yakalamamız lazım.
    # En güvenli yol: save_dir özelliğini kullanmaktır ama biz manuel takip edelim.
    
    hedef_klasor = os.path.join(PROJE_ADI, DENEY_ADI)
    
    try:
        # --- EĞİTİM AŞAMASI ---
        print("⏳ Eğitim başlıyor...")
        print("💡 İPUCU: Eğitimi istediğin zaman 'Ctrl + C' ile durdurabilirsin.")
        print("   Durdursan bile o ana kadarki grafikler çizilecektir.")
        
        results = model.train(
            data=YAML_DOSYASI,
            epochs=100,
            patience=15,
            imgsz=640,
            batch=16,
            project=PROJE_ADI,
            name=DENEY_ADI,
            plots=True,      # YOLO'nun kendi grafiklerini de aç
            exist_ok=False   # False yaptık ki her seferinde yeni klasör açsın (exp, exp2...)
        )
        
        # Eğitim bittiğinde sonuç klasörünü al
        kayit_yeri = str(results.save_dir)
        print(f"\n✅ Eğitim Başarıyla Tamamlandı! Kayıt Yeri: {kayit_yeri}")
        
        # Grafikleri Çiz
        grafikleri_olustur(kayit_yeri)

    except KeyboardInterrupt:
        print("\n\n🛑 EĞİTİM KULLANICI TARAFINDAN DURDURULDU (Ctrl+C)!")
        print("📊 Yine de mevcut verilerle grafik çizilmeye çalışılıyor...")
        
        # Yarım kalan eğitimin klasörünü bulmaya çalış
        # Proje klasöründeki en son değiştirilen klasörü bulur
        if os.path.exists(PROJE_ADI):
            alt_klasorler = [os.path.join(PROJE_ADI, d) for d in os.listdir(PROJE_ADI) if os.path.isdir(os.path.join(PROJE_ADI, d))]
            if alt_klasorler:
                en_yeni_klasor = max(alt_klasorler, key=os.path.getmtime)
                grafikleri_olustur(en_yeni_klasor)
            else:
                print("❌ Klasör bulunamadı, grafik çizilemiyor.")
        
    except Exception as e:
        print(f"\n❌ BEKLENMEYEN BİR HATA OLUŞTU: {e}")

if __name__ == "__main__":
    sistemi_baslat()