import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from pathlib import Path

# ==============================================================================
# ⚙️ AYARLAR
# ==============================================================================
HEDEF = r"C:\Users\Celal Demirsoy\Desktop\giysi_ai\deneme2"  
MODEL_ADI = "best.pt"
GUVEN_ESIGI = 0.25

# Çalışma Dizini
MEVCUT_KLASOR = os.path.dirname(os.path.abspath(__file__))
# ==============================================================================

# --- KLASÖR YÖNETİMİ (YOLO Tarzı: runs/detect/exp...) ---
def yeni_klasor_olustur():
    ana_yol = os.path.join(MEVCUT_KLASOR, "runs", "detect")
    os.makedirs(ana_yol, exist_ok=True)
    
    # Sıradaki exp klasörünü bul (exp1, exp2, exp3...)
    i = 1
    while True:
        klasor_adi = f"exp{i}" if i > 1 else "exp"
        tam_yol = os.path.join(ana_yol, klasor_adi)
        if not os.path.exists(tam_yol):
            os.makedirs(tam_yol)
            return tam_yol
        i += 1

# --- RENK KATALOĞU ---
RENK_KATALOGU = {
    "Siyah": (0, 0, 0), "Beyaz": (255, 255, 255), "Gri": (128, 128, 128),
    "Kirmizi": (255, 0, 0), "Yesil": (0, 128, 0), "Mavi": (0, 0, 255),
    "Lacivert": (0, 0, 128), "Sari": (255, 255, 0), "Turuncu": (255, 165, 0),
    "Mor": (128, 0, 128), "Pembe": (255, 192, 203), "Bej": (245, 245, 220),
    "Kahverengi": (165, 42, 42), "Bordo": (128, 0, 0), "Krem": (255, 253, 208)
}

def en_yakin_rengi_bul(bulunan_rgb):
    min_mesafe = float("inf")
    en_yakin_isim = "Bilinmiyor"
    r1, g1, b1 = int(bulunan_rgb[0]), int(bulunan_rgb[1]), int(bulunan_rgb[2])
    for isim, (r2, g2, b2) in RENK_KATALOGU.items():
        rmean = int((r1 + int(r2)) / 2)
        r = r1 - int(r2)
        g = g1 - int(g2)
        b = b1 - int(b2)
        mesafe = np.sqrt((((512+rmean)*r*r)>>8) + 4*g*g + (((767-rmean)*b*b)>>8))
        if mesafe < min_mesafe:
            min_mesafe = mesafe
            en_yakin_isim = isim
    return en_yakin_isim

def rengi_analiz_et(image_crop):
    if image_crop.size == 0 or image_crop.shape[0] < 5 or image_crop.shape[1] < 5: return ""
    h, w, _ = image_crop.shape
    center_crop = image_crop[int(h*0.35):int(h*0.65), int(w*0.35):int(w*0.65)]
    try:
        preview_img = cv2.resize(center_crop, (64, 64), interpolation=cv2.INTER_AREA)
        image_array = preview_img.reshape((preview_img.shape[0] * preview_img.shape[1], 3))
        clt = KMeans(n_clusters=1, n_init='auto').fit(image_array)
        dominant_bgr = clt.cluster_centers_[0]
        dominant_rgb = (dominant_bgr[2], dominant_bgr[1], dominant_bgr[0])
        return en_yakin_rengi_bul(dominant_rgb)
    except: return ""

def kareyi_isle(frame, model):
    results = model.predict(frame, conf=GUVEN_ESIGI, verbose=False)
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = model.names[int(box.cls[0])]
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            renk = rengi_analiz_et(frame[y1:y2, x1:x2])
            label = f"{renk}_{cls_name}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, 0, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

def baslat():
    # KAYIT YERİNİ OLUŞTUR
    KAYIT_KLASORU = yeni_klasor_olustur()
    
    print(f"\n🚀 PROGRAM BAŞLATILIYOR...")
    print(f"📂 Sonuçlar Buraya Kaydedilecek: {KAYIT_KLASORU}")
    
    model_yolu = os.path.join(MEVCUT_KLASOR, MODEL_ADI)
    if not os.path.exists(model_yolu):
        print("❌ HATA: Model bulunamadı.")
        return
    model = YOLO(model_yolu)
    
    dosya_listesi = []
    if str(HEDEF) == "0":
        print("📷 Kamera Modu")
        cap = cv2.VideoCapture(0)
        # Kamera kaydı için video dosyası oluştur
        w, h = int(cap.get(3)), int(cap.get(4))
        vid_yol = os.path.join(KAYIT_KLASORU, "kamera_kaydi.mp4")
        out = cv2.VideoWriter(vid_yol, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = kareyi_isle(frame, model)
            out.write(frame)
            cv2.imshow("Kamera (Cikis: q)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        out.release()
        return

    if os.path.isdir(HEDEF):
        for f in os.listdir(HEDEF):
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.mp4', '.avi')):
                dosya_listesi.append(os.path.join(HEDEF, f))
    else:
        dosya_listesi.append(HEDEF)

    for dosya in dosya_listesi:
        dosya_adi = os.path.basename(dosya)
        cikti_yolu = os.path.join(KAYIT_KLASORU, dosya_adi)
        print(f"▶ İşleniyor: {dosya_adi}")
        
        if dosya.lower().endswith(('.mp4', '.avi')):
            cap = cv2.VideoCapture(dosya)
            w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
            writer = cv2.VideoWriter(cikti_yolu, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = kareyi_isle(frame, model)
                writer.write(frame)
                resized = cv2.resize(frame, (800, int(800*h/w)))
                cv2.imshow("Video Analiz", resized)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            cap.release()
            writer.release()
        else:
            frame = cv2.imread(dosya)
            if frame is None: continue
            frame = kareyi_isle(frame, model)
            cv2.imwrite(cikti_yolu, frame)
            h, w = frame.shape[:2]
            show_frame = cv2.resize(frame, (800, int(800*h/w))) if w > 800 else frame
            cv2.imshow("Resim Analiz", show_frame)
            cv2.waitKey(1000)

    cv2.destroyAllWindows()
    print(f"\n✅ İŞLEM TAMAM! Dosyalara buradan bakabilirsin:\n📂 {KAYIT_KLASORU}")

if __name__ == "__main__":
    baslat()