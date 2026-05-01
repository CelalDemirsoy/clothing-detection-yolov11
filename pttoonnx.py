from ultralytics import YOLO

# Modeli yükle
model = YOLO('best.pt')

# ONNX formatına export et
model.export(
    format='onnx',
    imgsz=640,      # Görüntü boyutu (modelinize göre değiştirin)
    opset=12,       # ONNX opset versiyonu
    simplify=True,  # Modeli sadeleştir
    dynamic=False   # Sabit batch size için False
)

print("Model başarıyla best.onnx olarak kaydedildi!")