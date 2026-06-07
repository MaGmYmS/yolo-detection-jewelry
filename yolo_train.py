"""
🚀 Скрипт обучения YOLO для GTX 1650 (4GB VRAM)
Оптимизирован для минимального потребления видеопамяти.
"""

from ultralytics import YOLO
import os
import torch
import gc

# ================= НАСТРОЙКИ ДЛЯ 4GB VRAM =================
DATASET_YAML = "yolo_dataset_augmented/data.yaml"

# 1. МОДЕЛЬ: Используем только Nano (n) или Small (s). 
# Medium (m) и выше гарантированно вызовут нехватку памяти.
PRETRAINED_MODEL = "yolov26n.pt"  # Начните с 'n'. Если всё пойдет хорошо, можно попробовать 's'

# 2. РАЗРЕШЕНИЕ: 640 или максимум 800. 
# YOLO сам сожмет ваши 1280x1280 до этого размера при загрузке. 
# Это сэкономит ~60% видеопамяти без критической потери качества для начала.
IMG_SIZE = 960 

# 3. BATCH SIZE: Строго 1 или 2. 
# Если получите ошибку OOM, немедленно меняйте на 1.
BATCH_SIZE = 8 

EPOCHS = 100
PATIENCE = 5
PROJECT_DIR = "runs/detect"
EXPERIMENT_NAME = "jewelry_v1"
DEVICE = 0
# ==========================================================

def main():
    # Очистка кэша перед стартом (полезно на Windows)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"🚀 Запуск обучения на GTX 1650 (Оптимизированный режим)...")
    print(f"📂 Датасет : {DATASET_YAML}")
    print(f"🤖 Модель  : {PRETRAINED_MODEL}")
    print(f"📐 Размер  : {IMG_SIZE}x{IMG_SIZE}")
    print(f"🔄 Эпохи   : {EPOCHS} | Batch: {BATCH_SIZE}")
    print("-" * 60)

    model = YOLO(PRETRAINED_MODEL)

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        patience=PATIENCE,
        save=True,
        save_period=10,
        optimizer='auto',
        device=DEVICE,
        close_mosaic=10,         
        verbose=True,
    )

    print("\n" + "="*60)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"🏆 Лучшая модель: {os.path.join(PROJECT_DIR, EXPERIMENT_NAME, 'weights', 'best.pt')}")
    print("="*60)

    # Финальная валидация
    print("\n🔍 Запуск финальной валидации...")
    metrics = model.val(data=DATASET_YAML, split='val', device=DEVICE, batch=BATCH_SIZE, workers=0)
    
    print("\n📊 ИТОГОВЫЕ МЕТРИКИ:")
    print(f"   • mAP50     : {metrics.box.map50:.4f}")
    print(f"   • mAP50-95  : {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()