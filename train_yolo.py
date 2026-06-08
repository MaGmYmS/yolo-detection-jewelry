from ultralytics import YOLO
import os
import torch
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==================== КОНФИГУРАЦИЯ (фиксированные параметры) ====================
BASE_DATASETS = {
    "no_aug": r"datasets\yolo_dataset\data.yaml",
    "augmented": r"datasets\yolo_dataset_augmented\data.yaml"
}
MODELS = {
    "n": "yolo26n.pt",
    "s": "yolo26s.pt"
}
IMG_SIZE = 960
BATCH_SIZE = 16
EPOCHS = 150
PROJECT_DIR = "runs/detect-n2"
DEVICE = 0

# Параметры для перебора 
GRID_SEARCH = {
    "dataset_key": ["no_aug", "augmented"],         
    "model_key": ["n"],                         
    "patience": [3, 5, 7],                      
    "warmup_epochs": [0, 5]                     
}

# GRID_SEARCH = {
#     "dataset_key": ["augmented"],          # 2 варианта
#     "model_key": ["n"],                         # 2 варианта
#     "patience": [5],                      # 4 варианта
#     "warmup_epochs": [0]                      # 3 варианта (0 = без warmup)
# }

# Дополнительные параметры, которые НЕ перебираются, но могут быть изменены
EXTRA_PARAMS = {
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "close_mosaic": 10,
    "optimizer": "auto"
}

# ==================== ФУНКЦИЯ ОБУЧЕНИЯ ОДНОГО ЭКСПЕРИМЕНТА ====================
def run_experiment(dataset_key, model_key, patience, warmup_epochs, extra_params):
    """Запускает один эксперимент. Возвращает словарь с результатами."""
    dataset_yaml = BASE_DATASETS[dataset_key]
    model_path = MODELS[model_key]
    exp_name = f"{dataset_key}_{model_key}_p{patience}_w{warmup_epochs}"
    
    print("\n" + "="*70)
    print(f"🚀 Эксперимент: {exp_name}")
    print(f"   Датасет : {dataset_yaml}")
    print(f"   Модель  : {model_path}")
    print(f"   Patience: {patience}")
    print(f"   Warmup  : {warmup_epochs} эпох")
    print("="*70)
    
    model = YOLO(model_path)
    train_results = model.train(
        data=dataset_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_DIR,
        name=exp_name,
        exist_ok=True,
        patience=patience,
        workers=2,
        save=True,
        save_period=10,
        device=DEVICE,
        verbose=True,
        warmup_epochs=warmup_epochs,
        lr0=extra_params["lr0"],
        lrf=extra_params["lrf"],
        momentum=extra_params["momentum"],
        weight_decay=extra_params["weight_decay"],
        close_mosaic=extra_params["close_mosaic"],
        optimizer=extra_params["optimizer"],
        seed=42,
    )
    
    # Валидация на test выборке для определения лучшей модели по F1
    exp_dir = str(train_results.save_dir)  # Ultralytics возвращает Path
    best_weights = os.path.join(exp_dir, "weights", "best.pt")
    print(f"Лучшие веса: {best_weights}")
    if os.path.exists(best_weights):
        best_model = YOLO(best_weights)
        
        # Валидация на test выборке
        test_metrics = best_model.val(data=dataset_yaml, split='test', device=DEVICE, batch=BATCH_SIZE, workers=0)
        
        # Метрики на test
        # Преобразуем массивы в числа (берём среднее)
        precision_test = float(test_metrics.box.p.mean()) if hasattr(test_metrics.box.p, 'mean') else float(test_metrics.box.p)
        recall_test = float(test_metrics.box.r.mean()) if hasattr(test_metrics.box.r, 'mean') else float(test_metrics.box.r)
        f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test + 1e-8)
        mAP50_test = float(test_metrics.box.map50)
        mAP5095_test = float(test_metrics.box.map)
        
        # Конвертация в ONNX
        onnx_path = os.path.join(PROJECT_DIR, exp_name, "weights", "best.onnx")
        best_model.export(format="onnx", imgsz=IMG_SIZE, device=DEVICE, simplify=False)
        print(f"✅ Модель экспортирована в ONNX: {onnx_path}")
    else:
        print(f"⚠️ Лучшие веса не найдены для {exp_name}")
        precision_test = recall_test = mAP50_test = mAP5095_test = f1_test = 0.0
    
    return {
        "exp_name": exp_name,
        "dataset": dataset_key,
        "model": model_key,
        "patience": patience,
        "warmup_epochs": warmup_epochs,
        "precision_test": precision_test,  
        "recall_test": recall_test,        
        "f1_test": f1_test,
        "mAP50_test": mAP50_test,
        "mAP5095_test": mAP5095_test,
        "exp_dir": str(exp_dir)
    }

# ==================== ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ (train и val) ====================
# ==================== ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ ====================
def plot_training_metrics(exp_dir, save_path="training_curves.png"):
    """Строит графики для train и val метрик (сетка 3x3)"""
   
    csv_path = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Файл {csv_path} не найден, графики не построены.")
        return
    
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    epochs = df["epoch"].values
    n_epochs = len(epochs)
    
    print(f"📊 Доступные колонки в CSV: {list(df.columns)}")
    
    # === ПРАВИЛЬНЫЕ НАЗВАНИЯ КОЛОНОК ДЛЯ YOLOv26 ===
    # Precision, Recall, mAP50, mAP50-95 — ЭТО УЖЕ VALIDATION метрики!
    # Отдельных train метрик для них НЕТ в CSV
    precision_val = df.get("metrics/precision(B)", None)
    recall_val = df.get("metrics/recall(B)", None)
    map50_val = df.get("metrics/mAP50(B)", None)
    map95_val = df.get("metrics/mAP50-95(B)", None)
    
    # Train метрики (есть только для loss)
    box_loss_train = df.get("train/box_loss", None)
    box_loss_val = df.get("val/box_loss", None)
    cls_loss_train = df.get("train/cls_loss", None)
    cls_loss_val = df.get("val/cls_loss", None)
    dfl_loss_train = df.get("train/dfl_loss", None)
    dfl_loss_val = df.get("val/dfl_loss", None)
    
    # Learning rate
    lr = df.get("lr/pg0", None)  # В YOLOv26 lr/pg0 вместо lr/0
    if lr is None:
        lr = df.get("lr/0", None)
    
    # Вычисляем F1 из precision и recall (только для val, т.к. train нет)
    f1_val = None
    if precision_val is not None and recall_val is not None:
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + 1e-8)
    
    # Так как train метрик для P/R/mAP нет, используем val для обоих (с одинаковым стилем)
    # Но чтобы показать, что train метрик нет — можно просто показывать только val
    
    # Создаём сетку 3x3
    fig, axs = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f"Кривые обучения – {os.path.basename(exp_dir)}\nЭпохи: {n_epochs}", 
                 fontsize=16, fontweight='bold')
    
    # 1. Precision (только val, т.к. train метрики нет в CSV)
    ax = axs[0, 0]
    if precision_val is not None:
        ax.plot(epochs, precision_val, label='Validation', color='blue', linewidth=2)
        ax.set_title("Precision (Validation only)", fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "Precision data missing", ha='center', transform=ax.transAxes)
        ax.set_title("Precision", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.legend() if precision_val is not None else None
    ax.grid(True, alpha=0.3)
    
    # 2. Recall (только val)
    ax = axs[0, 1]
    if recall_val is not None:
        ax.plot(epochs, recall_val, label='Validation', color='green', linewidth=2)
        ax.set_title("Recall (Validation only)", fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "Recall data missing", ha='center', transform=ax.transAxes)
        ax.set_title("Recall", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.legend() if recall_val is not None else None
    ax.grid(True, alpha=0.3)
    
    # 3. F1-score (только val)
    ax = axs[0, 2]
    if f1_val is not None:
        ax.plot(epochs, f1_val, label='Validation', color='purple', linewidth=2)
        ax.set_title("F1-score (Validation only)", fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "F1 data missing", ha='center', transform=ax.transAxes)
        ax.set_title("F1-score", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1-score")
    ax.legend() if f1_val is not None else None
    ax.grid(True, alpha=0.3)
    
    # 4. mAP50 (только val)
    ax = axs[1, 0]
    if map50_val is not None:
        ax.plot(epochs, map50_val, label='Validation', color='red', linewidth=2)
        ax.set_title("mAP50 (Validation only)", fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "mAP50 data missing", ha='center', transform=ax.transAxes)
        ax.set_title("mAP50", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50")
    ax.legend() if map50_val is not None else None
    ax.grid(True, alpha=0.3)
    
    # 5. mAP50-95 (только val)
    ax = axs[1, 1]
    if map95_val is not None:
        ax.plot(epochs, map95_val, label='Validation', color='darkred', linewidth=2)
        ax.set_title("mAP50-95 (Validation only)", fontsize=12, fontweight='bold')
    else:
        ax.text(0.5, 0.5, "mAP50-95 data missing", ha='center', transform=ax.transAxes)
        ax.set_title("mAP50-95", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.legend() if map95_val is not None else None
    ax.grid(True, alpha=0.3)
    
    # 6. Learning Rate
    ax = axs[1, 2]
    if lr is not None:
        lr_data = lr.values if hasattr(lr, 'values') else lr
        if len(lr_data) > 0 and not np.all(np.isnan(lr_data)):
            ax.plot(epochs[:len(lr_data)], lr_data, label='Learning Rate', color='black', linewidth=2)
            ax.set_title("Learning Rate", fontsize=12, fontweight='bold')
            ax.set_yscale('log')
            ax.legend()
        else:
            ax.text(0.5, 0.5, "LR data is empty/NaN", ha='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "LR data missing", ha='center', transform=ax.transAxes)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(True, alpha=0.3)
    
    # 7. Box Loss (Train + Val)
    ax = axs[2, 0]
    if box_loss_train is not None:
        ax.plot(epochs, box_loss_train, label='Train', color='darkblue', linewidth=2)
    if box_loss_val is not None:
        ax.plot(epochs, box_loss_val, label='Val', color='skyblue', linewidth=2, linestyle='--')
    ax.set_title("Box Loss", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if box_loss_train is None and box_loss_val is None:
        ax.text(0.5, 0.5, "Box Loss data missing", ha='center', transform=ax.transAxes)
    
    # 8. Class Loss (Train + Val)
    ax = axs[2, 1]
    if cls_loss_train is not None:
        ax.plot(epochs, cls_loss_train, label='Train', color='brown', linewidth=2)
    if cls_loss_val is not None:
        ax.plot(epochs, cls_loss_val, label='Val', color='gold', linewidth=2, linestyle='--')
    ax.set_title("Class Loss", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if cls_loss_train is None and cls_loss_val is None:
        ax.text(0.5, 0.5, "Class Loss data missing", ha='center', transform=ax.transAxes)
    
    # 9. DFL Loss (Train + Val)
    ax = axs[2, 2]
    if dfl_loss_train is not None or dfl_loss_val is not None:
        if dfl_loss_train is not None:
            ax.plot(epochs, dfl_loss_train, label='Train', color='darkcyan', linewidth=2)
        if dfl_loss_val is not None:
            ax.plot(epochs, dfl_loss_val, label='Val', color='teal', linewidth=2, linestyle='--')
        ax.set_title("DFL Loss", fontsize=12, fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, "DFL Loss\n(not available)", ha='center', transform=ax.transAxes, fontsize=10)
        ax.set_title("DFL Loss", fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"📈 Графики сохранены: {save_path}")
    print(f"   Всего эпох: {n_epochs}")

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def run_train():
    print(f"🔍 Поиск наилучших гиперпараметров на {DEVICE=}")
    print(f"📊 Перебираемые параметры: {GRID_SEARCH}")
    print(f"⚙️ Фиксированные параметры: {EXTRA_PARAMS}")
    print(f"⚠️ Всего экспериментов: {np.prod([len(v) for v in GRID_SEARCH.values()])}")
    
    all_results = []
    
    for values in itertools.product(*GRID_SEARCH.values()):
        params = dict(zip(GRID_SEARCH.keys(), values))
        res = run_experiment(
            dataset_key=params["dataset_key"],
            model_key=params["model_key"],
            patience=params["patience"],
            warmup_epochs=params["warmup_epochs"],
            extra_params=EXTRA_PARAMS
        )
        all_results.append(res)
    
    # Лучшая модель по F1 на test выборке
    best = max(all_results, key=lambda x: x["f1_test"])
    
    print("\n" + "🏆"*30)
    print("ЛУЧШАЯ КОНФИГУРАЦИЯ (по F1 на test выборке):")
    print(f"   Эксперимент : {best['exp_name']}")
    print(f"   Датасет     : {best['dataset']}")
    print(f"   Модель      : {best['model']}")
    print(f"   Patience    : {best['patience']}")
    print(f"   Warmup эпох : {best['warmup_epochs']}")
    print(f"\n📊 Метрики на TEST выборке:")
    print(f"      • Precision : {best['precision_test']:.4f}")
    print(f"      • Recall    : {best['recall_test']:.4f}")
    print(f"      • F1-score  : {best['f1_test']:.4f} ⭐ (критерий выбора)")
    print(f"      • mAP50     : {best['mAP50_test']:.4f}")
    print(f"      • mAP50-95  : {best['mAP5095_test']:.4f}")
    print("🏆"*30)
    
    # Построение графиков для лучшего эксперимента
    onnx_path = os.path.join(best['exp_dir'], "weights", "best.onnx")
    if os.path.exists(onnx_path):
        print(f"\n📦 ONNX модель сохранена: {onnx_path}")
    
    plot_training_metrics(best["exp_dir"], save_path=f"best_training_curves_{best['exp_name']}.png")
    
    # Сохраняем таблицу результатов
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("grid_search_results.csv", index=False)
    print("\n📄 Полная таблица результатов сохранена в grid_search_results.csv")

if __name__ == "__main__":
    run_train()