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
BATCH_SIZE = 8
EPOCHS = 100
PROJECT_DIR = "runs/detect"
DEVICE = 0

# Параметры для перебора (можно расширить)
# GRID_SEARCH = {
#     "dataset_key": ["no_aug", "augmented"],          # 2 варианта
#     "model_key": ["n", "s"],                         # 2 варианта
#     "patience": [5, 7, 10, 15],                      # 4 варианта
#     "warmup_epochs": [0, 3, 5]                      # 3 варианта (0 = без warmup)
# }

GRID_SEARCH = {
    "dataset_key": ["augmented"],          # 2 варианта
    "model_key": ["s"],                         # 2 варианта
    "patience": [5],                      # 4 варианта
    "warmup_epochs": [5]                      # 3 варианта (0 = без warmup)
}

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
    results = model.train(
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
    )
    
    # Валидация на test выборке для определения лучшей модели по F1
    best_weights = os.path.join(PROJECT_DIR, exp_name, "weights", "best.pt")
    if os.path.exists(best_weights):
        best_model = YOLO(best_weights)
        
        # Валидация на test выборке
        test_metrics = best_model.val(data=dataset_yaml, split='test', device=DEVICE, batch=BATCH_SIZE, workers=0)
        
        # Метрики на test
        precision_test = test_metrics.box.p
        recall_test = test_metrics.box.r
        mAP50_test = test_metrics.box.map50
        mAP5095_test = test_metrics.box.map
        
        # Вычисление F1 на test
        f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
        
        # Конвертация в ONNX
        onnx_path = os.path.join(PROJECT_DIR, exp_name, "weights", "best.onnx")
        best_model.export(format="onnx", imgsz=IMG_SIZE, device=DEVICE)
        print(f"✅ Модель экспортирована в ONNX: {onnx_path}")
        
        # Также получим метрики на val для графиков
        val_metrics = best_model.val(data=dataset_yaml, split='val', device=DEVICE, batch=BATCH_SIZE, workers=0)
        precision_val = val_metrics.box.p
        recall_val = val_metrics.box.r
        f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    else:
        print(f"⚠️ Лучшие веса не найдены для {exp_name}")
        precision_test = recall_test = mAP50_test = mAP5095_test = f1_test = 0.0
        precision_val = recall_val = f1_val = 0.0
    
    return {
        "exp_name": exp_name,
        "dataset": dataset_key,
        "model": model_key,
        "patience": patience,
        "warmup_epochs": warmup_epochs,
        "mAP50_test": mAP50_test,
        "mAP5095_test": mAP5095_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "f1_test": f1_test,
        "precision_val": precision_val,
        "recall_val": recall_val,
        "f1_val": f1_val,
        "exp_dir": os.path.join(PROJECT_DIR, exp_name)
    }

# ==================== ФУНКЦИЯ ПОСТРОЕНИЯ ГРАФИКОВ (train и val) ====================
def plot_training_metrics(exp_dir, save_path="training_curves.png"):
    """Строит графики для train и val метрик"""
    csv_path = os.path.join(exp_dir, "results.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Файл {csv_path} не найден, графики не построены.")
        return
    
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    epochs = df["epoch"].values
    
    # Метрики (train и val где возможно)
    metrics_data = {
        "Precision": {
            "train": df.get("metrics/precision(B)", None),
            "val": df.get("metrics/precision(B)", None)  # В YOLO это уже val precision
        },
        "Recall": {
            "train": df.get("metrics/recall(B)", None),
            "val": df.get("metrics/recall(B)", None)
        },
        "F1-score": {},
        "Box Loss": {
            "train": df.get("train/box_loss", None),
            "val": df.get("val/box_loss", None)
        }
    }
    
    # Вычисляем F1 из precision и recall
    if metrics_data["Precision"]["val"] is not None and metrics_data["Recall"]["val"] is not None:
        p_val = metrics_data["Precision"]["val"]
        r_val = metrics_data["Recall"]["val"]
        metrics_data["F1-score"]["val"] = 2 * (p_val * r_val) / (p_val + r_val + 1e-8)
    
    # Learning rate
    lr = df.get("lr/0", None)
    
    # Создаём холст 3x2
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Кривые обучения (Train/Val) – {os.path.basename(exp_dir)}", fontsize=14)
    
    # Precision
    if metrics_data["Precision"]["val"] is not None:
        axs[0,0].plot(epochs, metrics_data["Precision"]["val"], label='Val', color='blue', linewidth=2)
        axs[0,0].set_title("Precision (Validation)")
        axs[0,0].set_xlabel("Epoch")
        axs[0,0].legend()
        axs[0,0].grid(True)
    else:
        axs[0,0].text(0.5, 0.5, "Precision data missing", ha='center')
    
    # Recall
    if metrics_data["Recall"]["val"] is not None:
        axs[0,1].plot(epochs, metrics_data["Recall"]["val"], label='Val', color='green', linewidth=2)
        axs[0,1].set_title("Recall (Validation)")
        axs[0,1].set_xlabel("Epoch")
        axs[0,1].legend()
        axs[0,1].grid(True)
    else:
        axs[0,1].text(0.5, 0.5, "Recall data missing", ha='center')
    
    # F1-score (val)
    if metrics_data["F1-score"]["val"] is not None:
        axs[1,0].plot(epochs, metrics_data["F1-score"]["val"], label='Val', color='purple', linewidth=2)
        axs[1,0].set_title("F1-score (Validation)")
        axs[1,0].set_xlabel("Epoch")
        axs[1,0].legend()
        axs[1,0].grid(True)
    else:
        axs[1,0].text(0.5, 0.5, "F1 data missing", ha='center')
    
    # Box Loss (Train и Val вместе)
    axs[1,1].set_title("Box Loss")
    if metrics_data["Box Loss"]["train"] is not None:
        axs[1,1].plot(epochs, metrics_data["Box Loss"]["train"], label='Train', color='red', linewidth=2)
    if metrics_data["Box Loss"]["val"] is not None:
        axs[1,1].plot(epochs, metrics_data["Box Loss"]["val"], label='Val', color='orange', linewidth=2)
    axs[1,1].set_xlabel("Epoch")
    axs[1,1].legend()
    axs[1,1].grid(True)
    
    # Learning rate
    if lr is not None:
        axs[2,0].plot(epochs, lr, label='Learning rate', color='black', linewidth=2)
        axs[2,0].set_title("Learning Rate")
        axs[2,0].set_xlabel("Epoch")
        axs[2,0].legend()
        axs[2,0].grid(True)
    else:
        axs[2,0].text(0.5, 0.5, "LR data missing", ha='center')
    
    # Убираем пустой subplot
    axs[2,1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📈 Графики сохранены: {save_path}")

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
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
    print(f"\n📊 Метрики на VAL выборке (справочно):")
    print(f"      • Precision : {best['precision_val']:.4f}")
    print(f"      • Recall    : {best['recall_val']:.4f}")
    print(f"      • F1-score  : {best['f1_val']:.4f}")
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
    main()