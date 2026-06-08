"""
ONNX Object Detection Inference
================================
Поддержка: изображения (.jpg, .png, и др.) и видео (.mp4, .avi, и др.)
Тип модели: детекция объектов (bounding boxes)

Установка зависимостей:
    pip install onnxruntime opencv-python numpy

Запуск:
    # Изображение
    python onnx_detector.py --model best.onnx --input hand.mp4 --imgsz 960

    # Видео
    python onnx_detector.py --model best.onnx --input hand.png --imgsz 960

    # Камера
    python onnx_detector.py --model best.onnx --input 0 --imgsz 960

    # С дополнительными параметрами
    python onnx_detector.py --model model.onnx --input image.jpg \
        --conf 0.5 --iou 0.45 --imgsz 640 \
        --labels labels.txt --output result.jpg
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(path: str | None) -> list[str]:
    """Загружает список меток классов из текстового файла (по одной на строку)."""
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    # Резервные метки — нумерация классов
    return [str(i) for i in range(1000)]


def get_colors(n: int) -> list[tuple[int, int, int]]:
    """Генерирует n визуально различимых BGR-цветов."""
    np.random.seed(42)
    hsv = np.stack(
        [np.linspace(0, 179, n, dtype=np.uint8),
         np.full(n, 220, dtype=np.uint8),
         np.full(n, 220, dtype=np.uint8)],
        axis=1,
    ).reshape(n, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)
    return [tuple(int(c) for c in row) for row in bgr]


# ─────────────────────────────────────────────────────────────────────────────
# Препроцессинг
# ─────────────────────────────────────────────────────────────────────────────

def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Масштабирует изображение с сохранением пропорций и добавляет padding.
    Возвращает: (изображение, коэффициент масштабирования, (pad_x, pad_y))
    """
    h, w = img.shape[:2]
    nh, nw = new_shape

    scale = min(nw / w, nh / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))

    dw = (nw - rw) / 2
    dh = (nh - rh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, scale, (left, top)


def preprocess(
    frame: np.ndarray,
    imgsz: int = 640,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """BGR-кадр → NCHW float32 тензор [0,1] для ONNX."""
    img, scale, pad = letterbox(frame, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))          # HWC → CHW
    img = np.expand_dims(img, axis=0)            # CHW → NCHW
    return img, scale, pad


# ─────────────────────────────────────────────────────────────────────────────
# Постпроцессинг
# ─────────────────────────────────────────────────────────────────────────────

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """cx,cy,w,h → x1,y1,x2,y2"""
    out = boxes.copy()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    """Non-Maximum Suppression (чистый NumPy)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]

    return keep


def postprocess(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    scale: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    end2end: bool = False,
) -> list[dict]:
    """
    Обрабатывает сырой выход ONNX и возвращает список детекций.

    Поддерживаются три формата выхода:
      • end2end=True: [1, num_det, 6] — (cx,cy,w,h, conf, class_id), NMS уже встроен
      • [1, num_det, 5+num_classes]   — YOLO-стиль (cx,cy,w,h,obj, cls…), NMS применяется
      • [1, 5+num_classes, num_det]   — транспонированный вариант, NMS применяется

    Каждая детекция: {'box': [x1,y1,x2,y2], 'score': float, 'class_id': int}
    """
    pred = output[0]  # (N, 6) или (N, 5+C) или (5+C, N)

    # ── end2end / NMS-free формат: (N, 6) = cx,cy,w,h,conf,class_id ───────
    if end2end:
        # Нормализуем к (N, 6)
        if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
            pred = pred.T

        confidences = pred[:, 4]
        class_ids = pred[:, 5].astype(int)
        mask = confidences >= conf_threshold
        if not mask.any():
            return []

        pred = pred[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # cx,cy,w,h → x1,y1,x2,y2
        boxes_raw = xywh2xyxy(pred[:, :4].copy())

        pad_x, pad_y = pad
        boxes_raw[:, [0, 2]] = (boxes_raw[:, [0, 2]] - pad_x) / scale
        boxes_raw[:, [1, 3]] = (boxes_raw[:, [1, 3]] - pad_y) / scale

        orig_h, orig_w = orig_shape
        boxes_raw[:, [0, 2]] = boxes_raw[:, [0, 2]].clip(0, orig_w)
        boxes_raw[:, [1, 3]] = boxes_raw[:, [1, 3]].clip(0, orig_h)

        # NMS уже применён моделью — просто возвращаем
        detections = []
        for i in range(len(boxes_raw)):
            x1, y1, x2, y2 = boxes_raw[i].astype(int)
            detections.append({
                "box": [x1, y1, x2, y2],
                "score": float(confidences[i]),
                "class_id": int(class_ids[i]),
            })
        return detections

    # ── Стандартный YOLO-формат ─────────────────────────────────────────────
    # Нормализуем к форме (N, 5+C)
    if pred.ndim == 2 and pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.ndim != 2 or pred.shape[1] < 5:
        return []

    num_classes = pred.shape[1] - 5

    # ── Фильтрация по уверенности ──────────────────────────────────────────
    if num_classes > 0:
        objectness = pred[:, 4]
        class_scores = pred[:, 5:]
        scores = objectness[:, None] * class_scores          # (N, C)
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
    else:
        # 6-колоночный формат (x1,y1,x2,y2,conf,cls)
        confidences = pred[:, 4]
        class_ids = pred[:, 5].astype(int)

    mask = confidences >= conf_threshold
    if not mask.any():
        return []

    pred = pred[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # ── Координаты → оригинальный размер ───────────────────────────────────
    boxes_raw = pred[:, :4].copy()

    # Если модель выдаёт cx,cy,w,h — конвертируем
    if num_classes > 0:
        boxes_raw = xywh2xyxy(boxes_raw)

    pad_x, pad_y = pad
    boxes_raw[:, [0, 2]] = (boxes_raw[:, [0, 2]] - pad_x) / scale
    boxes_raw[:, [1, 3]] = (boxes_raw[:, [1, 3]] - pad_y) / scale

    orig_h, orig_w = orig_shape
    boxes_raw[:, [0, 2]] = boxes_raw[:, [0, 2]].clip(0, orig_w)
    boxes_raw[:, [1, 3]] = boxes_raw[:, [1, 3]].clip(0, orig_h)

    # ── NMS ────────────────────────────────────────────────────────────────
    keep = nms(boxes_raw, confidences, iou_threshold)

    detections = []
    for idx in keep:
        x1, y1, x2, y2 = boxes_raw[idx].astype(int)
        detections.append({
            "box": [x1, y1, x2, y2],
            "score": float(confidences[idx]),
            "class_id": int(class_ids[idx]),
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Визуализация
# ─────────────────────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    labels: list[str],
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    """Рисует bounding boxes и подписи на кадре."""
    vis = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        score = det["score"]

        color = colors[cls_id % len(colors)]
        label = labels[cls_id] if cls_id < len(labels) else str(cls_id)
        text = f"{label} {score:.2f}"

        # Рамка
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Фон подписи
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(vis, (x1, ty - th - baseline - 2), (x1 + tw + 2, ty + 2), color, -1)

        # Текст
        cv2.putText(
            vis, text, (x1 + 1, ty - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return vis


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Отображает FPS в левом верхнем углу."""
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
    )
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Инференс
# ─────────────────────────────────────────────────────────────────────────────

class ONNXDetector:
    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        conf: float = 0.5,
        iou: float = 0.45,
        labels: list[str] | None = None,
        device: str = "cpu",
    ):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf
        self.iou = iou

        # ── Читаем метаданные модели (Ultralytics ONNX) ───────────────────
        import ast
        meta = dict(self.session.get_modelmeta().custom_metadata_map)
        
        # end2end: NMS уже встроен в модель
        self.end2end = meta.get("end2end", "False").lower() in ("true", "1", "yes")

        # imgsz из метаданных имеет приоритет над аргументом
        if "imgsz" in meta:
            try:
                sz = ast.literal_eval(meta["imgsz"])
                imgsz = sz[0] if isinstance(sz, (list, tuple)) else int(sz)
            except Exception:
                pass
        self.imgsz = imgsz

        # Имена классов из метаданных (если не переданы явно)
        if not labels and "names" in meta:
            try:
                names = ast.literal_eval(meta["names"])
                if isinstance(names, dict):
                    labels = [names[i] for i in sorted(names)]
                elif isinstance(names, list):
                    labels = names
            except Exception:
                pass

        self.labels = labels or [str(i) for i in range(1000)]
        self.colors = get_colors(max(len(self.labels), 1))

        inp = self.session.get_inputs()[0]
        print(f"[INFO] Модель загружена: {Path(model_path).name}")
        print(f"[INFO] Вход: {inp.name}  форма: {inp.shape}  тип: {inp.type}")
        print(f"[INFO] Провайдеры: {self.session.get_providers()}")
        print(f"[INFO] imgsz={self.imgsz}  end2end={self.end2end}  классы: {self.labels}")

    def predict(self, frame: np.ndarray) -> list[dict]:
        """Выполняет инференс одного кадра (BGR numpy array)."""
        tensor, scale, pad = preprocess(frame, self.imgsz)
        outputs = self.session.run(None, {self.input_name: tensor})
        return postprocess(
            outputs[0], self.conf, self.iou, scale, pad, frame.shape[:2],
            end2end=self.end2end,
        )

    def visualize(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        return draw_detections(frame, detections, self.labels, self.colors)


# ─────────────────────────────────────────────────────────────────────────────
# Обработка файлов
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


def letterbox_image(img: np.ndarray, target_w: int, target_h: int,
                    color: tuple = (0, 0, 0)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def imshow_adaptive(win_name: str, img: np.ndarray) -> None:
    target_w, target_h = WINDOW_WIDTH, WINDOW_HEIGHT
    try:
        rect = cv2.getWindowImageRect(win_name)
        if rect[2] > 0 and rect[3] > 0:
            target_w, target_h = rect[2], rect[3]
    except:
        pass
    img = letterbox_image(img, target_w, target_h)
    cv2.imshow(win_name, img)


def process_image(detector: ONNXDetector, input_path: str, output_path: str | None):
    frame = cv2.imread(input_path)
    if frame is None:
        raise ValueError(f"Не удалось открыть изображение: {input_path}")

    t0 = time.perf_counter()
    detections = detector.predict(frame)
    elapsed = time.perf_counter() - t0

    vis = detector.visualize(frame, detections)

    print(f"\n[INFO] Изображение: {input_path}")
    print(f"[INFO] Время инференса: {elapsed * 1000:.1f} мс")
    print(f"[INFO] Обнаружено объектов: {len(detections)}")
    for i, d in enumerate(detections):
        label = detector.labels[d['class_id']] if d['class_id'] < len(detector.labels) else d['class_id']
        print(f"  [{i+1}] {label:20s}  уверенность: {d['score']:.3f}  box: {d['box']}")

    if output_path:
        cv2.imwrite(output_path, vis)
        print(f"[INFO] Результат сохранён: {output_path}")
    else:
        cv2.namedWindow("ONNX Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ONNX Detection", WINDOW_WIDTH, WINDOW_HEIGHT)
        imshow_adaptive("ONNX Detection", vis)
        print("[INFO] Нажмите любую клавишу или закройте окно для выхода...")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key != 0xFF:
                break
            if cv2.getWindowProperty("ONNX Detection", cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


def process_video(detector: ONNXDetector, input_path: str, output_path: str | None,
                  output_size: tuple[int, int] | None = None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {input_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Размер выходного видео: кастомный или оригинальный
    out_w, out_h = output_size if output_size else (width, height)

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    frame_idx = 0
    times = []

    print(f"\n[INFO] Видео: {input_path}  ({width}×{height}, {fps:.1f} fps, ~{total} кадров)")
    if output_size:
        print(f"[INFO] Размер выходного видео: {out_w}×{out_h}")

    if not output_path:
        cv2.namedWindow("ONNX Detection  (q — выход)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ONNX Detection  (q — выход)", WINDOW_WIDTH, WINDOW_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            detections = detector.predict(frame)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            cur_fps = 1.0 / (sum(times[-30:]) / min(len(times), 30))
            vis = detector.visualize(frame, detections)
            vis = draw_fps(vis, cur_fps)

            if output_size:
                vis = cv2.resize(vis, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            if writer:
                writer.write(vis)
            else:
                imshow_adaptive("ONNX Detection  (q — выход)", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if cv2.getWindowProperty("ONNX Detection  (q — выход)", cv2.WND_PROP_VISIBLE) < 1:
                    break

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"  кадр {frame_idx}/{total}  FPS: {cur_fps:.1f}  "
                      f"объектов: {len(detections)}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    avg_ms = sum(times) / len(times) * 1000 if times else 0
    print(f"\n[INFO] Обработано кадров: {frame_idx}")
    print(f"[INFO] Среднее время на кадр: {avg_ms:.1f} мс  ({1000/avg_ms:.1f} FPS)")
    if output_path:
        print(f"[INFO] Результат сохранён: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ONNX Object Detection Inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model",  required=True, help="Путь к .onnx файлу")
    p.add_argument("--input",  required=True, help="Путь к изображению или видео")
    p.add_argument("--output", default=None,  help="Путь для сохранения результата (опционально)")
    p.add_argument("--labels", default=None,  help="Файл с именами классов (по одному на строку)")
    p.add_argument("--conf",   type=float, default=0.5,  help="Порог уверенности (default: 0.5)")
    p.add_argument("--iou",    type=float, default=0.45, help="Порог NMS IoU (default: 0.45)")
    p.add_argument("--imgsz",  type=int,   default=640,  help="Размер входа модели (default: 640)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="Устройство: cpu | cuda (default: cpu)")
    p.add_argument("--output-size", default=None, metavar="WxH",
                   help="Разрешение выходного видео, например: 1024x640")
    return p.parse_args()


def main():
    args = parse_args()

    labels = load_labels(args.labels)
    detector = ONNXDetector(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        labels=labels,
        device=args.device,
    )

    ext = Path(args.input).suffix.lower()

    output_size = None
    if args.output_size:
        try:
            ow, oh = args.output_size.lower().split("x")
            output_size = (int(ow), int(oh))
        except ValueError:
            print("[WARN] Неверный формат --output-size, ожидается WxH (например, 1024x640). Игнорируется.")

    if ext in IMAGE_EXTS:
        process_image(detector, args.input, args.output)
    elif ext in VIDEO_EXTS:
        process_video(detector, args.input, args.output, output_size)
    else:
        try:
            source = int(args.input)
        except ValueError:
            source = args.input
        process_video(detector, source, args.output, output_size)


if __name__ == "__main__":
    main()