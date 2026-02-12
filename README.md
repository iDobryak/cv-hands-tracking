# Realtime Face + Hand Tracking (Python Desktop MVP)

Desktop-приложение на Python с realtime-трекингом лица и обеих рук через веб-камеру.

Стек:
- Python 3.11+
- `uv` (менеджер зависимостей и запуск)
- OpenCV (захват видео)
- MediaPipe (face/hand landmarks + handedness)
- PySide6 (GUI, через пакет `pyside6-essentials`)
- pyqtgraph (realtime bar-графики)

## Что реализовано

- Источник видео: дефолтная веб-камера (`camera_index=0`).
- Расширяемая архитектура источников через интерфейс `VideoSource`.
- Детекция:
  - Лицо: ключевые точки нос/глаза/рот + простые флаги мимики (`smile`, `mouth_open`, `brow_raise`).
  - Руки: до двух одновременно, с маркировкой `LEFT`/`RIGHT`.
  - Отрисовка структуры кисти (landmarks + connections).
- GUI:
  - Верхняя часть: видеопоток с оверлеем.
  - Нижняя часть: 2 независимых блока графиков (левая/правая рука).
- Метрики для каждой руки (0..100):
  - `Rot` (поворот ладонь/тыльная сторона)
  - `Thumb`, `Index`, `Middle`, `Ring`, `Pinky` (сгиб пальцев)
- Realtime обновление метрик со сглаживанием (EMA) и деградацией к нулю при потере детекции.
- Потоковая модель без блокировки UI:
  - поток чтения камеры + очередь кадров
  - отдельный QThread для инференса и передачи готовых кадров в UI.

## Структура проекта

```text
main.py
src/
  capture/
    base.py
    camera.py
  vision/
    mediapipe_engine.py
  metrics/
    hand_metrics.py
  ui/
    runtime.py
    main_window.py
tests/
  test_hand_metrics.py
```

## Установка и запуск (UV workflow)

1. Установить зависимости:

```bash
uv sync
```

2. Запустить приложение:

```bash
uv run python main.py
```

3. Запустить тесты:

```bash
uv run pytest
```

## Примечания по производительности

- Целевой FPS зависит от камеры/CPU/GPU и разрешения.
- По умолчанию поток камеры и инференс разделены, чтобы UI оставался отзывчивым.
- При необходимости можно снизить разрешение в `CameraSource` (`src/capture/camera.py`) для роста FPS.
