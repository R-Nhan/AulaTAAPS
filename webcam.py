"""Executa detecção YOLOv8 em tempo real usando a webcam."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent


def escolher_dispositivo(preferencia: str) -> str:
    if preferencia != "auto":
        if preferencia == "mps":
            try:
                import torch
            except ImportError as exc:
                raise SystemExit(
                    "Para usar MPS, instale PyTorch ou use --device cpu."
                ) from exc
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise SystemExit(
                    "MPS não está disponível neste computador. "
                    "Use --device auto, cpu ou 0."
                )
        if preferencia.isdigit():
            try:
                import torch
            except ImportError as exc:
                raise SystemExit(
                    "Para usar CUDA, instale PyTorch com suporte à NVIDIA."
                ) from exc
            if not torch.cuda.is_available():
                raise SystemExit(
                    "CUDA não está disponível. Use --device auto ou --device cpu."
                )
        return preferencia

    try:
        import torch
    except ImportError:
        return "cpu"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Detecta objetos pela webcam.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "runs" / "meu_modelo" / "weights" / "best.pt"),
        help="Caminho para os pesos .pt treinados.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, mps (Apple Silicon), cpu ou 0 (CUDA/NVIDIA).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Tamanho usado na inferência. Valores menores deixam a webcam mais rápida.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=2,
        help="Processa 1 a cada N frames; use 1 para processar todos.",
    )
    args = parser.parse_args()

    if args.skip < 1:
        raise SystemExit("--skip deve ser maior ou igual a 1.")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Modelo não encontrado: {model_path}\n"
            "Execute primeiro: python treinar.py"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Dependência ausente. Instale com: "
            "python -m pip install -r requirements.txt"
        ) from exc

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(
            f"Não foi possível abrir a câmera {args.camera}. "
            "Verifique a permissão da câmera e tente --camera 1."
        )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    model = YOLO(str(model_path))
    device = escolher_dispositivo(args.device)
    print(f"Modelo: {model_path}")
    print(
        f"Classes: {model.names} | confiança mínima: {args.conf} | "
        f"tamanho: {args.imgsz} | dispositivo: {device}"
    )
    print("Webcam iniciada. Pressione Q ou ESC para sair.")

    try:
        frame_number = 0
        annotated = None
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Não foi possível ler um frame da webcam.")
                break

            if frame_number % args.skip == 0 or annotated is None:
                result = model.predict(
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=device,
                    verbose=False,
                )[0]
                annotated = result.plot()

            cv2.imshow("YOLOv8 - Webcam", annotated)
            frame_number += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
