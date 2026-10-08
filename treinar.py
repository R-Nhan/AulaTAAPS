"""Treina um detector YOLOv8 com um dataset no formato YOLO.

Exemplos:
    python treinar.py --fast
    python treinar.py --epochs 100 --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
DATA_YAML = ROOT / "data.yaml"


def criar_estrutura() -> None:
    """Cria as pastas esperadas pelo Ultralytics."""
    for split in ("train", "val", "test"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        DATA_YAML.write_text(
            "path: dataset\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            "names:\n"
            "  0: objeto\n",
            encoding="utf-8",
        )


def contar_arquivos(pasta: Path, extensoes: tuple[str, ...]) -> int:
    return sum(
        1
        for arquivo in pasta.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes
    )


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
    parser = argparse.ArgumentParser(
        description="Treina um modelo YOLOv8 usando dataset no formato YOLO."
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--name", default="meu_modelo")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, mps (Apple Silicon), cpu ou 0 (CUDA).",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Mantém as imagens em memória; acelera, mas usa mais RAM.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Modo rápido: 30 épocas e imagens 416x416.",
    )
    args = parser.parse_args()

    if args.fast:
        args.epochs = 30
        args.imgsz = 416
        args.patience = min(args.patience, 10)

    criar_estrutura()

    imagens_treino = contar_arquivos(
        DATASET_DIR / "images" / "train", (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
    imagens_validacao = contar_arquivos(
        DATASET_DIR / "images" / "val", (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
    if imagens_treino == 0:
        raise SystemExit(
            "Coloque imagens em dataset/images/train e dataset/images/val "
            "antes de treinar."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Dependência ausente. Instale com: "
            "python -m pip install -r requirements.txt"
        ) from exc

    print(f"Imagens de treino: {imagens_treino}")
    print(f"Imagens de validação: {imagens_validacao}")
    print(f"Dispositivo: {escolher_dispositivo(args.device)}")
    print("Iniciando treinamento...")

    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=escolher_dispositivo(args.device),
        workers=args.workers,
        cache=args.cache,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
    )

    print(f"Treinamento concluído. Pesos: {ROOT / 'runs' / args.name / 'weights'}")


if __name__ == "__main__":
    main()