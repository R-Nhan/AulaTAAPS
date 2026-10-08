from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps


PASTA_FOTOS = Path(__file__).resolve().parent
EXTENSOES_DE_IMAGEM = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
}


def converter_para_png(origem: Path, destino: Path) -> None:
    arquivo_temporario = destino.with_name(f".__convertendo_{uuid4().hex}.png")

    try:
        with Image.open(origem) as imagem:
            imagem.seek(0)  # Em GIFs animados, converte o primeiro quadro.
            imagem = ImageOps.exif_transpose(imagem)

            # PNG nao aceita alguns modos, como CMYK. Preserva transparencia.
            if "A" in imagem.getbands() or "transparency" in imagem.info:
                imagem = imagem.convert("RGBA")
            else:
                imagem = imagem.convert("RGB")

            imagem.save(arquivo_temporario, format="PNG")

        arquivo_temporario.replace(destino)
    finally:
        arquivo_temporario.unlink(missing_ok=True)


def renomear_fotos() -> None:
    fotos = sorted(
        (
            arquivo
            for arquivo in PASTA_FOTOS.iterdir()
            if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_DE_IMAGEM
        ),
        key=lambda arquivo: arquivo.name.lower(),
    )

    if not fotos:
        print("Nenhuma foto encontrada na pasta.")
        return

    # Primeiro usa nomes temporarios para evitar conflitos com img1, img2 etc.
    temporarios: list[tuple[Path, str]] = []
    for foto in fotos:
        temporario = foto.with_name(f".__renomeando_{uuid4().hex}{foto.suffix}")
        foto.rename(temporario)
        temporarios.append((temporario, foto.name))

    convertidas = 0
    for numero, (temporario, nome_anterior) in enumerate(temporarios, start=1):
        destino = PASTA_FOTOS / f"img{numero}.png"

        try:
            converter_para_png(temporario, destino)
        except Exception as erro:
            # Devolve a foto original caso ela nao possa ser convertida.
            temporario.rename(PASTA_FOTOS / nome_anterior)
            print(f"Erro ao converter {nome_anterior}: {erro}")
            continue

        temporario.unlink()
        convertidas += 1
        print(f"{nome_anterior} -> {destino.name}")

    print(f"\n{convertidas} foto(s) convertida(s) para PNG.")


if __name__ == "__main__":
    renomear_fotos()
