# Detector de objetos com YOLOv8

Projeto de detecção de objetos usando YOLOv8, com treinamento personalizado e
detecção em tempo real pela webcam.

## Requisitos

- Python 3.9 ou superior.
- Webcam para executar `webcam.py`.
- Git é opcional, caso o projeto seja clonado de um repositório.

Os comandos abaixo devem ser executados dentro da pasta do projeto.

## Linux

### 1. Criar e ativar a venv

```bash
cd /caminho/para/seminario
python3 -m venv .venv
source .venv/bin/activate
```

Se o comando `venv` não estiver disponível no Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3-venv
```

### 2. Instalar as dependências

Com a venv ativada:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o Linux apresentar erro relacionado ao OpenCV e à biblioteca `libGL`, instale:

```bash
sudo apt install libgl1
```

### 3. Executar o treinamento

```bash
python treinar.py --fast
```

Para um treinamento mais completo:

```bash
python treinar.py --epochs 100 --imgsz 512
```

Em uma máquina Linux com GPU NVIDIA e PyTorch configurado para CUDA:

```bash
python treinar.py --device 0
```

### 4. Executar a webcam

```bash
python webcam.py
```

Se houver mais de uma câmera:

```bash
python webcam.py --camera 1
```

Para sair, pressione `Q` ou `ESC`.

## Windows

Abra o PowerShell ou o Prompt de Comando na pasta do projeto.

### 1. Criar e ativar a venv

No PowerShell:

```powershell
cd C:\caminho\para\seminario
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Prompt de Comando (`cmd`):

```bat
cd C:\caminho\para\seminario
py -m venv .venv
.venv\Scripts\activate.bat
```

Se o PowerShell bloquear a ativação por causa da política de execução, execute
uma vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Depois feche e abra o PowerShell novamente e ative a venv.

### 2. Instalar as dependências

Com a venv ativada:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Executar o treinamento

```powershell
python treinar.py --fast
```

Para um treinamento mais completo:

```powershell
python treinar.py --epochs 100 --imgsz 512
```

Em um computador Windows com GPU NVIDIA e PyTorch configurado para CUDA:

```powershell
python treinar.py --device 0
```

### 4. Executar a webcam

```powershell
python webcam.py
```

Se houver mais de uma câmera:

```powershell
python webcam.py --camera 1
```

Para sair, pressione `Q` ou `ESC`.

## macOS

O projeto também pode ser executado no macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python treinar.py --device auto
python webcam.py --device auto
```

Em Macs com Apple Silicon, o script usa `mps` automaticamente quando essa
aceleração estiver disponível.

## Colocando as imagens do dataset

As imagens e seus labels devem ter o mesmo nome:

```text
dataset/
├── images/
│   ├── train/
│   │   └── foto01.jpg
│   └── val/
│       └── foto02.jpg
└── labels/
    ├── train/
    │   └── foto01.txt
    └── val/
        └── foto02.txt
```

Cada arquivo `.txt` deve usar o formato YOLO:

```text
classe centro_x centro_y largura altura
```

Os valores de posição devem estar normalizados entre `0` e `1`. Use uma
ferramenta de anotação como LabelImg, CVAT ou Roboflow para criar os labels.

As classes são configuradas em [data.yaml](data.yaml). Por exemplo:

```yaml
names:
  0: objeto
```

Não coloque a mesma foto em `train` e `val`. Use fotos variadas, com diferentes
ângulos, distâncias e iluminações.

## Arquivos gerados

Depois do treinamento, o melhor modelo fica em:

```text
runs/meu_modelo/weights/best.pt
```

Esse modelo é carregado automaticamente pelo `webcam.py`.

Para usar outro arquivo de pesos:

```bash
python webcam.py --model caminho/para/best.pt
```

## Verificar o dispositivo disponível

O argumento `--device auto` escolhe automaticamente:

- `mps` em Macs Apple Silicon;
- `0` em máquinas com CUDA/NVIDIA disponível;
- `cpu` nos demais casos.

Para conferir CUDA no ambiente Python:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Se o resultado for `True`, é possível usar:

```bash
python treinar.py --device 0
```

## Desativar a venv

Quando terminar:

```bash
deactivate
```
