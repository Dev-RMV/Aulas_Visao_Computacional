#!/usr/bin/env python3
"""
preparar_ambiente.py

Prepara o ambiente do projeto no Windows ou no Linux (inclusive WSL), sem
privilégios de administrador:

1. Detecta o sistema operacional.
2. Localiza um Python 3.13. Se não houver, instala o gerenciador `uv` na pasta
   do usuário e usa ele para baixar um Python 3.13 isolado.
3. Cria o ambiente virtual (`.venv` no Windows, `.venv_linux` no Linux, porque
   a mesma pasta pode ser vista pelos dois sistemas via /mnt/c).
4. Instala `requirements.txt` e, com --no-deps, `requirements_face_recognition.txt`
   e `requirements_ultralytics.txt`.
5. No Linux, confere as bibliotecas de sistema que o OpenCV exige.
6. Roda uma verificação: versões, Keras com backend TensorFlow, OpenCV contrib,
   dlib e face_recognition, PyTorch com torchvision, ultralytics.

Uso:
    Windows:  py preparar_ambiente.py
    Linux:    python3 preparar_ambiente.py

Opções:
    --recriar           apaga o venv antes de criar de novo
    --sem-verificacao   pula o passo 6
    --venv-dir PASTA    usa outra pasta para o venv
    --python CAMINHO    usa esse interpretador 3.13 em vez de procurar

Só usa a biblioteca padrão e roda em qualquer Python 3.8 ou mais novo.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
VERSAO_PYTHON = "3.13"
REQUISITOS = RAIZ / "requirements.txt"
REQUISITOS_FACE = RAIZ / "requirements_face_recognition.txt"
REQUISITOS_ULTRA = RAIZ / "requirements_ultralytics.txt"
URL_UV_LINUX = "https://astral.sh/uv/install.sh"
URL_UV_WINDOWS = "https://astral.sh/uv/install.ps1"


# ----------------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------------

def titulo(texto: str) -> None:
    print()
    print("=" * 72)
    print(texto)
    print("=" * 72)


def passo(texto: str) -> None:
    print()
    print(f"--> {texto}")


def rodar(cmd: list[str], **kw) -> None:
    """Executa mostrando o comando. Aborta em erro."""
    print("$ " + " ".join(str(c) for c in cmd))
    sys.stdout.flush()
    subprocess.run(cmd, check=True, **kw)


def saida(cmd: list[str], **kw) -> str:
    """Executa e devolve stdout como texto. Devolve '' em erro."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)
        return r.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def versao_de(python_exe: str) -> str:
    return saida([python_exe, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"])


def caminho_de(python_exe_ou_cmd: list[str]) -> str:
    return saida(python_exe_ou_cmd + ["-c", "import sys; print(sys.executable)"])


# ----------------------------------------------------------------------------
# 1. sistema operacional
# ----------------------------------------------------------------------------

def detectar_sistema() -> tuple[str, bool]:
    """Devolve ('windows' | 'linux', eh_wsl)."""
    nome = platform.system()
    if nome == "Windows":
        return "windows", False
    if nome == "Linux":
        eh_wsl = False
        try:
            eh_wsl = "microsoft" in Path("/proc/version").read_text().lower()
        except OSError:
            pass
        return "linux", eh_wsl
    sys.exit(f"Sistema não suportado por este script: {nome}. Só Windows e Linux.")


# ----------------------------------------------------------------------------
# 2. Python 3.13
# ----------------------------------------------------------------------------

def achar_python(sistema: str) -> str | None:
    """Procura um Python 3.13 já instalado. Devolve o caminho do executável."""
    candidatos: list[list[str]] = []
    if sistema == "windows":
        candidatos += [["py", f"-{VERSAO_PYTHON}"], [f"python{VERSAO_PYTHON}"], ["python3"], ["python"]]
    else:
        candidatos += [[f"python{VERSAO_PYTHON}"], ["python3"], ["python"]]
    # o próprio interpretador que está rodando este script também conta
    candidatos.append([sys.executable])

    for cmd in candidatos:
        if cmd[0] != sys.executable and shutil.which(cmd[0]) is None:
            continue
        exe = caminho_de(cmd)
        if exe and versao_de(exe) == VERSAO_PYTHON:
            return exe
    return None


def achar_uv(sistema: str) -> str | None:
    achado = shutil.which("uv")
    if achado:
        return achado
    home = Path.home()
    locais = [home / ".local" / "bin" / ("uv.exe" if sistema == "windows" else "uv")]
    if sistema == "windows":
        locais.append(Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "uv.exe")
    for p in locais:
        if p.is_file():
            return str(p)
    return None


def instalar_uv(sistema: str) -> str:
    passo("Instalando o gerenciador uv na pasta do usuário (não exige administrador)")
    if sistema == "windows":
        rodar(["powershell", "-NoProfile", "-ExecutionPolicy", "ByPass", "-c",
               f"irm {URL_UV_WINDOWS} | iex"])
    else:
        if shutil.which("curl"):
            rodar(["sh", "-c", f"curl -LsSf {URL_UV_LINUX} | sh"])
        elif shutil.which("wget"):
            rodar(["sh", "-c", f"wget -qO- {URL_UV_LINUX} | sh"])
        else:
            sys.exit("Nem curl nem wget disponíveis para baixar o uv. Instale um deles ou um Python 3.13.")
    uv = achar_uv(sistema)
    if not uv:
        sys.exit("O uv foi instalado mas não foi encontrado. Abra um terminal novo e rode o script de novo.")
    return uv


def python_via_uv(sistema: str) -> str:
    uv = achar_uv(sistema) or instalar_uv(sistema)
    passo(f"Baixando um Python {VERSAO_PYTHON} isolado com o uv (fica em ~/.local/share/uv)")
    rodar([uv, "python", "install", VERSAO_PYTHON])
    exe = saida([uv, "python", "find", VERSAO_PYTHON])
    if not exe or versao_de(exe) != VERSAO_PYTHON:
        sys.exit(f"O uv não conseguiu fornecer um Python {VERSAO_PYTHON}.")
    return exe


# ----------------------------------------------------------------------------
# 3. venv
# ----------------------------------------------------------------------------

def python_do_venv(venv: Path, sistema: str) -> Path:
    if sistema == "windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def criar_venv(python_exe: str, venv: Path, sistema: str, recriar: bool) -> Path:
    py_venv = python_do_venv(venv, sistema)
    if venv.exists() and recriar:
        passo(f"Apagando o venv antigo em {venv}")
        shutil.rmtree(venv)
    if py_venv.exists():
        ver = versao_de(str(py_venv))
        if ver == VERSAO_PYTHON:
            passo(f"Venv já existe em {venv} (Python {ver}). Reaproveitando.")
            return py_venv
        passo(f"Venv em {venv} usa Python {ver}, não {VERSAO_PYTHON}. Recriando.")
        shutil.rmtree(venv)
    passo(f"Criando o venv em {venv}")
    try:
        rodar([python_exe, "-m", "venv", str(venv)])
    except subprocess.CalledProcessError:
        if sistema == "linux":
            sys.exit(
                "Falha ao criar o venv. Em Debian e Ubuntu o módulo venv vem separado:\n"
                f"    sudo apt-get install python{VERSAO_PYTHON}-venv\n"
                "Ou rode de novo sem esse Python no PATH, para o script usar o uv."
            )
        raise
    return py_venv


# ----------------------------------------------------------------------------
# 4. pacotes
# ----------------------------------------------------------------------------

def instalar_pacotes(py_venv: Path) -> None:
    for arq in (REQUISITOS, REQUISITOS_FACE, REQUISITOS_ULTRA):
        if not arq.is_file():
            sys.exit(f"Arquivo não encontrado: {arq}")
    passo("Atualizando o pip")
    rodar([str(py_venv), "-m", "pip", "install", "--upgrade", "pip", "-q"])
    passo(f"Instalando {REQUISITOS.name} (TensorFlow tem cerca de 350 MB e o PyTorch cerca de 250 MB, pode demorar)")
    rodar([str(py_venv), "-m", "pip", "install", "-r", str(REQUISITOS)])
    passo(f"Instalando {REQUISITOS_FACE.name} com --no-deps (a dependência dlib é atendida pelo dlib-bin)")
    rodar([str(py_venv), "-m", "pip", "install", "--no-deps", "-r", str(REQUISITOS_FACE)])
    passo(f"Instalando {REQUISITOS_ULTRA.name} com --no-deps (o ultralytics declara opencv-python, mas o cv2 vem do contrib)")
    rodar([str(py_venv), "-m", "pip", "install", "--no-deps", "-r", str(REQUISITOS_ULTRA)])


# ----------------------------------------------------------------------------
# 5. bibliotecas de sistema (Linux)
# ----------------------------------------------------------------------------

def checar_libs_linux() -> None:
    passo("Conferindo bibliotecas de sistema que o OpenCV exige")
    if not shutil.which("ldconfig"):
        print("ldconfig não encontrado, pulando a checagem.")
        return
    tabela = saida(["ldconfig", "-p"])
    faltando = []
    for lib, pacote in (("libGL.so.1", "libgl1"), ("libglib-2.0.so.0", "libglib2.0-0")):
        if lib in tabela:
            print(f"ok      {lib}")
        else:
            print(f"FALTA   {lib}  (pacote {pacote})")
            faltando.append(pacote)
    if faltando:
        print("Instale com privilégios de administrador e rode o script de novo:")
        print("    sudo apt-get install -y " + " ".join(faltando))


# ----------------------------------------------------------------------------
# 6. verificação
# ----------------------------------------------------------------------------

CODIGO_VERIFICACAO = r'''
import os, sys, warnings, platform
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    import absl.logging; absl.logging.set_verbosity(absl.logging.ERROR)
except Exception: pass
falhas = []
def linha(nome, valor): print(f"{nome:<22} {valor}")
linha("python", platform.python_version() + "  " + sys.executable)
try:
    import numpy, pandas, matplotlib, sklearn, joblib, scipy
    linha("numpy", numpy.__version__); linha("pandas", pandas.__version__)
    linha("matplotlib", matplotlib.__version__); linha("scikit-learn", sklearn.__version__)
    linha("joblib", joblib.__version__); linha("scipy", scipy.__version__)
except Exception as e: falhas.append(f"numérico: {e}")
try:
    import cv2
    linha("opencv", f"{cv2.__version__}  face={hasattr(cv2, 'face')} dnn={hasattr(cv2, 'dnn')} SIFT={hasattr(cv2, 'SIFT_create')}")
    cv2.face.LBPHFaceRecognizer_create()
    hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
except Exception as e: falhas.append(f"opencv: {e}")
try:
    import keras, tensorflow as tf
    from keras import ops
    linha("tensorflow", tf.__version__)
    linha("keras", f"{keras.__version__}  backend={keras.backend.backend()}")
    if keras.backend.backend() != "tensorflow": falhas.append("backend do Keras não é tensorflow")
    import numpy as np
    m = keras.Sequential([keras.Input((4,)), keras.layers.Dense(2, activation="softmax")])
    m.compile("adam", "sparse_categorical_crossentropy")
    m.fit(np.random.rand(16, 4).astype("float32"), np.random.randint(0, 2, 16), epochs=1, verbose=0)
    linha("keras fit", "ok (1 época, modelo mínimo)")
    linha("tf.keras", f"{tf.keras.__version__}  ImageDataGenerator={hasattr(tf.keras.preprocessing.image, 'ImageDataGenerator')}")
except Exception as e: falhas.append(f"tensorflow/keras: {e}")
try:
    import dlib, face_recognition
    import numpy as np
    face_recognition.face_locations(np.zeros((64, 64, 3), np.uint8))
    linha("dlib", dlib.__version__); linha("face_recognition", "importa e roda face_locations")
except Exception as e: falhas.append(f"face_recognition: {e}")
try:
    import torch, torchvision, tabulate, psutil
    from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
    from torchvision import transforms
    (torch.zeros(2, 3) + 1).sum().item()
    linha("torch", f"{torch.__version__}  cuda={torch.cuda.is_available()}")
    linha("torchvision", f"{torchvision.__version__}  fcn_resnet50 e deeplabv3_resnet50 importam")
    linha("tabulate / psutil", f"{tabulate.__version__} / {psutil.__version__}")
except Exception as e: falhas.append(f"torch/torchvision: {e}")
try:
    import ultralytics
    from ultralytics import YOLO
    linha("ultralytics", f"{ultralytics.__version__}  (o peso yolov8n.pt é baixado no primeiro uso)")
except Exception as e: falhas.append(f"ultralytics: {e}")
print()
if falhas:
    print("FALHAS:"); [print("  - " + f) for f in falhas]; sys.exit(1)
print("Verificação concluída sem falhas.")
'''


def verificar(py_venv: Path) -> bool:
    passo("Verificando o ambiente")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(CODIGO_VERIFICACAO)
        arq = f.name
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0")
    try:
        sys.stdout.flush()
        r = subprocess.run([str(py_venv), arq], env=env)
        return r.returncode == 0
    finally:
        os.unlink(arq)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Prepara o ambiente do projeto no Windows ou no Linux.")
    ap.add_argument("--recriar", action="store_true", help="apaga o venv antes de criar de novo")
    ap.add_argument("--sem-verificacao", action="store_true", help="não roda a verificação final")
    ap.add_argument("--venv-dir", help="pasta do venv (padrão: .venv no Windows, .venv_linux no Linux)")
    ap.add_argument("--python", help=f"caminho de um Python {VERSAO_PYTHON} a usar")
    args = ap.parse_args()

    inicio = time.time()
    sistema, eh_wsl = detectar_sistema()
    titulo(f"Preparação do ambiente  |  sistema: {sistema}{' (WSL)' if eh_wsl else ''}  |  raiz: {RAIZ}")

    if sistema == "windows":
        venv = Path(args.venv_dir) if args.venv_dir else RAIZ / ".venv"
    else:
        venv = Path(args.venv_dir) if args.venv_dir else RAIZ / ".venv_linux"
        if eh_wsl and str(RAIZ).startswith("/mnt/"):
            print("Aviso: o projeto está em /mnt/, disco do Windows visto pelo WSL. Instalar e importar")
            print("o TensorFlow por aí é mais lento do que no disco do Linux. Funciona, só demora mais.")

    # 2. Python 3.13
    passo(f"Procurando um Python {VERSAO_PYTHON}")
    if args.python:
        python_exe = args.python
        if versao_de(python_exe) != VERSAO_PYTHON:
            sys.exit(f"{python_exe} não é Python {VERSAO_PYTHON}.")
    else:
        python_exe = achar_python(sistema)
    if python_exe:
        print(f"Encontrado: {python_exe}")
    else:
        print(f"Nenhum Python {VERSAO_PYTHON} no sistema.")
        if sistema == "windows":
            print("Alternativa manual: instale o 3.13 em https://www.python.org/downloads/windows/ ou 'winget install Python.Python.3.13'.")
        else:
            print(f"Alternativa manual em Ubuntu: PPA deadsnakes e 'sudo apt-get install python{VERSAO_PYTHON} python{VERSAO_PYTHON}-venv'.")
        print("Seguindo pelo caminho sem administrador, com o uv.")
        python_exe = python_via_uv(sistema)
        print(f"Python via uv: {python_exe}")

    # 3. venv
    py_venv = criar_venv(python_exe, venv, sistema, args.recriar)

    # 4. pacotes
    instalar_pacotes(py_venv)

    # 5. libs de sistema
    if sistema == "linux":
        checar_libs_linux()

    # 6. verificação
    ok = True
    if not args.sem_verificacao:
        ok = verificar(py_venv)

    # resumo
    minutos = (time.time() - inicio) / 60
    titulo(f"{'Ambiente pronto' if ok else 'Ambiente com falhas'}  |  {minutos:.1f} min")
    print(f"Venv: {venv}")
    if sistema == "windows":
        print("Ativar no PowerShell:   .\\" + venv.name + "\\Scripts\\Activate.ps1")
        print("Ativar no cmd:          " + venv.name + "\\Scripts\\activate.bat")
        print("Ou rodar direto:        .\\" + venv.name + "\\Scripts\\python.exe caminho\\do\\script.py")
    else:
        print("Ativar:                 source " + venv.name + "/bin/activate")
        print("Ou rodar direto:        ./" + venv.name + "/bin/python caminho/do/script.py")
        if eh_wsl:
            print("Janelas do OpenCV (cv2.imshow) no WSL exigem WSLg (Windows 11) ou um servidor X.")
    print("Observação: 'pip check' reclama que faltam dlib e opencv-python. É esperado: o dlib vem do")
    print("dlib-bin e o cv2 vem do opencv-contrib-python (ver os cabeçalhos dos requirements).")
    print("FIM_PREPARO " + ("OK" if ok else "FALHA"))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nComando falhou com código {e.returncode}: {' '.join(str(c) for c in e.cmd)}")
        print("FIM_PREPARO FALHA")
        sys.exit(e.returncode or 1)
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
