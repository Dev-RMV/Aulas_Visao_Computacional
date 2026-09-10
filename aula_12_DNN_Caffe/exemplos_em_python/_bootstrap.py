"""Inicializacao comum dos exemplos.

Permite executar qualquer arquivo diretamente, inclusive pelo VS Code/PowerShell,
sem precisar configurar PYTHONPATH manualmente. Tambem fixa a pasta de trabalho
na raiz do projeto, para que caminhos como data/, models/ e resultados/ sejam
sempre resolvidos no lugar correto.
"""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

os.chdir(PROJECT_ROOT)
