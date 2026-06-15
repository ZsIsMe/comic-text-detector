#!/usr/bin/env python3
"""CTD 疊圖檢視器啟動器。"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path


APP_NAME = 'CTD 疊圖檢視器'
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
VENV_DIR = ROOT / '.venv'
REQUIREMENTS = ROOT / 'requirements.txt'
PROJECT_REQUIREMENTS = PROJECT_ROOT / 'requirements.txt'
VIEWER = ROOT / 'viewer.py'


def info(message: str) -> None:
    print(f'[{APP_NAME}] {message}', flush=True)


def fail(message: str, code: int = 1) -> None:
    print(f'[{APP_NAME}] 錯誤：{message}', file=sys.stderr, flush=True)
    raise SystemExit(code)


def ensure_python_version() -> None:
    if sys.version_info < (3, 10):
        fail('需要 Python 3.10 或更新版本。')


def venv_python() -> Path:
    if os.name == 'nt':
        return VENV_DIR / 'Scripts' / 'python.exe'
    return VENV_DIR / 'bin' / 'python'


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    info('執行：' + ' '.join(cmd))
    try:
        subprocess.check_call(cmd, cwd=str(cwd or ROOT))
    except subprocess.CalledProcessError as exc:
        fail(f'命令失敗，退出碼 {exc.returncode}：{" ".join(cmd)}', exc.returncode)


def ensure_venv() -> Path:
    python = venv_python()
    if python.exists():
        return python

    info('正在建立虛擬環境...')
    try:
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    except Exception as exc:
        fail(f'無法建立虛擬環境：{exc}')

    if not python.exists():
        fail(f'找不到虛擬環境 Python：{python}')
    return python


def ensure_dependencies(python: Path) -> None:
    if not REQUIREMENTS.exists():
        fail(f'找不到依賴清單：{REQUIREMENTS}')

    stamp = VENV_DIR / '.ctd_overlay_requirements_installed'
    req_mtime = max(
        REQUIREMENTS.stat().st_mtime,
        PROJECT_REQUIREMENTS.stat().st_mtime if PROJECT_REQUIREMENTS.exists() else 0,
    )
    if stamp.exists():
        try:
            if float(stamp.read_text(encoding='utf-8')) >= req_mtime:
                return
        except ValueError:
            pass

    info('正在安裝 Python 依賴。第一次啟動會比較久...')
    run([str(python), '-m', 'pip', 'install', '-U', 'pip'])
    if PROJECT_REQUIREMENTS.exists():
        info('正在安裝偵測流程依賴...')
        run([str(python), '-m', 'pip', 'install', '-r', str(PROJECT_REQUIREMENTS)], cwd=PROJECT_ROOT)
    run([str(python), '-m', 'pip', 'install', '-r', str(REQUIREMENTS)])
    stamp.write_text(str(req_mtime), encoding='utf-8')


def launch_app(python: Path) -> None:
    if not VIEWER.exists():
        fail(f'找不到檢視器腳本：{VIEWER}')
    info('正在啟動介面...')
    run([str(python), str(VIEWER)])


def main() -> None:
    ensure_python_version()
    python = ensure_venv()
    ensure_dependencies(python)
    launch_app(python)


if __name__ == '__main__':
    main()
