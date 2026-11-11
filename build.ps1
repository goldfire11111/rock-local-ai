# Builds Rock.exe (a single-file Windows app) from the source in this folder.
# Usage:  powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venv = "build\venv"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    py -3 -m venv $venv
    & "$venv\Scripts\python.exe" -m pip install --upgrade pip
}
& "$venv\Scripts\python.exe" -m pip install -q pyinstaller pystray pillow

# App icon
& "$venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, '.'); from rockllm.app import make_icon_image; make_icon_image(256).save('build/rock.ico', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])"

$root = (Get-Location).Path
& "$venv\Scripts\pyinstaller.exe" --noconfirm --clean --onefile --name Rock `
    --icon "$root\build\rock.ico" `
    --add-data "$root\rockllm\web;rockllm\web" `
    --hidden-import rockllm.app --hidden-import rockllm.server --hidden-import pystray._win32 `
    --exclude-module tkinter --exclude-module unittest --exclude-module pydoc `
    --hide-console hide-early `
    --distpath "$root" --workpath "$root\build\work" --specpath "$root\build" `
    rock.py

Write-Host "`nBuilt: $root\Rock.exe"
