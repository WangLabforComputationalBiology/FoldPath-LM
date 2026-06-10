@echo off
echo ============================================================
echo   FoldPath-LLM CUDA Setup Script
echo   Installing PyTorch with CUDA support for RTX 4060
echo ============================================================
echo.

:: Check current PyTorch
python -c "import torch; print('Current PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo.
echo Uninstalling CPU-only PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo Installing PyTorch with CUDA 12.6...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo.
echo Installing ESM-2 dependencies...
pip install fair-esm transformers sentencepiece

echo.
echo Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo Testing ESM-2 import...
python -c "import esm; print('fair-esm OK'); print('Models:', esm.pretrained.__all__[:2])" 2>nul || echo "fair-esm import warning (may need transformers instead)"

echo.
echo ============================================================
echo   Setup complete! Run: python webapp.py
echo ============================================================
pause
