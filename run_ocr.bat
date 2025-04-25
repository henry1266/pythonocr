@echo off
echo 正在安裝必要的套件...
pip install pytesseract
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pillow
pip install reportlab
pip install argparse
echo.
echo 正在執行OCR程式...
python ocr.py --threshold 240 --min_area 5 --kernel_size 2 --iterations 1 --show
echo.
echo 程式執行完畢，請按任意鍵退出。
pause > nul
