@echo off
echo 正在執行OCR程式...
python ocr.py --method adaptive --adaptive_block_size 11 --adaptive_c 2 --show
echo.
echo 程式執行完畢，請按任意鍵退出。
pause > nul
