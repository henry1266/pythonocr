@echo off
echo 正在執行雜點清理程式...
python clean_noise.py --noise_threshold 220 --median_blur_size 5 --morph_kernel_size 3 --morph_iterations 2 --show
echo.
echo 程式執行完畢，請按任意鍵退出。
pause > nul
