@echo off
echo 正在執行雜點清理程式...
python clean_noise.py --noise_threshold 220 --median_blur_size 5 --morph_kernel_size 1 --morph_iterations 3 --contrast_alpha 1.3 --contrast_beta 10 --sharpen_kernel_size 3 --sharpen_strength 1.5 --show
echo.
echo 程式執行完畢，請按任意鍵退出。
pause > nul
