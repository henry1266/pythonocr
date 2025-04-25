@echo off
echo 正在執行批量處理程式...
python batch_process.py --noise_threshold 200 --median_blur_size 5 --morph_kernel_size 1 --morph_iterations 3 --contrast_alpha 1.5 --contrast_beta 15 --sharpen_kernel_size 3 --sharpen_strength 2.0 --ink_saving --ink_threshold 245 --auto_rotate
echo.
echo 程式執行完畢，請按任意鍵退出。
pause > nul
