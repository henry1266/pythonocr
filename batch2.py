import os
import cv2
import numpy as np

# 資料夾設定
input_folder = "input"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# CLAHE 增強函式（只輸出成品）
def enhance_text_clahe(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取：{image_path}")
        return

    # 1. 轉灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Normalize
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # 3. CLAHE 對比增強
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(norm)

    # 4. 膨脹處理（最終成品）
    kernel = np.ones((1, 1), np.uint8)
    final_image = cv2.dilate(contrast_enhanced, kernel, iterations=1)

    # 只儲存最終結果
    cv2.imwrite(save_path, final_image)
    print(f"✅ 已儲存成品：{os.path.basename(save_path)}")

# 遍歷 input 資料夾
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        enhance_text_clahe(input_path, output_path)
