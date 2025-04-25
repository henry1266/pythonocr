import os
import cv2
import numpy as np

# 資料夾設定
input_folder = "input"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# CLAHE 增強函式
def enhance_text_clahe(image_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取：{image_path}")
        return

    # 轉灰階
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 2: 背景模糊建模（去除大片光影）
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # 套用 CLAHE（局部對比增強）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    contrast_enhanced = clahe.apply(norm)

    # 二值化：強化文字
    binary = cv2.adaptiveThreshold(
        contrast_enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=10
    )

    # 可選：形態學膨脹（讓細文字更粗一點）
    kernel = np.ones((1, 1), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 儲存
    cv2.imwrite(save_path, binary)
    print(f"✅ 已處理：{os.path.basename(save_path)}")

# 遍歷 input 資料夾
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        enhance_text_clahe(input_path, output_path)
