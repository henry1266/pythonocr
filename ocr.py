import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

# 讀取圖片
image = cv2.imread('input_image.jpg')  # 請替換為您的圖片路徑

# 轉換為灰度圖
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 應用二值化
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# 尋找文字區域
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 創建遮罩
mask = np.zeros_like(gray)

# 在遮罩上繪製文字區域
for contour in contours:
    # 過濾小區域
    if cv2.contourArea(contour) > 50:
        cv2.drawContours(mask, [contour], -1, 255, -1)

# 將非文字區域補為白色背景，而非黑色
# 創建白色背景
white_bg = np.ones_like(image) * 255

# 將原始影像的文字區塊貼到白色背景上
result_on_white = np.where(mask[:, :, np.newaxis] == 255, image, white_bg)

# 轉換為 RGB 以供顯示
result_rgb_white_bg = cv2.cvtColor(result_on_white.astype(np.uint8), cv2.COLOR_BGR2RGB)

# 顯示圖片
plt.figure(figsize=(12, 10))
plt.imshow(result_rgb_white_bg)
plt.axis('off')
plt.title('Only Text Retained on White Background')
plt.show()

# 使用 Tesseract 進行 OCR
text = pytesseract.image_to_string(Image.fromarray(result_on_white), lang='chi_tra+eng')
print("識別的文字：")
print(text)

# 保存結果
cv2.imwrite('text_on_white_bg.jpg', result_on_white)
with open('ocr_result.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("處理完成！結果已保存為 text_on_white_bg.jpg 和 ocr_result.txt")
