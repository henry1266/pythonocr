import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# 讀取圖片
image = cv2.imread('1.jpg')  # 使用同資料夾中的1.jpg檔案

# 轉換為灰度圖
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 應用自適應閾值處理，更好地處理不同亮度區域的文字
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv2.THRESH_BINARY_INV, 11, 2)

# 使用形態學操作來連接相近的文字區域
kernel = np.ones((3,3), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# 尋找文字區域
contours, hierarchy = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 創建遮罩
mask = np.zeros_like(gray)

# 分析輪廓特徵以識別文字區域
text_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    # 過濾條件：面積適中、寬高比合理的區域更可能是文字
    if area > 50 and area < 50000 and aspect_ratio > 0.1 and aspect_ratio < 10:
        text_contours.append(contour)
        cv2.drawContours(mask, [contour], -1, 255, -1)

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
plt.title('Text Regions on White Background')
plt.show()

# 保存為圖片
output_image_path = 'text_on_white_bg.jpg'
cv2.imwrite(output_image_path, result_on_white)

# 創建PDF
output_pdf_path = 'text_output.pdf'
img_pil = Image.fromarray(cv2.cvtColor(result_on_white, cv2.COLOR_BGR2RGB))
img_width, img_height = img_pil.size

# 計算PDF頁面大小，保持圖像比例
pdf_w, pdf_h = letter
if img_width > img_height:
    # 橫向圖像
    pdf_w, pdf_h = pdf_h, pdf_w

# 計算縮放比例，使圖像適合頁面並留有邊距
margin = 50
scale = min((pdf_w - 2*margin) / img_width, (pdf_h - 2*margin) / img_height)
new_width = img_width * scale
new_height = img_height * scale

# 創建PDF
c = canvas.Canvas(output_pdf_path, pagesize=(pdf_w, pdf_h))
# 計算居中位置
x_centered = (pdf_w - new_width) / 2
y_centered = (pdf_h - new_height) / 2
# 在PDF中繪製圖像
c.drawImage(output_image_path, x_centered, y_centered, width=new_width, height=new_height)
c.save()

print(f"處理完成！結果已保存為 {output_image_path} 和 {output_pdf_path}")
