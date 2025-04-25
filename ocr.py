import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import argparse

def process_image(input_image_path, threshold=240, min_area=5, kernel_size=2, iterations=1):
    """
    處理圖片，保留文字區域並輸出為白色背景的圖片和PDF
    
    參數:
    input_image_path: 輸入圖片路徑
    threshold: 二值化閾值 (0-255)，越高越能檢測淺色文字
    min_area: 最小文字區域面積，越小越能保留小文字
    kernel_size: 形態學操作的核大小，影響文字連接程度
    iterations: 膨脹操作的迭代次數，影響文字連接程度
    """
    # 讀取圖片
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"無法讀取圖片: {input_image_path}")
        return None, None
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用較高的閾值進行二值化，確保淺色文字也能被檢測到
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # 使用形態學操作來連接相近的文字
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
    
    # 尋找所有可能的文字區域
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 創建遮罩
    mask = np.zeros_like(gray)
    
    # 在遮罩上繪製所有可能的文字區域，使用非常保守的過濾
    for contour in contours:
        # 只過濾非常小的雜訊
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # 將非文字區域補為白色背景
    white_bg = np.ones_like(image) * 255
    
    # 將原始影像的文字區塊貼到白色背景上
    result_on_white = np.where(mask[:, :, np.newaxis] == 255, image, white_bg)
    
    # 轉換為 RGB 以供顯示
    result_rgb_white_bg = cv2.cvtColor(result_on_white.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    return result_on_white, result_rgb_white_bg

def save_output(result_image, output_image_path, output_pdf_path):
    """
    保存處理後的圖片為JPG和PDF
    
    參數:
    result_image: 處理後的圖片
    output_image_path: 輸出JPG路徑
    output_pdf_path: 輸出PDF路徑
    """
    # 保存為圖片
    cv2.imwrite(output_image_path, result_image)
    
    # 創建PDF
    img_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
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
    
    return output_image_path, output_pdf_path

def main():
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='處理圖片，保留文字區域並輸出為白色背景的圖片和PDF')
    parser.add_argument('--input', type=str, default='1.jpg', help='輸入圖片路徑')
    parser.add_argument('--threshold', type=int, default=240, help='二值化閾值 (0-255)，越高越能檢測淺色文字')
    parser.add_argument('--min_area', type=int, default=5, help='最小文字區域面積，越小越能保留小文字')
    parser.add_argument('--kernel_size', type=int, default=2, help='形態學操作的核大小，影響文字連接程度')
    parser.add_argument('--iterations', type=int, default=1, help='膨脹操作的迭代次數，影響文字連接程度')
    parser.add_argument('--output_image', type=str, default='text_on_white_bg.jpg', help='輸出圖片路徑')
    parser.add_argument('--output_pdf', type=str, default='text_output.pdf', help='輸出PDF路徑')
    parser.add_argument('--show', action='store_true', help='顯示處理結果')
    
    args = parser.parse_args()
    
    # 處理圖片
    result_on_white, result_rgb_white_bg = process_image(
        args.input, 
        args.threshold, 
        args.min_area, 
        args.kernel_size, 
        args.iterations
    )
    
    if result_on_white is None:
        return
    
    # 保存結果
    output_image_path, output_pdf_path = save_output(
        result_on_white, 
        args.output_image, 
        args.output_pdf
    )
    
    # 顯示結果
    if args.show:
        plt.figure(figsize=(12, 10))
        plt.imshow(result_rgb_white_bg)
        plt.axis('off')
        plt.title('Text Regions on White Background')
        plt.show()
    
    print(f"處理完成！結果已保存為 {output_image_path} 和 {output_pdf_path}")

if __name__ == "__main__":
    main()
