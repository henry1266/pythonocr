import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import argparse

def process_image(input_image_path, output_image_path, output_pdf_path, 
                 method='canny', canny_low=50, canny_high=150, 
                 adaptive_block_size=11, adaptive_c=2):
    """
    處理圖片，保留文字區域並移除非文字元素
    
    參數:
    input_image_path: 輸入圖片路徑
    output_image_path: 輸出圖片路徑
    output_pdf_path: 輸出PDF路徑
    method: 文字檢測方法，'canny'或'adaptive'
    canny_low: Canny邊緣檢測的低閾值
    canny_high: Canny邊緣檢測的高閾值
    adaptive_block_size: 自適應閾值的塊大小
    adaptive_c: 自適應閾值的常數
    """
    # 讀取圖片
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"無法讀取圖片: {input_image_path}")
        return None
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 創建白色背景
    white_bg = np.ones_like(image) * 255
    
    # 根據選擇的方法處理圖片
    if method == 'canny':
        # 使用Canny邊緣檢測
        edges = cv2.Canny(gray, canny_low, canny_high)
        # 膨脹邊緣以連接相近的文字
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # 將邊緣區域視為文字
        mask = dilated_edges
    else:  # 'adaptive'
        # 使用自適應閾值處理
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c)
        # 膨脹以連接相近的文字
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(binary, kernel, iterations=1)
    
    # 將原始影像的文字區塊貼到白色背景上
    result_on_white = np.where(mask[:, :, np.newaxis] == 255, image, white_bg)
    
    # 保存為圖片
    cv2.imwrite(output_image_path, result_on_white)
    
    # 創建PDF
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
    
    return result_on_white

def main():
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='處理圖片，保留文字區域並移除非文字元素')
    parser.add_argument('--input', type=str, default='1.jpg', help='輸入圖片路徑')
    parser.add_argument('--output_image', type=str, default='text_on_white_bg.jpg', help='輸出圖片路徑')
    parser.add_argument('--output_pdf', type=str, default='text_output.pdf', help='輸出PDF路徑')
    parser.add_argument('--method', type=str, choices=['canny', 'adaptive'], default='adaptive', 
                        help='文字檢測方法: canny或adaptive')
    parser.add_argument('--canny_low', type=int, default=50, help='Canny邊緣檢測的低閾值')
    parser.add_argument('--canny_high', type=int, default=150, help='Canny邊緣檢測的高閾值')
    parser.add_argument('--adaptive_block_size', type=int, default=11, help='自適應閾值的塊大小')
    parser.add_argument('--adaptive_c', type=int, default=2, help='自適應閾值的常數')
    parser.add_argument('--show', action='store_true', help='顯示處理結果')
    
    args = parser.parse_args()
    
    # 處理圖片
    result = process_image(
        args.input, 
        args.output_image, 
        args.output_pdf,
        args.method,
        args.canny_low,
        args.canny_high,
        args.adaptive_block_size,
        args.adaptive_c
    )
    
    if result is None:
        return
    
    # 顯示結果
    if args.show:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 10))
        plt.imshow(result_rgb)
        plt.axis('off')
        plt.title('Text Regions on White Background')
        plt.show()
    
    print(f"處理完成！結果已保存為 {args.output_image} 和 {args.output_pdf}")

if __name__ == "__main__":
    main()
