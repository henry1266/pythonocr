import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def clean_noise(input_image_path, output_image_path, output_pdf_path, 
               noise_threshold=220, median_blur_size=5, 
               morph_kernel_size=3, morph_iterations=2):
    """
    清理圖像周圍的雜點，保留文字內容
    
    參數:
    input_image_path: 輸入圖片路徑
    output_image_path: 輸出圖片路徑
    output_pdf_path: 輸出PDF路徑
    noise_threshold: 雜訊閾值，越高越能保留淺色文字
    median_blur_size: 中值濾波器大小，用於去除椒鹽噪聲
    morph_kernel_size: 形態學操作的核大小
    morph_iterations: 形態學操作的迭代次數
    """
    # 讀取圖片
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"無法讀取圖片: {input_image_path}")
        return None
    
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用中值濾波器去除椒鹽噪聲
    median_blurred = cv2.medianBlur(gray, median_blur_size)
    
    # 使用閾值處理將灰色雜訊轉換為純白色
    _, binary = cv2.threshold(median_blurred, noise_threshold, 255, cv2.THRESH_BINARY)
    
    # 使用形態學操作移除孤立的雜訊點
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    
    # 先進行開運算（先腐蝕後膨脹）去除小雜點
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=morph_iterations)
    
    # 再進行閉運算（先膨脹後腐蝕）填充文字內部的小洞
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    
    # 創建白色背景
    white_bg = np.ones_like(image) * 255
    
    # 將處理後的二值圖像轉換為遮罩
    mask = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    
    # 將原始圖像中的文字區域與白色背景合併
    # 注意：這裡我們將二值圖像反轉，因為我們想要保留黑色文字區域
    result = np.where(mask == 0, image, white_bg)
    
    # 保存為圖片
    cv2.imwrite(output_image_path, result)
    
    # 創建PDF
    img_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
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
    
    return result

def main():
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='清理圖像周圍的雜點，保留文字內容')
    parser.add_argument('--input', type=str, default='1.jpg', help='輸入圖片路徑')
    parser.add_argument('--output_image', type=str, default='clean_text.jpg', help='輸出圖片路徑')
    parser.add_argument('--output_pdf', type=str, default='clean_text.pdf', help='輸出PDF路徑')
    parser.add_argument('--noise_threshold', type=int, default=220, 
                        help='雜訊閾值 (0-255)，越高越能保留淺色文字')
    parser.add_argument('--median_blur_size', type=int, default=5, 
                        help='中值濾波器大小，必須是奇數，用於去除椒鹽噪聲')
    parser.add_argument('--morph_kernel_size', type=int, default=3, 
                        help='形態學操作的核大小，影響雜訊移除的強度')
    parser.add_argument('--morph_iterations', type=int, default=2, 
                        help='形態學操作的迭代次數，影響雜訊移除的強度')
    parser.add_argument('--show', action='store_true', help='顯示處理結果')
    
    args = parser.parse_args()
    
    # 處理圖片
    result = clean_noise(
        args.input, 
        args.output_image, 
        args.output_pdf,
        args.noise_threshold,
        args.median_blur_size,
        args.morph_kernel_size,
        args.morph_iterations
    )
    
    if result is None:
        return
    
    # 顯示結果
    if args.show:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 10))
        plt.imshow(result_rgb)
        plt.axis('off')
        plt.title('Cleaned Text Image')
        plt.show()
    
    print(f"處理完成！結果已保存為 {args.output_image} 和 {args.output_pdf}")

if __name__ == "__main__":
    main()
