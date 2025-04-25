import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
import glob

def detect_orientation(image):
    """
    檢測圖像方向並返回旋轉角度
    
    參數:
    image: 輸入圖像
    
    返回:
    rotation_angle: 旋轉角度 (0, 90, 180, 或 270)
    """
    # 獲取圖像尺寸
    height, width = image.shape[:2]
    
    # 如果寬度大於高度，假設圖像需要旋轉
    if width > height:
        # 轉換為灰度圖
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        
        # 使用霍夫變換檢測直線
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            # 計算水平和垂直線的數量
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 45 or angle > 135:
                    horizontal_lines += 1
                else:
                    vertical_lines += 1
            
            # 如果垂直線比水平線多，旋轉90度
            if vertical_lines > horizontal_lines:
                return 90
            else:
                return 0
        else:
            # 如果沒有檢測到線條，根據文字區域分布判斷
            # 將圖像分為左右兩半，計算每半部分的文字像素數量
            left_half = gray[:, :width//2]
            right_half = gray[:, width//2:]
            
            left_text = np.sum(left_half < 128)
            right_text = np.sum(right_half < 128)
            
            # 如果左半部分文字明顯多於右半部分，可能需要旋轉
            if left_text > 2 * right_text:
                return 180
            elif right_text > 2 * left_text:
                return 0
    
    # 默認不旋轉
    return 0

def clean_noise(input_image_path, output_image_path, 
               noise_threshold=200, median_blur_size=5, 
               morph_kernel_size=1, morph_iterations=3,
               contrast_alpha=1.5, contrast_beta=15,
               sharpen_kernel_size=3, sharpen_strength=2.0,
               ink_saving_mode=True, ink_threshold=245,
               auto_rotate=True):
    """
    清理圖像周圍的雜點，保留文字內容，並增強對比度
    
    參數:
    input_image_path: 輸入圖片路徑
    output_image_path: 輸出圖片路徑
    noise_threshold: 雜訊閾值，越高越能保留淺色文字
    median_blur_size: 中值濾波器大小，用於去除椒鹽噪聲
    morph_kernel_size: 形態學操作的核大小
    morph_iterations: 形態學操作的迭代次數
    contrast_alpha: 對比度增強係數，大於1增加對比度
    contrast_beta: 亮度調整值，正值增加亮度
    sharpen_kernel_size: 銳化核大小
    sharpen_strength: 銳化強度
    ink_saving_mode: 是否啟用省墨模式
    ink_threshold: 省墨模式下的閾值，高於此值的像素將變為純白色
    auto_rotate: 是否自動檢測並修正圖像方向
    """
    # 讀取圖片
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"無法讀取圖片: {input_image_path}")
        return None
    
    # 自動旋轉圖像
    if auto_rotate:
        rotation_angle = detect_orientation(image)
        if rotation_angle != 0:
            print(f"檢測到圖像需要旋轉 {rotation_angle} 度")
            if rotation_angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif rotation_angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
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
    
    # 增強對比度
    enhanced = cv2.convertScaleAbs(result, alpha=contrast_alpha, beta=contrast_beta)
    
    # 銳化處理，使文字更清晰
    if sharpen_kernel_size > 0 and sharpen_strength > 0:
        # 創建銳化核
        sharpen_kernel = np.array([[-1, -1, -1],
                                  [-1, 9 + sharpen_strength, -1],
                                  [-1, -1, -1]])
        # 應用銳化
        enhanced = cv2.filter2D(enhanced, -1, sharpen_kernel)
    
    # 進行額外的雜點清理
    enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    _, enhanced_binary = cv2.threshold(enhanced_gray, noise_threshold, 255, cv2.THRESH_BINARY)
    
    # 使用連通區域分析移除小雜點
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - enhanced_binary, connectivity=8)
    
    # 創建輸出圖像
    final_result = np.ones_like(enhanced) * 255
    
    # 只保留足夠大的連通區域（文字）
    min_size = 10  # 最小連通區域大小
    for i in range(1, num_labels):  # 從1開始，跳過背景
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            final_result[labels == i] = enhanced[labels == i]
    
    # 省墨模式：將接近白色的像素轉換為純白色
    if ink_saving_mode:
        # 轉換為灰度圖
        final_gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
        
        # 創建遮罩，標記所有高於閾值的像素
        ink_mask = final_gray > ink_threshold
        
        # 將這些像素設為純白色
        final_result[ink_mask] = [255, 255, 255]
        
        # 進行二值化處理，使文字更加黑白分明
        for c in range(3):  # 對每個顏色通道
            _, final_result[:,:,c] = cv2.threshold(final_result[:,:,c], ink_threshold, 255, cv2.THRESH_BINARY)
    
    # 保存為圖片
    cv2.imwrite(output_image_path, final_result)
    
    return final_result

def process_folder(input_folder, output_folder, 
                  noise_threshold=200, median_blur_size=5, 
                  morph_kernel_size=1, morph_iterations=3,
                  contrast_alpha=1.5, contrast_beta=15,
                  sharpen_kernel_size=3, sharpen_strength=2.0,
                  ink_saving_mode=True, ink_threshold=245,
                  auto_rotate=True,
                  show_results=False):
    """
    處理資料夾中的所有圖片
    
    參數:
    input_folder: 輸入資料夾路徑
    output_folder: 輸出資料夾路徑
    其他參數與clean_noise函數相同
    """
    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 獲取輸入資料夾中的所有圖片
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
    
    if not image_files:
        print(f"在 {input_folder} 中未找到任何圖片")
        return
    
    print(f"找到 {len(image_files)} 個圖片檔案")
    
    # 處理每個圖片
    for i, image_file in enumerate(image_files):
        # 獲取檔案名（不含路徑和副檔名）
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        
        # 設定輸出路徑
        output_image_path = os.path.join(output_folder, f"{base_name}_clean.jpg")
        
        print(f"處理圖片 {i+1}/{len(image_files)}: {image_file}")
        
        # 處理圖片
        result = clean_noise(
            image_file, 
            output_image_path,
            noise_threshold,
            median_blur_size,
            morph_kernel_size,
            morph_iterations,
            contrast_alpha,
            contrast_beta,
            sharpen_kernel_size,
            sharpen_strength,
            ink_saving_mode,
            ink_threshold,
            auto_rotate
        )
        
        if result is None:
            print(f"處理 {image_file} 失敗")
            continue
        
        print(f"已保存處理結果為 {output_image_path}")
        
        # 顯示結果
        if show_results:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 10))
            plt.imshow(result_rgb)
            plt.axis('off')
            plt.title(f'Cleaned Image: {base_name}')
            plt.show()
    
    print(f"所有圖片處理完成！結果已保存到 {output_folder}")

def main():
    # 設定命令行參數
    parser = argparse.ArgumentParser(description='清理圖像周圍的雜點，保留文字內容，並增強對比度')
    parser.add_argument('--input', type=str, default='input', 
                        help='輸入圖片路徑或資料夾路徑')
    parser.add_argument('--output', type=str, default='output', 
                        help='輸出資料夾路徑')
    parser.add_argument('--noise_threshold', type=int, default=200, 
                        help='雜訊閾值 (0-255)，越高越能保留淺色文字')
    parser.add_argument('--median_blur_size', type=int, default=5, 
                        help='中值濾波器大小，必須是奇數，用於去除椒鹽噪聲')
    parser.add_argument('--morph_kernel_size', type=int, default=1, 
                        help='形態學操作的核大小，影響雜訊移除的強度')
    parser.add_argument('--morph_iterations', type=int, default=3, 
                        help='形態學操作的迭代次數，影響雜訊移除的強度')
    parser.add_argument('--contrast_alpha', type=float, default=1.5, 
                        help='對比度增強係數，大於1增加對比度')
    parser.add_argument('--contrast_beta', type=int, default=15, 
                        help='亮度調整值，正值增加亮度')
    parser.add_argument('--sharpen_kernel_size', type=int, default=3, 
                        help='銳化核大小，設為0禁用銳化')
    parser.add_argument('--sharpen_strength', type=float, default=2.0, 
                        help='銳化強度，設為0禁用銳化')
    parser.add_argument('--ink_saving', action='store_true', default=True,
                        help='啟用省墨模式，將接近白色的像素轉換為純白色')
    parser.add_argument('--ink_threshold', type=int, default=245, 
                        help='省墨模式下的閾值 (0-255)，高於此值的像素將變為純白色')
    parser.add_argument('--auto_rotate', action='store_true', default=True,
                        help='自動檢測並修正圖像方向')
    parser.add_argument('--show', action='store_true', 
                        help='顯示處理結果')
    
    args = parser.parse_args()
    
    # 檢查輸入是資料夾還是單個文件
    if os.path.isdir(args.input):
        # 處理資料夾
        process_folder(
            args.input,
            args.output,
            args.noise_threshold,
            args.median_blur_size,
            args.morph_kernel_size,
            args.morph_iterations,
            args.contrast_alpha,
            args.contrast_beta,
            args.sharpen_kernel_size,
            args.sharpen_strength,
            args.ink_saving,
            args.ink_threshold,
            args.auto_rotate,
            args.show
        )
    else:
        # 處理單個文件
        # 確保輸出資料夾存在
        os.makedirs(args.output, exist_ok=True)
        
        # 獲取檔案名（不含路徑和副檔名）
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        
        # 設定輸出路徑
        output_image_path = os.path.join(args.output, f"{base_name}_clean.jpg")
        
        # 處理圖片
        result = clean_noise(
            args.input, 
            output_image_path,
            args.noise_threshold,
            args.median_blur_size,
            args.morph_kernel_size,
            args.morph_iterations,
            args.contrast_alpha,
            args.contrast_beta,
            args.sharpen_kernel_size,
            args.sharpen_strength,
            args.ink_saving,
            args.ink_threshold,
            args.auto_rotate
        )
        
        if result is None:
            return
        
        print(f"處理完成！結果已保存為 {output_image_path}")
        
        # 顯示結果
        if args.show:
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 10))
            plt.imshow(result_rgb)
            plt.axis('off')
            plt.title('Cleaned Text Image with Enhanced Contrast')
            plt.show()

if __name__ == "__main__":
    main()
