# Python OCR 處方圖像處理工具

這是一個專為處理和優化處方圖像設計的Python工具，能夠清理圖像中的雜訊、增強文字對比度，並自動調整文字方向，使處方內容更加清晰易讀。

## 功能特點

- **雜訊清理**：移除圖像中的背景雜訊，保留文字內容
- **對比度增強**：提高文字與背景的對比度，使淺色文字更加清晰可見
- **文字銳化**：增強文字邊緣，提高可讀性
- **省墨模式**：將接近白色的像素轉換為純白色，減少列印時的墨水消耗
- **批量處理**：支持處理整個資料夾中的多張圖像
- **自動轉向**：檢測並自動調整文字方向，確保文字正確顯示
- **輸出格式**：處理後的圖像以JPG格式保存

## 安裝說明

### 必要條件

- Python 3.6 或更高版本
- OpenCV (cv2)
- NumPy
- Matplotlib (可選，用於顯示結果)
- PIL (Python Imaging Library)

### 安裝步驟

1. 克隆此倉庫：
   ```
   git clone https://github.com/henry1266/pythonocr.git
   cd pythonocr
   ```

2. 安裝依賴項：
   ```
   pip install opencv-python numpy matplotlib pillow
   ```

3. Windows用戶可以直接運行`install.bat`安裝所需依賴：
   ```
   install.bat
   ```

## 使用方法

### 命令行使用

基本用法：
```
python batch_process.py --input <輸入路徑> --output <輸出路徑>
```

處理單個圖像：
```
python batch_process.py --input path/to/image.jpg --output output_folder
```

處理整個資料夾：
```
python batch_process.py --input input_folder --output output_folder
```

Windows用戶可以直接運行`run_batch_process.bat`：
```
run_batch_process.bat
```

### 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--input` | 輸入圖片路徑或資料夾路徑 | `input` |
| `--output` | 輸出資料夾路徑 | `output` |
| `--noise_threshold` | 雜訊閾值 (0-255)，越高越能保留淺色文字 | 220 |
| `--median_blur_size` | 中值濾波器大小，必須是奇數，用於去除椒鹽噪聲 | 5 |
| `--morph_kernel_size` | 形態學操作的核大小，影響雜訊移除的強度 | 1 |
| `--morph_iterations` | 形態學操作的迭代次數，影響雜訊移除的強度 | 3 |
| `--contrast_alpha` | 對比度增強係數，大於1增加對比度 | 1.0 |
| `--contrast_beta` | 亮度調整值，正值增加亮度 | 10 |
| `--sharpen_kernel_size` | 銳化核大小，設為0禁用銳化 | 3 |
| `--sharpen_strength` | 銳化強度，設為0禁用銳化 | 1.0 |
| `--ink_saving` | 啟用省墨模式，將接近白色的像素轉換為純白色 | True |
| `--ink_threshold` | 省墨模式下的閾值 (0-255)，高於此值的像素將變為純白色 | 245 |
| `--show` | 顯示處理結果 | False |

## 使用範例

### 增強淺色文字

對於含有淺色文字的處方圖像，可以調低雜訊閾值並增加對比度：
```
python batch_process.py --input faded_prescription.jpg --output output --noise_threshold 180 --contrast_alpha 1.5 --contrast_beta 20
```

### 處理高雜訊圖像

對於含有大量雜訊的圖像，可以增加形態學操作的強度：
```
python batch_process.py --input noisy_prescription.jpg --output output --morph_kernel_size 2 --morph_iterations 5
```

### 批量處理並顯示結果

處理整個資料夾並顯示處理結果：
```
python batch_process.py --input prescriptions_folder --output processed_folder --show
```

## 目錄結構

```
pythonocr/
├── batch_process.py     # 主程式
├── install.bat          # Windows安裝腳本
├── run_batch_process.bat # Windows運行腳本
├── input/               # 預設輸入資料夾
└── output/              # 預設輸出資料夾（自動創建）
```

## 注意事項

- 處理大量圖像時可能需要較長時間
- 最佳參數設置可能因圖像質量和內容而異
- 建議先使用少量圖像測試參數效果，再進行批量處理

## 授權

此專案僅供學習和研究使用。

## 貢獻

歡迎提交問題報告和改進建議。
