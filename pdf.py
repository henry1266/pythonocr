import fitz  # PyMuPDF
from PIL import Image
import os
import glob

input_folder = "pdf"
output_folder = "input"
os.makedirs(output_folder, exist_ok=True)

pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

if not pdf_files:
    print("❗ 沒有找到 PDF 檔案於 input/ 資料夾中。")
    exit()

for pdf_path in pdf_files:
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"📄 處理中：{filename}.pdf")

    doc = fitz.open(pdf_path)
    images = []

    for page_number in range(len(doc)):
        pix = doc.load_page(page_number).get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    for i in range(0, len(images), 2):
        imgs_to_merge = images[i:i+2]

        if len(imgs_to_merge) == 1:
            blank = Image.new("RGB", imgs_to_merge[0].size, color="white")
            imgs_to_merge.append(blank)

        width = max(img.width for img in imgs_to_merge)
        height = sum(img.height for img in imgs_to_merge)
        combined = Image.new("RGB", (width, height), color="white")

        y_offset = 0
        for img in imgs_to_merge:
            combined.paste(img, (0, y_offset))
            y_offset += img.height

        output_name = f"{filename}_merged_{i//2 + 1}.jpg"
        output_path = os.path.join(output_folder, output_name)
        combined.save(output_path, "JPEG")

    print(f"✅ 完成：{filename}.pdf → {output_folder}")

print("🎉 所有 PDF 處理完成！")
