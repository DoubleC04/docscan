from PIL import Image
import cv2
import numpy as np
import io

def create_pdf_from_images(image_list_bgr):
    """
    Tạo một file PDF từ danh sách các ảnh OpenCV (BGR).

    :param image_list_bgr: Danh sách các mảng ảnh numpy theo định dạng BGR.
    :return: Dữ liệu bytes của file PDF đã tạo.
    """
    if not image_list_bgr:
        return None

    pil_images = []
    for img_bgr in image_list_bgr:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(img_rgb))

    pdf_buffer = io.BytesIO()

    pil_images[0].save(
        pdf_buffer,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=pil_images[1:]
    )

    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    
    return pdf_bytes