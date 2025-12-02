import streamlit as st
import numpy as np
from PIL import Image
import cv2
import threading
import io

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from docscan.processing.rectify import DocumentRectifier
from docscan.processing.dewarp import DocumentDewarper
from docscan.services import pdf_generator

document_rectifier = DocumentRectifier()
document_dewarper = DocumentDewarper()
st.set_page_config(layout="wide", page_title="DocScan Pro")

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "STREAMING"
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'scanned_pages' not in st.session_state:
    st.session_state.scanned_pages = [] 
if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "Scan ảnh thông thường"

# --- LỚP XỬ LÝ VIDEO FRAME ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        with self.lock:
            self.latest_frame = img
            
        return frame

def process_image(image, mode):
    if mode == "Scan ảnh thông thường":
        return document_rectifier.rectify(image, apply_threshold=True)
    elif mode == "Làm phẳng trang":
        return document_dewarper.dewarp(image)
    elif mode == "Vá trang bị rách":
        st.warning("Tính năng 'Vá trang bị rách' sẽ được phát triển trong tương lai.")
        return image
    return image

# ==============================================================================
# GIAO DIỆN SIDEBAR 
# ==============================================================================
with st.sidebar:
    st.title("📚 Danh sách Trang")
    st.markdown("Các trang bạn đã scan sẽ xuất hiện ở đây.")

    if not st.session_state.scanned_pages:
        st.info("Chưa có trang nào được thêm vào.")
    else:
        for i, page_img in enumerate(st.session_state.scanned_pages):
            st.image(page_img, channels="BGR", caption=f"Trang {i+1}", width='stretch')
            st.markdown("---")
        
        st.success(f"Tổng cộng: {len(st.session_state.scanned_pages)} trang.")
        
        if st.button("📄 Tạo file PDF", type="primary", width='stretch'):
            with st.spinner("Đang tạo file PDF..."):
                pdf_bytes = pdf_generator.create_pdf_from_images(st.session_state.scanned_pages)
                
                st.download_button(
                    label="📥 Tải file PDF",
                    data=pdf_bytes,
                    file_name="scanned_documents.pdf",
                    mime="application/pdf",
                    width='stretch'
                )

# ==============================================================================
# GIAO DIỆN CHÍNH - DỰA TRÊN TABS 
# ==============================================================================
st.title("📄 DocScan Pro - Ứng dụng Scan Tài liệu Thông minh")
tab_upload, tab_camera = st.tabs(["📁 Tải ảnh lên", "📷 Sử dụng Camera"])

# === TAB 1: TẢI ẢNH TỪ MÁY TÍNH ===
with tab_upload:
    st.header("Bước 1: Chọn Chức năng và Tải ảnh")
    
    processing_mode_upload = st.radio(
        "Chọn chức năng bạn muốn sử dụng:",
        ("Scan ảnh thông thường", "Làm phẳng trang", "Vá trang bị rách"),
        key="radio_upload"
    )
    
    uploaded_file = st.file_uploader("Tải lên một file ảnh...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert('RGB')
        original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        processed_image = process_image(original_image, processing_mode_upload)
        
        st.header("Bước 2: Xem lại và Thêm vào danh sách")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, channels="BGR", caption="Ảnh gốc", width='stretch')
        with col2:
            st.image(processed_image, channels="BGR", caption="Ảnh đã xử lý", width='stretch')
        
        if st.button("✅ Thêm vào danh sách PDF", key="add_upload"):
            st.session_state.scanned_pages.append(processed_image)
            st.success(f"Đã thêm ảnh vào danh sách! Hiện có {len(st.session_state.scanned_pages)} trang.")
            st.rerun()

# === TAB 2: SỬ DỤNG CAMERA ===
with tab_camera:
    if st.session_state.app_mode == "STREAMING":
        st.header("Bước 1: Chọn Chức năng và Chụp ảnh")
        
        col_cam_1, col_cam_2 = st.columns(2)
        
        with col_cam_1:
            camera_choice = st.radio(
                "Chọn Camera",
                ("Camera sau", "Camera trước"),
                horizontal=True,
                key="camera_choice"
            )
        
        with col_cam_2:
            resolution_options = {
                "Vừa (1280x720)": (1280, 720),
                "Cao (1920x1080)": (1920, 1080),
                "Rất cao (3840x2160)": (3840, 2160), # 4K
                "Thấp (640x480)": (640, 480),
            }
            selected_resolution_key = st.selectbox(
                "Chọn Độ phân giải",
                options=list(resolution_options.keys()),
                key="resolution_choice"
            )
            res_width, res_height = resolution_options[selected_resolution_key]

        facing_mode = "environment" if camera_choice == "Camera sau" else "user"

        constraints = {
            "video": {
                "facingMode": facing_mode,
                "width": {"ideal": res_width},
                "height": {"ideal": res_height}
            },
            "audio": False,
        }
        
        st.session_state.processing_mode = st.radio(
            "Chọn chức năng bạn muốn sử dụng:",
            ("Scan ảnh thông thường", "Làm phẳng trang", "Vá trang bị rách"),
            key="radio_camera"
        )
        
        ctx = webrtc_streamer(
            key="camera", 
            video_processor_factory=VideoProcessor, 
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints=constraints 
        )
        
        capture_button = st.button("📸 Chụp ảnh", type="primary")

        if capture_button and ctx.video_processor:
            captured = ctx.video_processor.latest_frame
            
            if captured is not None:
                st.session_state.captured_image = captured
                st.session_state.app_mode = "REVIEWING"
                st.rerun()
            else:
                st.warning("Camera chưa sẵn sàng hoặc chưa có khung hình nào được ghi nhận, vui lòng thử lại.")

    elif st.session_state.app_mode == "REVIEWING":
        # ... (Phần này không cần thay đổi gì)
        st.header("Bước 2: Xem lại và Lựa chọn")
        
        captured_image_np = st.session_state.captured_image
        if captured_image_np is not None:
            processed_image = process_image(captured_image_np, st.session_state.processing_mode)

            col1, col2 = st.columns(2)
            with col1:
                st.image(captured_image_np, channels="BGR", caption="Ảnh vừa chụp", width='stretch')
            with col2:
                st.image(processed_image, channels="BGR", caption="Ảnh đã xử lý", width='stretch')

            btn_cols = st.columns(2)
            if btn_cols[0].button("✅ Thêm vào danh sách & Chụp tiếp", width='stretch', type="primary"):
                st.session_state.scanned_pages.append(processed_image)
                st.toast(f"Đã thêm! Hiện có {len(st.session_state.scanned_pages)} trang.")
                st.session_state.app_mode = "STREAMING"
                st.rerun()
            
            if btn_cols[1].button("🔄 Chụp lại", width='stretch'):
                st.session_state.app_mode = "STREAMING"
                st.rerun()