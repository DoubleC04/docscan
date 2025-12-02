import cv2
import numpy as np

class DocumentRectifier:
    """
    Bộ quét tài liệu từ ảnh: phát hiện biên, cắt, và hiệu chỉnh phối cảnh (perspective).
    Đã cập nhật tính năng lưu ảnh debug (debug_images) để phục vụ giải trình.
    """

    def __init__(self, canny_thresh1=50, canny_thresh2=200, min_contour_area_factor=0.1, analysis_width=500):
        """
        Khởi tạo DocumentRectifier.
        """
        self.canny_thresh1 = canny_thresh1
        self.canny_thresh2 = canny_thresh2
        self.min_contour_area_factor = min_contour_area_factor
        self.analysis_width = analysis_width
        
        self.debug_images = {}

    def _reorder(self, points):
        """Sắp xếp lại 4 điểm theo thứ tự: TL, TR, BL, BR."""
        points = points.reshape((4, 2))
        new_points = np.zeros((4, 2), dtype="float32")
        
        add = points.sum(axis=1)
        new_points[0] = points[np.argmin(add)]
        new_points[3] = points[np.argmax(add)]
        
        diff = np.array([p[1] - p[0] for p in points])
        new_points[1] = points[np.argmin(diff)]
        new_points[2] = points[np.argmax(diff)]
        
        return new_points

    def _find_biggest_contour(self, contours, min_area):
        """Tìm contour lớn nhất có 4 đỉnh."""
        biggest = None
        max_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest

    def rectify(self, image_input, apply_threshold=True):
        """
        Xử lý ảnh và lưu các bước vào self.debug_images.
        """
        self.debug_images = {}

        if isinstance(image_input, str):
            image_orig = cv2.imread(image_input)
            if image_orig is None:
                raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_input}")
        elif isinstance(image_input, np.ndarray):
            image_orig = image_input.copy()
        else:
            raise TypeError("Đầu vào phải là đường dẫn ảnh hoặc numpy array")

        h_orig, w_orig = image_orig.shape[:2]
        ratio = h_orig / self.analysis_width
        analysis_height = int(w_orig / ratio)
        image_small = cv2.resize(image_orig, (analysis_height, self.analysis_width))

        img_gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
        self.debug_images['gray'] = img_gray

        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        self.debug_images['blur'] = img_blur
        
        img_edges = cv2.Canny(img_blur, self.canny_thresh1, self.canny_thresh2)
        self.debug_images['edges'] = img_edges 
        
        kernel = np.ones((5, 5), np.uint8)
        img_dil = cv2.dilate(img_edges, kernel, iterations=2)
        img_thresh = cv2.erode(img_dil, kernel, iterations=1)
        self.debug_images['morphology'] = img_thresh 
        
        min_area = (self.analysis_width * analysis_height) * self.min_contour_area_factor
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        debug_all_cnt = image_small.copy()
        cv2.drawContours(debug_all_cnt, contours, -1, (0, 0, 255), 2)
        self.debug_images['all_contours'] = debug_all_cnt

        biggest_contour = self._find_biggest_contour(contours, min_area)

        if biggest_contour is None:
            print("[Scanner] Không tìm thấy tài liệu. Trả về ảnh gốc.")
            return image_orig

        debug_found_cnt = image_small.copy()
        cv2.drawContours(debug_found_cnt, [biggest_contour], -1, (0, 255, 0), 3)
        self.debug_images['found_contour'] = debug_found_cnt

        biggest_contour_scaled = biggest_contour.astype(np.float32) * ratio
        pts1 = self._reorder(biggest_contour_scaled)

        (tl, tr, bl, br) = pts1
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))
        
        pts2 = np.array([
            [0, 0],
            [max_width - 1, 0],
            [0, max_height - 1],
            [max_width - 1, max_height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warped_high_res = cv2.warpPerspective(image_orig, matrix, (max_width, max_height))

        if apply_threshold:
            warped_gray = cv2.cvtColor(warped_high_res, cv2.COLOR_BGR2GRAY)
            final_image = cv2.adaptiveThreshold(warped_gray, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 21, 10)
            return final_image
        
        return warped_high_res