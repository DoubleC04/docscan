# import os
# import sys
# import datetime
# import cv2
# from PIL import Image
# import numpy as np
# import scipy.optimize

# class DocumentDewarper:
#     """
#     Làm phẳng bề mặt tài liệu bị cong hoặc gợn sóng.
#     """

#     def __init__(self):
#         self.PAGE_MARGIN_X = 20       # reduced px to ignore near L/R edge
#         self.PAGE_MARGIN_Y = 20       # reduced px to ignore near T/B edge

#         self.OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
#         self.OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
#         self.REMAP_DECIMATE = 16      # downscaling factor for remapping image

#         self.ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

#         self.TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
#         self.TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
#         self.TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
#         self.TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour

#         self.EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
#         self.EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
#         self.EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
#         self.EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

#         self.RVEC_IDX = slice(0, 3)   # index of rvec in params vector
#         self.TVEC_IDX = slice(3, 6)   # index of tvec in params vector
#         self.CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

#         self.SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
#         self.SPAN_PX_PER_STEP = 20    # reduced px spacing for sampling along spans
#         self.FOCAL_LENGTH = 1.2       # normalized focal length of camera

#         self.NO_BINARY = 0

#         self.DEBUG_LEVEL = 0          # 0=none, 1=some, 2=lots, 3=all
#         self.DEBUG_OUTPUT = 'file'    # file, screen, both

#         self.WINDOW_NAME = 'Dewarp'   # Window name for visualization
        
#         self.debug_images = {}
#         self.force_debug = False # Bật lên khi cần lấy ảnh debug để show

#         # nice color palette for visualizing contours, etc.
#         self.CCOLORS = [
#             (255, 0, 0),
#             (255, 63, 0),
#             (255, 127, 0),
#             (255, 191, 0),
#             (255, 255, 0),
#             (191, 255, 0),
#             (127, 255, 0),
#             (63, 255, 0),
#             (0, 255, 0),
#             (0, 255, 63),
#             (0, 255, 127),
#             (0, 255, 191),
#             (0, 255, 255),
#             (0, 191, 255),
#             (0, 127, 255),
#             (0, 63, 255),
#             (0, 0, 255),
#             (63, 0, 255),
#             (127, 0, 255),
#             (191, 0, 255),
#             (255, 0, 255),
#             (255, 0, 191),
#             (255, 0, 127),
#             (255, 0, 63),
#         ]

#         # default intrinsic parameter matrix
#         self.K = np.array([
#             [self.FOCAL_LENGTH, 0, 0],
#             [0, self.FOCAL_LENGTH, 0],
#             [0, 0, 1]], dtype=np.float32)

#     def debug_show(self, name, step, text, display):

#         key_name = text.replace(' ', '_')
#         self.debug_images[key_name] = display.copy()

#         # Logic cũ (lưu file hoặc show window) chỉ chạy nếu cấu hình
#         if self.DEBUG_OUTPUT != 'screen' and self.DEBUG_LEVEL > 0:
#             filetext = text.replace(' ', '_')
#             outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
#             # cv2.imwrite(outfile, display) # Tạm comment để không rác ổ cứng

#         if self.DEBUG_OUTPUT != 'file' and self.DEBUG_LEVEL > 0:
#             image = display.copy()
#             height = image.shape[0]
#             cv2.putText(image, text, (16, height-16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
#             cv2.putText(image, text, (16, height-16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
#             cv2.imshow(self.WINDOW_NAME, image)
#             while cv2.waitKey(5) < 0:
#                 pass


#     def round_nearest_multiple(self, i, factor):
#         i = int(i)
#         rem = i % factor
#         return i + factor - rem if rem else i


#     def pix2norm(self, shape, pts):
#         height, width = shape[:2]
#         scl = 2.0/(max(height, width))
#         offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5
#         return (pts - offset) * scl


#     def norm2pix(self, shape, pts, as_integer):
#         height, width = shape[:2]
#         scl = max(height, width)*0.5
#         offset = np.array([0.5*width, 0.5*height],
#                         dtype=pts.dtype).reshape((-1, 1, 2))
#         rval = pts * scl + offset
        
#         return (rval + 0.5).astype(int) if as_integer else rval


#     def fltp(self, point):
#         return tuple(point.astype(int).flatten())


#     def draw_correspondences(self, img, dstpoints, projpts):

#         display = img.copy()
#         dstpoints = self.norm2pix(img.shape, dstpoints, True)
#         projpts = self.norm2pix(img.shape, projpts, True)

#         for pts, color in [(projpts, (255, 0, 0)),
#                         (dstpoints, (0, 0, 255))]:

#             for point in pts:
#                 cv2.circle(display, self.fltp(point), 3, color, -1, cv2.LINE_AA)

#         for point_a, point_b in zip(projpts, dstpoints):
#             cv2.line(display, self.fltp(point_a), self.fltp(point_b),
#                     (255, 255, 255), 1, cv2.LINE_AA)

#         return display


#     def get_default_params(self, corners, ycoords, xcoords):

#         # page width and height
#         page_width, page_height = (np.linalg.norm(corners[i] - corners[0]) for i in (1, -1))

#         # our initial guess for the cubic has no slope
#         cubic_slopes = [0.0, 0.0]

#         # object points of flat page in 3D coordinates
#         corners_object3d = np.array([
#             [0, 0, 0],
#             [page_width, 0, 0],
#             [page_width, page_height, 0],
#             [0, page_height, 0]])

#         # estimate rotation and translation from four 2D-to-3D point
#         # correspondences
#         _, rvec, tvec = cv2.solvePnP(corners_object3d,
#                                     corners, self.K, np.zeros(5))

#         span_counts = [len(xc) for xc in xcoords]

#         params = np.hstack((np.array(rvec).flatten(),
#                             np.array(tvec).flatten(),
#                             np.array(cubic_slopes).flatten(),
#                             ycoords.flatten()) +
#                         tuple(xcoords))

#         return (page_width, page_height), span_counts, params


#     def project_xy(self, xy_coords, pvec):

#         # get cubic polynomial coefficients given
#         #
#         #  f(0) = 0, f'(0) = alpha
#         #  f(1) = 0, f'(1) = beta

#         alpha, beta = tuple(pvec[self.CUBIC_IDX])
        
#         alpha = np.clip(alpha, -0.5, 0.5)
#         beta = np.clip(beta, -0.5, 0.5)

#         poly = np.array([
#             alpha + beta,
#             -2*alpha - beta,
#             alpha,
#             0])

#         xy_coords = xy_coords.reshape((-1, 2))
#         z_coords = np.polyval(poly, xy_coords[:, 0])

#         objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

#         image_points, _ = cv2.projectPoints(objpoints,
#                                             pvec[self.RVEC_IDX],
#                                             pvec[self.TVEC_IDX],
#                                             self.K, np.zeros(5))

#         return image_points


#     def project_keypoints(self, pvec, keypoint_index):

#         xy_coords = pvec[keypoint_index]
#         xy_coords[0, :] = 0

#         return self.project_xy(xy_coords, pvec)


#     def resize_to_screen(self, src, maxw=1280, maxh=700, copy=False):

#         height, width = src.shape[:2]

#         scl_x = float(width) / maxw
#         scl_y = float(height) / maxh

#         scl = int(np.ceil(max(scl_x, scl_y)))

#         if scl > 1.0:
#             inv_scl = 1.0 / scl
#             img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
#         elif copy:
#             img = src.copy()
#         else:
#             img = src

#         return img


#     def box(self, width, height):
#         return np.ones((height, width), dtype=np.uint8)


#     def get_page_extents(self, small):

#         height, width = small.shape[:2]

#         xmin = self.PAGE_MARGIN_X
#         ymin = self.PAGE_MARGIN_Y
#         xmax = width - self.PAGE_MARGIN_X
#         ymax = height - self.PAGE_MARGIN_Y

#         page = np.zeros((height, width), dtype=np.uint8)
#         cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

#         outline = np.array([
#             [xmin, ymin],
#             [xmin, ymax],
#             [xmax, ymax],
#             [xmax, ymin]])

#         return page, outline


#     def get_mask(self, name, small, pagemask, masktype):

#         sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

#         if masktype == 'text':

#             mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                         cv2.THRESH_BINARY_INV,
#                                         self.ADAPTIVE_WINSZ,
#                                         25)

#             if self.DEBUG_LEVEL >= 3 or self.force_debug:
#                 self.debug_show(name, 0.1, 'thresholded_text', mask)

#             mask = cv2.dilate(mask, self.box(9, 1))

#             if self.DEBUG_LEVEL >= 3 or self.force_debug:
#                 self.debug_show(name, 0.2, 'dilated_text', mask)

#             mask = cv2.erode(mask, self.box(1, 3))

#             if self.DEBUG_LEVEL >= 3 or self.force_debug:
#                 self.debug_show(name, 0.3, 'eroded_text', mask)

#         else:

#             mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                         cv2.THRESH_BINARY_INV,
#                                         self.ADAPTIVE_WINSZ,
#                                         7)

#             if self.DEBUG_LEVEL >= 3 or self.force_debug:
#                 self.debug_show(name, 0.4, 'thresholded_line', mask)

#             mask = cv2.erode(mask, self.box(3, 1), iterations=3)

#             if self.DEBUG_LEVEL >= 3 or self.force_debug:
#                 self.debug_show(name, 0.5, 'eroded_line', mask)

#             mask = cv2.dilate(mask, self.box(8, 2))

#             if self.DEBUG_LEVEL >= 3 or self.force_debug:
#                 self.debug_show(name, 0.6, 'dilated_line', mask)

#         return np.minimum(mask, pagemask)


#     def interval_measure_overlap(self, int_a, int_b):
#         return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


#     def angle_dist(self, angle_b, angle_a):

#         diff = angle_b - angle_a

#         while diff > np.pi:
#             diff -= 2*np.pi

#         while diff < -np.pi:
#             diff += 2*np.pi

#         return np.abs(diff)


#     def blob_mean_and_tangent(self, contour):

#         moments = cv2.moments(contour)

#         area = moments['m00']

#         mean_x = moments['m10'] / area
#         mean_y = moments['m01'] / area

#         moments_matrix = np.array([
#             [moments['mu20'], moments['mu11']],
#             [moments['mu11'], moments['mu02']]
#         ]) / area

#         _, svd_u, _ = cv2.SVDecomp(moments_matrix)

#         center = np.array([mean_x, mean_y])
#         tangent = svd_u[:, 0].flatten().copy()

#         return center, tangent


#     class ContourInfo(object):

#         def __init__(self, contour, rect, mask):

#             self.contour = contour
#             self.rect = rect
#             self.mask = mask

#             self.center, self.tangent = DocumentDewarper().blob_mean_and_tangent(contour=contour)

#             self.angle = np.arctan2(self.tangent[1], self.tangent[0])

#             clx = [self.proj_x(point) for point in contour]

#             lxmin = min(clx)
#             lxmax = max(clx)

#             self.local_xrng = (lxmin, lxmax)

#             self.point0 = self.center + self.tangent * lxmin
#             self.point1 = self.center + self.tangent * lxmax

#             self.pred = None
#             self.succ = None

#         def proj_x(self, point):
#             return np.dot(self.tangent, point.flatten()-self.center)

#         def local_overlap(self, other):
#             xmin = self.proj_x(other.point0)
#             xmax = self.proj_x(other.point1)
#             return DocumentDewarper().interval_measure_overlap(self.local_xrng, (xmin, xmax))


#     def generate_candidate_edge(self, cinfo_a, cinfo_b):

#         # we want a left of b (so a's successor will be b and b's
#         # predecessor will be a) make sure right endpoint of b is to the
#         # right of left endpoint of a.
#         if cinfo_a.point0[0] > cinfo_b.point1[0]:
#             tmp = cinfo_a
#             cinfo_a = cinfo_b
#             cinfo_b = tmp

#         x_overlap_a = cinfo_a.local_overlap(cinfo_b)
#         x_overlap_b = cinfo_b.local_overlap(cinfo_a)

#         overall_tangent = cinfo_b.center - cinfo_a.center
#         overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

#         delta_angle = max(self.angle_dist(cinfo_a.angle, overall_angle),
#                         self.angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi

#         # we want the largest overlap in x to be small
#         x_overlap = max(x_overlap_a, x_overlap_b)

#         dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

#         if (dist > self.EDGE_MAX_LENGTH or
#                 x_overlap > self.EDGE_MAX_OVERLAP or
#                 delta_angle > self.EDGE_MAX_ANGLE):
#             return None
#         else:
#             score = dist + delta_angle * self.EDGE_ANGLE_COST
#             return (score, cinfo_a, cinfo_b)


#     def make_tight_mask(self, contour, xmin, ymin, width, height):

#         tight_mask = np.zeros((height, width), dtype=np.uint8)
#         tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

#         cv2.drawContours(tight_mask, [tight_contour], 0,
#                         (1, 1, 1), -1)

#         return tight_mask


#     def get_contours(self, name, small, pagemask, masktype):

#         mask = self.get_mask(name, small, pagemask, masktype)

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
#                                         cv2.CHAIN_APPROX_NONE)

#         contours_out = []

#         for contour in contours:

#             rect = cv2.boundingRect(contour)
#             xmin, ymin, width, height = rect

#             if (width < self.TEXT_MIN_WIDTH or
#                     height < self.TEXT_MIN_HEIGHT or
#                     width < self.TEXT_MIN_ASPECT*height):
#                 continue

#             tight_mask = self.make_tight_mask(contour, xmin, ymin, width, height)

#             if tight_mask.sum(axis=0).max() > self.TEXT_MAX_THICKNESS:
#                 continue

#             contours_out.append(self.ContourInfo(contour, rect, tight_mask))

#         if self.DEBUG_LEVEL >= 2 or self.force_debug:
#             self.visualize_contours(name, small, contours_out)

#         return contours_out


#     def assemble_spans(self, name, small, pagemask, cinfo_list):

#         # sort list
#         cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])

#         # generate all candidate edges
#         candidate_edges = []

#         for i, cinfo_i in enumerate(cinfo_list):
#             for j in range(i):
#                 # note e is of the form (score, left_cinfo, right_cinfo)
#                 edge = self.generate_candidate_edge(cinfo_i, cinfo_list[j])
#                 if edge is not None:
#                     candidate_edges.append(edge)

#         # sort candidate edges by score (lower is better)
#         candidate_edges.sort()

#         # for each candidate edge
#         for _, cinfo_a, cinfo_b in candidate_edges:
#             # if left and right are unassigned, join them
#             if cinfo_a.succ is None and cinfo_b.pred is None:
#                 cinfo_a.succ = cinfo_b
#                 cinfo_b.pred = cinfo_a

#         # generate list of spans as output
#         spans = []

#         # until we have removed everything from the list
#         while cinfo_list:

#             # get the first on the list
#             cinfo = cinfo_list[0]

#             # keep following predecessors until none exists
#             while cinfo.pred:
#                 cinfo = cinfo.pred

#             # start a new span
#             cur_span = []

#             width = 0.0

#             # follow successors til end of span
#             while cinfo:
#                 # remove from list (sadly making this loop *also* O(n^2)
#                 cinfo_list.remove(cinfo)
#                 # add to span
#                 cur_span.append(cinfo)
#                 width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
#                 # set successor
#                 cinfo = cinfo.succ

#             # add if long enough
#             if width > self.SPAN_MIN_WIDTH:
#                 spans.append(cur_span)

#         if self.DEBUG_LEVEL >= 2 or self.force_debug:
#             self.visualize_spans(name, small, pagemask, spans)

#         return spans


#     def sample_spans(self, shape, spans):

#         span_points = []

#         for span in spans:

#             contour_points = []

#             for cinfo in span:

#                 yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
#                 totals = (yvals * cinfo.mask).sum(axis=0)
#                 means = np.divide(totals, cinfo.mask.sum(axis=0))

#                 xmin, ymin = cinfo.rect[:2]

#                 step = self.SPAN_PX_PER_STEP
#                 start = np.floor_divide((np.mod((len(means) - 1), step)), 2)

#                 contour_points.extend(
#                     [(x + xmin, means[x] + ymin) for x in range(start, len(means), step)],
#                 )

#             contour_points = np.array(contour_points,
#                                     dtype=np.float32).reshape((-1, 1, 2))

#             contour_points = self.pix2norm(shape, contour_points)

#             span_points.append(contour_points)

#         return span_points


#     def keypoints_from_samples(self, name, small, pagemask, page_outline,
#                             span_points):

#         all_evecs = np.array([[0.0, 0.0]])
#         all_weights = 0

#         for points in span_points:

#             _, evec = cv2.PCACompute(points.reshape((-1, 2)),
#                                     None, maxComponents=1)

#             weight = np.linalg.norm(points[-1] - points[0])

#             all_evecs += evec * weight
#             all_weights += weight

#         evec = all_evecs / all_weights

#         x_dir = evec.flatten()

#         if x_dir[0] < 0:
#             x_dir = -x_dir

#         y_dir = np.array([-x_dir[1], x_dir[0]])

#         pagecoords = cv2.convexHull(page_outline)
#         pagecoords = self.pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
#         pagecoords = pagecoords.reshape((-1, 2))

#         px_coords = np.dot(pagecoords, x_dir)
#         py_coords = np.dot(pagecoords, y_dir)

#         px0 = px_coords.min()
#         px1 = px_coords.max()

#         py0 = py_coords.min()
#         py1 = py_coords.max()

#         p00 = px0 * x_dir + py0 * y_dir
#         p10 = px1 * x_dir + py0 * y_dir
#         p11 = px1 * x_dir + py1 * y_dir
#         p01 = px0 * x_dir + py1 * y_dir

#         corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

#         ycoords = []
#         xcoords = []

#         for points in span_points:
#             pts = points.reshape((-1, 2))
#             px_coords = np.dot(pts, x_dir)
#             py_coords = np.dot(pts, y_dir)
#             ycoords.append(py_coords.mean() - py0)
#             xcoords.append(px_coords - px0)

#         if self.DEBUG_LEVEL >= 2 or self.force_debug:
#             self.visualize_span_points(name, small, span_points, corners)

#         return corners, np.array(ycoords), xcoords


#     def visualize_contours(self, name, small, cinfo_list):

#         regions = np.zeros_like(small)

#         for j, cinfo in enumerate(cinfo_list):

#             cv2.drawContours(regions, [cinfo.contour], 0,
#                             self.CCOLORS[j % len(self.CCOLORS)], -1)

#         mask = (regions.max(axis=2) != 0)

#         display = small.copy()
#         display[mask] = (display[mask]/2) + (regions[mask]/2)

#         for j, cinfo in enumerate(cinfo_list):
#             color = self.CCOLORS[j % len(self.CCOLORS)]
#             color = tuple([c/4 for c in color])

#             cv2.circle(display, self.fltp(cinfo.center), 3,
#                     (255, 255, 255), 1, cv2.LINE_AA)

#             cv2.line(display, self.fltp(cinfo.point0), self.fltp(cinfo.point1),
#                     (255, 255, 255), 1, cv2.LINE_AA)

#         self.debug_show(name, 1, 'contours', display)


#     def visualize_spans(self, name, small, pagemask, spans):

#         regions = np.zeros_like(small)

#         for i, span in enumerate(spans):
#             contours = [cinfo.contour for cinfo in span]
#             cv2.drawContours(regions, contours, -1,
#                             self.CCOLORS[i*3 % len(self.CCOLORS)], -1)

#         mask = (regions.max(axis=2) != 0)

#         display = small.copy()
#         display[mask] = (display[mask]/2) + (regions[mask]/2)
#         display[pagemask == 0] /= 4

#         self.debug_show(name, 2, 'spans', display)


#     def visualize_span_points(self, name, small, span_points, corners):

#         display = small.copy()

#         for i, points in enumerate(span_points):

#             points = self.norm2pix(small.shape, points, False)

#             mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)),
#                                             None,
#                                             maxComponents=1)

#             dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
#             dpm = np.dot(mean.flatten(), small_evec.flatten())

#             point0 = mean + small_evec * (dps.min()-dpm)
#             point1 = mean + small_evec * (dps.max()-dpm)

#             for point in points:
#                 cv2.circle(display, self.fltp(point), 3,
#                         self.CCOLORS[i % len(self.CCOLORS)], -1, cv2.LINE_AA)

#             cv2.line(display, self.fltp(point0), self.fltp(point1),
#                     (255, 255, 255), 1, cv2.LINE_AA)

#         cv2.polylines(display, [self.norm2pix(small.shape, corners, True)],
#                     True, (255, 255, 255))

#         self.debug_show(name, 3, 'span_points', display)


#     def imgsize(self, img):
#         height, width = img.shape[:2]
#         return '{}x{}'.format(width, height)


#     def make_keypoint_index(self, span_counts):

#         nspans = len(span_counts)
#         npts = sum(span_counts)
#         keypoint_index = np.zeros((npts+1, 2), dtype=int)
#         start = 1

#         for i, count in enumerate(span_counts):
#             end = start + count
#             keypoint_index[start:start+end, 1] = 8+i
#             start = end

#         keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans

#         return keypoint_index


#     def optimize_params(self, name, small, dstpoints, span_counts, params):

#         keypoint_index = self.make_keypoint_index(span_counts)

#         def objective(pvec):
#             ppts = self.project_keypoints(pvec, keypoint_index)
#             return np.sum((dstpoints - ppts)**2)

#         print('  initial objective is', objective(params))

#         if self.DEBUG_LEVEL >= 1 or self.force_debug:
#             projpts = self.project_keypoints(params, keypoint_index)
#             display = self.draw_correspondences(small, dstpoints, projpts)
#             self.debug_show(name, 4, 'keypoints before', display)

#         print('  optimizing', len(params), 'parameters...')
#         start = datetime.datetime.now()
#         res = scipy.optimize.minimize(objective, params,
#                                     method='Powell')
#         end = datetime.datetime.now()
#         print('  optimization took', round((end-start).total_seconds(), 2), 'sec.')
#         print('  final objective is', res.fun)
#         params = res.x

#         if self.DEBUG_LEVEL >= 1 or self.force_debug:
#             projpts = self.project_keypoints(params, keypoint_index)
#             display = self.draw_correspondences(small, dstpoints, projpts)
#             self.debug_show(name, 5, 'keypoints after', display)

#         return params


#     def get_page_dims(self, corners, rough_dims, params):

#         dst_br = corners[2].flatten()

#         dims = np.array(rough_dims)

#         def objective(dims):
#             proj_br = self.project_xy(dims, params)
#             return np.sum((dst_br - proj_br.flatten())**2)

#         res = scipy.optimize.minimize(objective, dims, method='Powell')
#         dims = res.x

#         print('  got page dims', dims[0], 'x', dims[1])

#         return dims


#     def remap_image(self, name, img, small, page_dims, params):

#         height = 0.5 * page_dims[1] * self.OUTPUT_ZOOM * img.shape[0]
#         height = self.round_nearest_multiple(height, self.REMAP_DECIMATE)

#         width = self.round_nearest_multiple(height * page_dims[0] / page_dims[1],
#                                     self.REMAP_DECIMATE)

#         print('  output will be {}x{}'.format(width, height))

#         height_small, width_small = np.floor_divide(
#                 [height, width],
#                 self.REMAP_DECIMATE,
#             )

#         page_x_range = np.linspace(0, page_dims[0], width_small)
#         page_y_range = np.linspace(0, page_dims[1], height_small)

#         page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

#         page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
#                                     page_y_coords.flatten().reshape((-1, 1))))

#         page_xy_coords = page_xy_coords.astype(np.float32)

#         image_points = self.project_xy(page_xy_coords, params)
#         image_points = self.norm2pix(img.shape, image_points, False)

#         image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
#         image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

#         image_x_coords = cv2.resize(image_x_coords, (width, height),
#                                     interpolation=cv2.INTER_CUBIC).astype(np.float32)

#         image_y_coords = cv2.resize(image_y_coords, (width, height),
#                                     interpolation=cv2.INTER_CUBIC).astype(np.float32)

#         img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#         remapped = cv2.remap(img_gray, image_x_coords, image_y_coords,
#                             cv2.INTER_CUBIC,
#                             None, cv2.BORDER_REPLICATE)

#         if self.NO_BINARY:
#             thresh = remapped
#             pil_image = Image.fromarray(thresh)
#         else:
#             thresh = cv2.adaptiveThreshold(
#                 remapped,
#                 255,
#                 cv2.ADAPTIVE_THRESH_MEAN_C,
#                 cv2.THRESH_BINARY,
#                 self.ADAPTIVE_WINSZ,
#                 25,
#             )
        
#         self.debug_images['final_output'] = thresh
#         pil_image = Image.fromarray(thresh)
#         if not self.NO_BINARY:
#             pil_image = pil_image.convert("1")

#         threshfile = name + '_thresh.png'
#         pil_image.save(f"data/output_images/{threshfile}", dpi=(self.OUTPUT_DPI, self.OUTPUT_DPI))

#         if self.DEBUG_LEVEL >= 1:
#             height = small.shape[0]
#             width = int(round(height * float(thresh.shape[1])/thresh.shape[0]))
#             display = cv2.resize(thresh, (width, height),
#                                 interpolation=cv2.INTER_AREA)
#             self.debug_show(name, 6, 'output', display)

#         return thresh


#     def dewarp(self, image_input, return_debug=False):
#         """
#         Phương thức chính để làm phẳng ảnh.

#         Args:
#             image_input (str or np.ndarray): Đường dẫn tới ảnh hoặc ảnh dạng numpy array.
#             apply_threshold (bool): Áp dụng threshold để làm ảnh đen trắng.

#         Returns:
#             np.ndarray: Ảnh đã được làm phẳng hoặc ảnh gốc nếu xử lý thất bại.
#         """
#         self.debug_images = {}
#         self.force_debug = return_debug
        
#         if isinstance(image_input, str):
#             img = cv2.imread(image_input)
#             basename = os.path.basename(image_input)
#             name, _ = os.path.splitext(basename)
#         else:
#             img = image_input.copy()
#             name = "doc_image"

#         small = self.resize_to_screen(img)
#         self.debug_images['original'] = small

#         pagemask, page_outline = self.get_page_extents(small)

#         cinfo_list = self.get_contours(name, small, pagemask, 'text')
#         spans = self.assemble_spans(name, small, pagemask, cinfo_list)

#         if len(spans) < 3:
#             cinfo_list = self.get_contours(name, small, pagemask, 'line')
#             spans2 = self.assemble_spans(name, small, pagemask, cinfo_list)
#             if len(spans2) > len(spans):
#                 spans = spans2

#         if len(spans) < 1:
#             print('skipping', name, 'because only', len(spans), 'spans')
#             return img

#         span_points = self.sample_spans(small.shape, spans)

#         corners, ycoords, xcoords = self.keypoints_from_samples(name, small,
#                                                         pagemask,
#                                                         page_outline,
#                                                         span_points)

#         rough_dims, span_counts, params = self.get_default_params(corners,
#                                                             ycoords, xcoords)

#         dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) +
#                             tuple(span_points))

#         params = self.optimize_params(name, small,
#                                 dstpoints,
#                                 span_counts, params)

#         page_dims = self.get_page_dims(corners, rough_dims, params)

#         result_image = self.remap_image(name, img, small, page_dims, params)
        
#         return result_image































































































import os
import sys
import datetime
import cv2
from PIL import Image
import numpy as np
import scipy.optimize

class DocumentDewarper:
    """
    Làm phẳng bề mặt tài liệu bị cong hoặc gợn sóng.
    """

    def __init__(self):
        self.PAGE_MARGIN_X = 20       # reduced px to ignore near L/R edge
        self.PAGE_MARGIN_Y = 20       # reduced px to ignore near T/B edge

        self.OUTPUT_ZOOM = 1.0        # how much to zoom output relative to *original* image
        self.OUTPUT_DPI = 300         # just affects stated DPI of PNG, not appearance
        self.REMAP_DECIMATE = 16      # downscaling factor for remapping image

        self.ADAPTIVE_WINSZ = 55      # window size for adaptive threshold in reduced px

        self.TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
        self.TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
        self.TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
        self.TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour

        self.EDGE_MAX_OVERLAP = 1.0   # max reduced px horiz. overlap of contours in span
        self.EDGE_MAX_LENGTH = 100.0  # max reduced px length of edge connecting contours
        self.EDGE_ANGLE_COST = 10.0   # cost of angles in edges (tradeoff vs. length)
        self.EDGE_MAX_ANGLE = 7.5     # maximum change in angle allowed between contours

        self.RVEC_IDX = slice(0, 3)   # index of rvec in params vector
        self.TVEC_IDX = slice(3, 6)   # index of tvec in params vector
        self.CUBIC_IDX = slice(6, 8)  # index of cubic slopes in params vector

        self.SPAN_MIN_WIDTH = 30      # minimum reduced px width for span
        self.SPAN_PX_PER_STEP = 20    # reduced px spacing for sampling along spans
        self.FOCAL_LENGTH = 1.2       # normalized focal length of camera

        self.NO_BINARY = 0

        self.DEBUG_LEVEL = 3          # 0=none, 1=some, 2=lots, 3=all
        self.DEBUG_OUTPUT = 'screen'    # file, screen, both

        self.WINDOW_NAME = 'Dewarp'   # Window name for visualization

        # nice color palette for visualizing contours, etc.
        self.CCOLORS = [
            (255, 0, 0),
            (255, 63, 0),
            (255, 127, 0),
            (255, 191, 0),
            (255, 255, 0),
            (191, 255, 0),
            (127, 255, 0),
            (63, 255, 0),
            (0, 255, 0),
            (0, 255, 63),
            (0, 255, 127),
            (0, 255, 191),
            (0, 255, 255),
            (0, 191, 255),
            (0, 127, 255),
            (0, 63, 255),
            (0, 0, 255),
            (63, 0, 255),
            (127, 0, 255),
            (191, 0, 255),
            (255, 0, 255),
            (255, 0, 191),
            (255, 0, 127),
            (255, 0, 63),
        ]

        # default intrinsic parameter matrix
        self.K = np.array([
            [self.FOCAL_LENGTH, 0, 0],
            [0, self.FOCAL_LENGTH, 0],
            [0, 0, 1]], dtype=np.float32)

    def debug_show(self, name, step, text, display):

        if self.DEBUG_OUTPUT != 'screen':
            filetext = text.replace(' ', '_')
            outfile = name + '_debug_' + str(step) + '_' + filetext + '.png'
            cv2.imwrite(outfile, display)

        if self.DEBUG_OUTPUT != 'file':

            image = display.copy()
            height = image.shape[0]

            cv2.putText(image, text, (16, height-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 3, cv2.LINE_AA)

            cv2.putText(image, text, (16, height-16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(self.WINDOW_NAME, image)

            while cv2.waitKey(5) < 0:
                pass


    def round_nearest_multiple(self, i, factor):
        i = int(i)
        rem = i % factor
        return i + factor - rem if rem else i


    def pix2norm(self, shape, pts):
        height, width = shape[:2]
        scl = 2.0/(max(height, width))
        offset = np.array([width, height], dtype=pts.dtype).reshape((-1, 1, 2))*0.5
        return (pts - offset) * scl


    def norm2pix(self, shape, pts, as_integer):
        height, width = shape[:2]
        scl = max(height, width)*0.5
        offset = np.array([0.5*width, 0.5*height],
                        dtype=pts.dtype).reshape((-1, 1, 2))
        rval = pts * scl + offset
        
        return (rval + 0.5).astype(int) if as_integer else rval


    def fltp(self, point):
        return tuple(point.astype(int).flatten())


    def draw_correspondences(self, img, dstpoints, projpts):

        display = img.copy()
        dstpoints = self.norm2pix(img.shape, dstpoints, True)
        projpts = self.norm2pix(img.shape, projpts, True)

        for pts, color in [(projpts, (255, 0, 0)),
                        (dstpoints, (0, 0, 255))]:

            for point in pts:
                cv2.circle(display, self.fltp(point), 3, color, -1, cv2.LINE_AA)

        for point_a, point_b in zip(projpts, dstpoints):
            cv2.line(display, self.fltp(point_a), self.fltp(point_b),
                    (255, 255, 255), 1, cv2.LINE_AA)

        return display


    def get_default_params(self, corners, ycoords, xcoords):

        # page width and height
        page_width, page_height = (np.linalg.norm(corners[i] - corners[0]) for i in (1, -1))

        # our initial guess for the cubic has no slope
        cubic_slopes = [0.0, 0.0]

        # object points of flat page in 3D coordinates
        corners_object3d = np.array([
            [0, 0, 0],
            [page_width, 0, 0],
            [page_width, page_height, 0],
            [0, page_height, 0]])

        # estimate rotation and translation from four 2D-to-3D point
        # correspondences
        _, rvec, tvec = cv2.solvePnP(corners_object3d,
                                    corners, self.K, np.zeros(5))

        span_counts = [len(xc) for xc in xcoords]

        params = np.hstack((np.array(rvec).flatten(),
                            np.array(tvec).flatten(),
                            np.array(cubic_slopes).flatten(),
                            ycoords.flatten()) +
                        tuple(xcoords))

        return (page_width, page_height), span_counts, params


    def project_xy(self, xy_coords, pvec):

        # get cubic polynomial coefficients given
        #
        #  f(0) = 0, f'(0) = alpha
        #  f(1) = 0, f'(1) = beta

        alpha, beta = tuple(pvec[self.CUBIC_IDX])
        
        alpha = np.clip(alpha, -0.5, 0.5)
        beta = np.clip(beta, -0.5, 0.5)

        poly = np.array([
            alpha + beta,
            -2*alpha - beta,
            alpha,
            0])

        xy_coords = xy_coords.reshape((-1, 2))
        z_coords = np.polyval(poly, xy_coords[:, 0])

        objpoints = np.hstack((xy_coords, z_coords.reshape((-1, 1))))

        image_points, _ = cv2.projectPoints(objpoints,
                                            pvec[self.RVEC_IDX],
                                            pvec[self.TVEC_IDX],
                                            self.K, np.zeros(5))

        return image_points


    def project_keypoints(self, pvec, keypoint_index):

        xy_coords = pvec[keypoint_index]
        xy_coords[0, :] = 0

        return self.project_xy(xy_coords, pvec)


    def resize_to_screen(self, src, maxw=1280, maxh=700, copy=False):

        height, width = src.shape[:2]

        scl_x = float(width) / maxw
        scl_y = float(height) / maxh

        scl = int(np.ceil(max(scl_x, scl_y)))

        if scl > 1.0:
            inv_scl = 1.0 / scl
            img = cv2.resize(src, (0, 0), None, inv_scl, inv_scl, cv2.INTER_AREA)
        elif copy:
            img = src.copy()
        else:
            img = src

        return img


    def box(self, width, height):
        return np.ones((height, width), dtype=np.uint8)


    def get_page_extents(self, small):

        height, width = small.shape[:2]

        xmin = self.PAGE_MARGIN_X
        ymin = self.PAGE_MARGIN_Y
        xmax = width - self.PAGE_MARGIN_X
        ymax = height - self.PAGE_MARGIN_Y

        page = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(page, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)

        outline = np.array([
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymax],
            [xmax, ymin]])

        return page, outline


    def get_mask(self, name, small, pagemask, masktype):

        sgray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        if masktype == 'text':

            mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        self.ADAPTIVE_WINSZ,
                                        25)

            if self.DEBUG_LEVEL >= 3:
                self.debug_show(name, 0.1, 'thresholded', mask)

            mask = cv2.dilate(mask, self.box(9, 1))

            if self.DEBUG_LEVEL >= 3:
                self.debug_show(name, 0.2, 'dilated', mask)

            mask = cv2.erode(mask, self.box(1, 3))

            if self.DEBUG_LEVEL >= 3:
                self.debug_show(name, 0.3, 'eroded', mask)

        else:

            mask = cv2.adaptiveThreshold(sgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV,
                                        self.ADAPTIVE_WINSZ,
                                        7)

            if self.DEBUG_LEVEL >= 3:
                self.debug_show(name, 0.4, 'thresholded', mask)

            mask = cv2.erode(mask, self.box(3, 1), iterations=3)

            if self.DEBUG_LEVEL >= 3:
                self.debug_show(name, 0.5, 'eroded', mask)

            mask = cv2.dilate(mask, self.box(8, 2))

            if self.DEBUG_LEVEL >= 3:
                self.debug_show(name, 0.6, 'dilated', mask)

        return np.minimum(mask, pagemask)


    def interval_measure_overlap(self, int_a, int_b):
        return min(int_a[1], int_b[1]) - max(int_a[0], int_b[0])


    def angle_dist(self, angle_b, angle_a):

        diff = angle_b - angle_a

        while diff > np.pi:
            diff -= 2*np.pi

        while diff < -np.pi:
            diff += 2*np.pi

        return np.abs(diff)


    def blob_mean_and_tangent(self, contour):

        moments = cv2.moments(contour)

        area = moments['m00']

        mean_x = moments['m10'] / area
        mean_y = moments['m01'] / area

        moments_matrix = np.array([
            [moments['mu20'], moments['mu11']],
            [moments['mu11'], moments['mu02']]
        ]) / area

        _, svd_u, _ = cv2.SVDecomp(moments_matrix)

        center = np.array([mean_x, mean_y])
        tangent = svd_u[:, 0].flatten().copy()

        return center, tangent


    class ContourInfo(object):

        def __init__(self, contour, rect, mask):

            self.contour = contour
            self.rect = rect
            self.mask = mask

            self.center, self.tangent = DocumentDewarper().blob_mean_and_tangent(contour=contour)

            self.angle = np.arctan2(self.tangent[1], self.tangent[0])

            clx = [self.proj_x(point) for point in contour]

            lxmin = min(clx)
            lxmax = max(clx)

            self.local_xrng = (lxmin, lxmax)

            self.point0 = self.center + self.tangent * lxmin
            self.point1 = self.center + self.tangent * lxmax

            self.pred = None
            self.succ = None

        def proj_x(self, point):
            return np.dot(self.tangent, point.flatten()-self.center)

        def local_overlap(self, other):
            xmin = self.proj_x(other.point0)
            xmax = self.proj_x(other.point1)
            return DocumentDewarper().interval_measure_overlap(self.local_xrng, (xmin, xmax))


    def generate_candidate_edge(self, cinfo_a, cinfo_b):

        # we want a left of b (so a's successor will be b and b's
        # predecessor will be a) make sure right endpoint of b is to the
        # right of left endpoint of a.
        if cinfo_a.point0[0] > cinfo_b.point1[0]:
            tmp = cinfo_a
            cinfo_a = cinfo_b
            cinfo_b = tmp

        x_overlap_a = cinfo_a.local_overlap(cinfo_b)
        x_overlap_b = cinfo_b.local_overlap(cinfo_a)

        overall_tangent = cinfo_b.center - cinfo_a.center
        overall_angle = np.arctan2(overall_tangent[1], overall_tangent[0])

        delta_angle = max(self.angle_dist(cinfo_a.angle, overall_angle),
                        self.angle_dist(cinfo_b.angle, overall_angle)) * 180/np.pi

        # we want the largest overlap in x to be small
        x_overlap = max(x_overlap_a, x_overlap_b)

        dist = np.linalg.norm(cinfo_b.point0 - cinfo_a.point1)

        if (dist > self.EDGE_MAX_LENGTH or
                x_overlap > self.EDGE_MAX_OVERLAP or
                delta_angle > self.EDGE_MAX_ANGLE):
            return None
        else:
            score = dist + delta_angle * self.EDGE_ANGLE_COST
            return (score, cinfo_a, cinfo_b)


    def make_tight_mask(self, contour, xmin, ymin, width, height):

        tight_mask = np.zeros((height, width), dtype=np.uint8)
        tight_contour = contour - np.array((xmin, ymin)).reshape((-1, 1, 2))

        cv2.drawContours(tight_mask, [tight_contour], 0,
                        (1, 1, 1), -1)

        return tight_mask


    def get_contours(self, name, small, pagemask, masktype):

        mask = self.get_mask(name, small, pagemask, masktype)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)

        contours_out = []

        for contour in contours:

            rect = cv2.boundingRect(contour)
            xmin, ymin, width, height = rect

            if (width < self.TEXT_MIN_WIDTH or
                    height < self.TEXT_MIN_HEIGHT or
                    width < self.TEXT_MIN_ASPECT*height):
                continue

            tight_mask = self.make_tight_mask(contour, xmin, ymin, width, height)

            if tight_mask.sum(axis=0).max() > self.TEXT_MAX_THICKNESS:
                continue

            contours_out.append(self.ContourInfo(contour, rect, tight_mask))

        if self.DEBUG_LEVEL >= 2:
            self.visualize_contours(name, small, contours_out)

        return contours_out


    def assemble_spans(self, name, small, pagemask, cinfo_list):

        # sort list
        cinfo_list = sorted(cinfo_list, key=lambda cinfo: cinfo.rect[1])

        # generate all candidate edges
        candidate_edges = []

        for i, cinfo_i in enumerate(cinfo_list):
            for j in range(i):
                # note e is of the form (score, left_cinfo, right_cinfo)
                edge = self.generate_candidate_edge(cinfo_i, cinfo_list[j])
                if edge is not None:
                    candidate_edges.append(edge)

        # sort candidate edges by score (lower is better)
        candidate_edges.sort()

        # for each candidate edge
        for _, cinfo_a, cinfo_b in candidate_edges:
            # if left and right are unassigned, join them
            if cinfo_a.succ is None and cinfo_b.pred is None:
                cinfo_a.succ = cinfo_b
                cinfo_b.pred = cinfo_a

        # generate list of spans as output
        spans = []

        # until we have removed everything from the list
        while cinfo_list:

            # get the first on the list
            cinfo = cinfo_list[0]

            # keep following predecessors until none exists
            while cinfo.pred:
                cinfo = cinfo.pred

            # start a new span
            cur_span = []

            width = 0.0

            # follow successors til end of span
            while cinfo:
                # remove from list (sadly making this loop *also* O(n^2)
                cinfo_list.remove(cinfo)
                # add to span
                cur_span.append(cinfo)
                width += cinfo.local_xrng[1] - cinfo.local_xrng[0]
                # set successor
                cinfo = cinfo.succ

            # add if long enough
            if width > self.SPAN_MIN_WIDTH:
                spans.append(cur_span)

        if self.DEBUG_LEVEL >= 2:
            self.visualize_spans(name, small, pagemask, spans)

        return spans


    def sample_spans(self, shape, spans):

        span_points = []

        for span in spans:

            contour_points = []

            for cinfo in span:

                yvals = np.arange(cinfo.mask.shape[0]).reshape((-1, 1))
                totals = (yvals * cinfo.mask).sum(axis=0)
                means = np.divide(totals, cinfo.mask.sum(axis=0))

                xmin, ymin = cinfo.rect[:2]

                step = self.SPAN_PX_PER_STEP
                start = np.floor_divide((np.mod((len(means) - 1), step)), 2)

                contour_points.extend(
                    [(x + xmin, means[x] + ymin) for x in range(start, len(means), step)],
                )

            contour_points = np.array(contour_points,
                                    dtype=np.float32).reshape((-1, 1, 2))

            contour_points = self.pix2norm(shape, contour_points)

            span_points.append(contour_points)

        return span_points


    def keypoints_from_samples(self, name, small, pagemask, page_outline,
                            span_points):

        all_evecs = np.array([[0.0, 0.0]])
        all_weights = 0

        for points in span_points:

            _, evec = cv2.PCACompute(points.reshape((-1, 2)),
                                    None, maxComponents=1)

            weight = np.linalg.norm(points[-1] - points[0])

            all_evecs += evec * weight
            all_weights += weight

        evec = all_evecs / all_weights

        x_dir = evec.flatten()

        if x_dir[0] < 0:
            x_dir = -x_dir

        y_dir = np.array([-x_dir[1], x_dir[0]])

        pagecoords = cv2.convexHull(page_outline)
        pagecoords = self.pix2norm(pagemask.shape, pagecoords.reshape((-1, 1, 2)))
        pagecoords = pagecoords.reshape((-1, 2))

        px_coords = np.dot(pagecoords, x_dir)
        py_coords = np.dot(pagecoords, y_dir)

        px0 = px_coords.min()
        px1 = px_coords.max()

        py0 = py_coords.min()
        py1 = py_coords.max()

        p00 = px0 * x_dir + py0 * y_dir
        p10 = px1 * x_dir + py0 * y_dir
        p11 = px1 * x_dir + py1 * y_dir
        p01 = px0 * x_dir + py1 * y_dir

        corners = np.vstack((p00, p10, p11, p01)).reshape((-1, 1, 2))

        ycoords = []
        xcoords = []

        for points in span_points:
            pts = points.reshape((-1, 2))
            px_coords = np.dot(pts, x_dir)
            py_coords = np.dot(pts, y_dir)
            ycoords.append(py_coords.mean() - py0)
            xcoords.append(px_coords - px0)

        if self.DEBUG_LEVEL >= 2:
            self.visualize_span_points(name, small, span_points, corners)

        return corners, np.array(ycoords), xcoords


    def visualize_contours(self, name, small, cinfo_list):

        regions = np.zeros_like(small)

        for j, cinfo in enumerate(cinfo_list):

            cv2.drawContours(regions, [cinfo.contour], 0,
                            self.CCOLORS[j % len(self.CCOLORS)], -1)

        mask = (regions.max(axis=2) != 0)

        display = small.copy()
        display[mask] = (display[mask]/2) + (regions[mask]/2)

        for j, cinfo in enumerate(cinfo_list):
            color = self.CCOLORS[j % len(self.CCOLORS)]
            color = tuple([c/4 for c in color])

            cv2.circle(display, self.fltp(cinfo.center), 3,
                    (255, 255, 255), 1, cv2.LINE_AA)

            cv2.line(display, self.fltp(cinfo.point0), self.fltp(cinfo.point1),
                    (255, 255, 255), 1, cv2.LINE_AA)

        self.debug_show(name, 1, 'contours', display)


    def visualize_spans(self, name, small, pagemask, spans):

        regions = np.zeros_like(small)

        for i, span in enumerate(spans):
            contours = [cinfo.contour for cinfo in span]
            cv2.drawContours(regions, contours, -1,
                            self.CCOLORS[i*3 % len(self.CCOLORS)], -1)

        mask = (regions.max(axis=2) != 0)

        display = small.copy()
        display[mask] = (display[mask]/2) + (regions[mask]/2)
        display[pagemask == 0] /= 4

        self.debug_show(name, 2, 'spans', display)


    def visualize_span_points(self, name, small, span_points, corners):

        display = small.copy()

        for i, points in enumerate(span_points):

            points = self.norm2pix(small.shape, points, False)

            mean, small_evec = cv2.PCACompute(points.reshape((-1, 2)),
                                            None,
                                            maxComponents=1)

            dps = np.dot(points.reshape((-1, 2)), small_evec.reshape((2, 1)))
            dpm = np.dot(mean.flatten(), small_evec.flatten())

            point0 = mean + small_evec * (dps.min()-dpm)
            point1 = mean + small_evec * (dps.max()-dpm)

            for point in points:
                cv2.circle(display, self.fltp(point), 3,
                        self.CCOLORS[i % len(self.CCOLORS)], -1, cv2.LINE_AA)

            cv2.line(display, self.fltp(point0), self.fltp(point1),
                    (255, 255, 255), 1, cv2.LINE_AA)

        cv2.polylines(display, [self.norm2pix(small.shape, corners, True)],
                    True, (255, 255, 255))

        self.debug_show(name, 3, 'span points', display)


    def imgsize(self, img):
        height, width = img.shape[:2]
        return '{}x{}'.format(width, height)


    def make_keypoint_index(self, span_counts):

        nspans = len(span_counts)
        npts = sum(span_counts)
        keypoint_index = np.zeros((npts+1, 2), dtype=int)
        start = 1

        for i, count in enumerate(span_counts):
            end = start + count
            keypoint_index[start:start+end, 1] = 8+i
            start = end

        keypoint_index[1:, 0] = np.arange(npts) + 8 + nspans

        return keypoint_index


    def optimize_params(self, name, small, dstpoints, span_counts, params):

        keypoint_index = self.make_keypoint_index(span_counts)

        def objective(pvec):
            ppts = self.project_keypoints(pvec, keypoint_index)
            return np.sum((dstpoints - ppts)**2)

        print('  initial objective is', objective(params))

        if self.DEBUG_LEVEL >= 1:
            projpts = self.project_keypoints(params, keypoint_index)
            display = self.draw_correspondences(small, dstpoints, projpts)
            self.debug_show(name, 4, 'keypoints before', display)

        print('  optimizing', len(params), 'parameters...')
        start = datetime.datetime.now()
        res = scipy.optimize.minimize(objective, params,
                                    method='Powell')
        end = datetime.datetime.now()
        print('  optimization took', round((end-start).total_seconds(), 2), 'sec.')
        print('  final objective is', res.fun)
        params = res.x

        if self.DEBUG_LEVEL >= 1:
            projpts = self.project_keypoints(params, keypoint_index)
            display = self.draw_correspondences(small, dstpoints, projpts)
            self.debug_show(name, 5, 'keypoints after', display)

        return params


    def get_page_dims(self, corners, rough_dims, params):

        dst_br = corners[2].flatten()

        dims = np.array(rough_dims)

        def objective(dims):
            proj_br = self.project_xy(dims, params)
            return np.sum((dst_br - proj_br.flatten())**2)

        res = scipy.optimize.minimize(objective, dims, method='Powell')
        dims = res.x

        print('  got page dims', dims[0], 'x', dims[1])

        return dims


    def remap_image(self, name, img, small, page_dims, params):

        height = 0.5 * page_dims[1] * self.OUTPUT_ZOOM * img.shape[0]
        height = self.round_nearest_multiple(height, self.REMAP_DECIMATE)

        width = self.round_nearest_multiple(height * page_dims[0] / page_dims[1],
                                    self.REMAP_DECIMATE)

        print('  output will be {}x{}'.format(width, height))

        height_small, width_small = np.floor_divide(
                [height, width],
                self.REMAP_DECIMATE,
            )

        page_x_range = np.linspace(0, page_dims[0], width_small)
        page_y_range = np.linspace(0, page_dims[1], height_small)

        page_x_coords, page_y_coords = np.meshgrid(page_x_range, page_y_range)

        page_xy_coords = np.hstack((page_x_coords.flatten().reshape((-1, 1)),
                                    page_y_coords.flatten().reshape((-1, 1))))

        page_xy_coords = page_xy_coords.astype(np.float32)

        image_points = self.project_xy(page_xy_coords, params)
        image_points = self.norm2pix(img.shape, image_points, False)

        image_x_coords = image_points[:, 0, 0].reshape(page_x_coords.shape)
        image_y_coords = image_points[:, 0, 1].reshape(page_y_coords.shape)

        image_x_coords = cv2.resize(image_x_coords, (width, height),
                                    interpolation=cv2.INTER_CUBIC).astype(np.float32)

        image_y_coords = cv2.resize(image_y_coords, (width, height),
                                    interpolation=cv2.INTER_CUBIC).astype(np.float32)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        remapped = cv2.remap(img_gray, image_x_coords, image_y_coords,
                            cv2.INTER_CUBIC,
                            None, cv2.BORDER_REPLICATE)

        if self.NO_BINARY:
            thresh = remapped
            pil_image = Image.fromarray(thresh)
        else:
            thresh = cv2.adaptiveThreshold(
                remapped,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                self.ADAPTIVE_WINSZ,
                25,
            )
            pil_image = Image.fromarray(thresh)
            pil_image = pil_image.convert("1")

        threshfile = name + '_thresh.png'
        pil_image.save(f"data/output_images/{threshfile}", dpi=(self.OUTPUT_DPI, self.OUTPUT_DPI))

        if self.DEBUG_LEVEL >= 1:
            height = small.shape[0]
            width = int(round(height * float(thresh.shape[1])/thresh.shape[0]))
            display = cv2.resize(thresh, (width, height),
                                interpolation=cv2.INTER_AREA)
            self.debug_show(name, 6, 'output', display)

        return threshfile


    def dewarp(self, image_path):
        """
        Phương thức chính để làm phẳng ảnh.

        Args:
            image_input (str or np.ndarray): Đường dẫn tới ảnh hoặc ảnh dạng numpy array.
            apply_threshold (bool): Áp dụng threshold để làm ảnh đen trắng.

        Returns:
            np.ndarray: Ảnh đã được làm phẳng hoặc ảnh gốc nếu xử lý thất bại.
        """
        if self.DEBUG_LEVEL > 0 and self.DEBUG_OUTPUT != 'file':
            cv2.namedWindow(self.WINDOW_NAME)

        outfiles = []

        img = cv2.imread(image_path)
        small = self.resize_to_screen(img)
        basename = os.path.basename(image_path)
        name, _ = os.path.splitext(basename)

        print('loaded', basename, 'with size', self.imgsize(img),)
        print('and resized to', self.imgsize(small))

        if self.DEBUG_LEVEL >= 3:
            self.debug_show(name, 0.0, 'original', small)

        pagemask, page_outline = self.get_page_extents(small)

        cinfo_list = self.get_contours(name, small, pagemask, 'text')
        spans = self.assemble_spans(name, small, pagemask, cinfo_list)

        if len(spans) < 3:
            print('  detecting lines because only', len(spans), 'text spans')
            cinfo_list = self.get_contours(name, small, pagemask, 'line')
            spans2 = self.assemble_spans(name, small, pagemask, cinfo_list)
            if len(spans2) > len(spans):
                spans = spans2

        if len(spans) < 1:
            print('skipping', name, 'because only', len(spans), 'spans')

        span_points = self.sample_spans(small.shape, spans)

        print('  got', len(spans), 'spans',)
        print('with', sum([len(pts) for pts in span_points]), 'points.')

        corners, ycoords, xcoords = self.keypoints_from_samples(name, small,
                                                        pagemask,
                                                        page_outline,
                                                        span_points)

        rough_dims, span_counts, params = self.get_default_params(corners,
                                                            ycoords, xcoords)

        dstpoints = np.vstack((corners[0].reshape((1, 1, 2)),) +
                            tuple(span_points))

        params = self.optimize_params(name, small,
                                dstpoints,
                                span_counts, params)

        page_dims = self.get_page_dims(corners, rough_dims, params)

        outfile = self.remap_image(name, img, small, page_dims, params)

        outfiles.append(outfile)

        print('  wrote', outfile)
        print()

        print('to convert to PDF (requires ImageMagick):')
        print('  convert -compress Group4 ' + ' '.join(outfiles) + ' output.pdf')