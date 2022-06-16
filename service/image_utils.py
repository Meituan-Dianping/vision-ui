import base64
import os
import cv2
import numpy
import requests
from config import IMAGE_TEMP_DIR


def merge_rectangle_contours(rectangle_contours):
    merged_contours = [rectangle_contours[0]]
    for rec in rectangle_contours[1:]:
        for i in range(len(merged_contours)):
            x_min = rec[0][0]
            y_min = rec[0][1]
            x_max = rec[2][0]
            y_max = rec[2][1]
            merged_x_min = merged_contours[i][0][0]
            merged_y_min = merged_contours[i][0][1]
            merged_x_max = merged_contours[i][2][0]
            merged_y_max = merged_contours[i][2][1]
            if x_min >= merged_x_min and y_min >= merged_y_min and x_max <= merged_x_max and y_max <= merged_y_max:
                break
            else:
                if i == len(merged_contours)-1:
                    merged_contours.append(rec)
    return merged_contours


def contour_area_filter(binary, contours, thresh=1500):
    rectangle_contours =[]
    h, w = binary.shape
    for contour in contours:
        if thresh < cv2.contourArea(contour) < 0.2*h*w:
            rectangle_contours.append(contour)
    return rectangle_contours


def get_roi_image(img, rectangle_contour):
    roi_image = img[rectangle_contour[0][1]:rectangle_contour[2][1],
                    rectangle_contour[0][0]:rectangle_contour[1][0]]
    return roi_image


def get_pop_v(image):
    """
    calculate value if a pop window exit
    :param image: image path
    :return: mean of v channel
    """
    img = cv2.imread('capture/'+image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return numpy.mean(v)


def get_rectangle_contours(binary):
    _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangle_contours = []
    for counter in contours:
        x, y, w, h = cv2.boundingRect(counter)
        cnt = numpy.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        rectangle_contours.append(cnt)
    rectangle_contours = sorted(rectangle_contours, key=cv2.contourArea, reverse=True)[:100]
    rectangle_contours = contour_area_filter(binary, rectangle_contours)
    rectangle_contours = merge_rectangle_contours(rectangle_contours)
    return rectangle_contours


def get_center_pos(contour):
    x = int((contour[0][0]+contour[1][0])/2)
    y = int((contour[1][1]+contour[2][1])/2)
    return [x, y]


def get_label_pos(contour):
    center = get_center_pos(contour)
    x = int((int((center[0]+contour[2][0])/2)+contour[2][0])/2)
    y = int((int((center[1]+contour[2][1])/2)+contour[2][1])/2)
    return [x, y]


def draw_contours(img, contours, color="info"):
    if color == "info":
        cv2.drawContours(img, contours, -1, (255, 145, 30), 3)


def yolox_preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = numpy.ones((input_size[0], input_size[1], 3), dtype=numpy.uint8) * 114
    else:
        padded_img = numpy.ones(input_size, dtype=numpy.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(numpy.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = numpy.ascontiguousarray(padded_img, dtype=numpy.float32)
    return padded_img, r


def yolox_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = numpy.meshgrid(numpy.arange(wsize), numpy.arange(hsize))
        grid = numpy.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(numpy.full((*shape, 1), stride))
    grids = numpy.concatenate(grids, 1)
    expanded_strides = numpy.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = numpy.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = numpy.maximum(x1[i], x1[order[1:]])
        yy1 = numpy.maximum(y1[i], y1[order[1:]])
        xx2 = numpy.minimum(x2[i], x2[order[1:]])
        yy2 = numpy.minimum(y2[i], y2[order[1:]])

        w = numpy.maximum(0.0, xx2 - xx1 + 1)
        h = numpy.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = numpy.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[numpy.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = numpy.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = numpy.ones((len(keep), 1)) * cls_ind
                dets = numpy.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return numpy.concatenate(final_dets, 0)


def img_show(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    _COLORS = numpy.array([255, 0, 0,
                           195, 123, 40,
                           110, 176, 23]).astype(numpy.float32).reshape(-1, 3)
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = _COLORS[cls_id].astype(numpy.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if numpy.mean(_COLORS[cls_id]) > 128 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)

        txt_bk_color = (_COLORS[cls_id] * 0.7).astype(numpy.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def download_image(img_url, image_name):
    if not os.path.exists(IMAGE_TEMP_DIR):
        os.makedirs(IMAGE_TEMP_DIR)
    image_path = os.path.join(IMAGE_TEMP_DIR, image_name)
    r = requests.get(img_url, stream=True)
    if r.status_code == 200:
        with open(image_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=32):
                f.write(chunk)
    else:
        raise Exception(f'Download image fail, image url reponse code: {r.status_code}')
    return image_path


def save_base64_image(base64_image, image_name):
    if not os.path.exists(IMAGE_TEMP_DIR):
        os.makedirs(IMAGE_TEMP_DIR)
    image_path = os.path.join(IMAGE_TEMP_DIR, image_name)
    img_str = base64.b64decode(base64_image)
    img_np_arr = numpy.fromstring(img_str, numpy.uint8)
    image = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
    cv2.imwrite(image_path, image)
    return image_path


def get_hash_score(hash1, hash2, precision=8):
    """
    calculate similar score with line A and line B
    :param hash1: input line A with hash code
    :param hash2: input line B with hash code
    :return: similar score in 0-1.0
    """
    assert len(hash1) == len(hash2)
    score = 1 - sum([ch1 != ch2 for ch1, ch2 in zip(hash1, hash2)]) * 1.0 / (precision * precision)
    return score


def m_diff(e, f, i=0, j=0, equal_obj=object()):
    N, M, L, Z = len(e), len(f), len(e) + len(f), 2 * min(len(e), len(f)) + 2
    if N > 0 and M > 0:
        w, g, p = N - M, [0] * Z, [0] * Z
        for h in range(0, (L // 2 + (L % 2 != 0)) + 1):
            for r in range(0, 2):
                c, d, o, m = (g, p, 1, 1) if r == 0 else (p, g, 0, -1)
                for k in range(-(h - 2 * max(0, h - M)), h - 2 * max(0, h - N) + 1, 2):
                    a = c[(k + 1) % Z] if (k == -h or k != h and c[(k - 1) % Z] < c[(k + 1) % Z]) else c[(k - 1) % Z] + 1
                    b = a - k
                    s, t = a, b
                    while a < N and b < M and equal_obj.equal(e[(1 - o) * N + m * a + (o - 1)], f[(1 - o) * M + m * b + (o - 1)]):
                        a, b = a + 1, b + 1
                    c[k % Z], z = a, -(k - w)
                    if L % 2 == o and -(h - o) <= z <= h - o and c[k % Z] + d[z % Z] >= N:
                        D, x, y, u, v = (2 * h - 1, s, t, a, b) if o == 1 else (2 * h, N - a, M - b, N - s, M - t)
                        if D > 1 or (x != u and y != v):
                            return m_diff(e[0:x], f[0:y], i, j, equal_obj) + m_diff(e[u:N], f[v:M], i + u, j + v, equal_obj)
                        elif M > N:
                            return m_diff([], f[N:M], i + N, j + N, equal_obj)
                        elif M < N:
                            return m_diff(e[M:N], [], i + M, j + M, equal_obj)
                        else:
                            return []
    elif N > 0:
        return [{"operation": "delete", "position_old": i + n} for n in range(0, N)]
    else:
        return [{"operation": "insert", "position_old": i, "position_new": j + n} for n in range(0, M)]


def compute_iou(rect1, rect2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    rec1 = [rect1[1], rect1[0], rect1[3], rect1[2]]
    rec2 = [rect2[1], rect2[0], rect2[3], rect2[2]]
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def proposal_fine_tune(scores, proposals, thresh):
    _ids = [i for i, score in enumerate(scores) if score > thresh]
    top_k_ids = numpy.argsort(scores)[-2:]
    if len(_ids) > 1:
        index_i = top_k_ids[0]
        index_j = top_k_ids[1]
        region_i = proposals[index_i]['elem_det_region']
        region_j = proposals[index_j]['elem_det_region']
        iou = compute_iou(region_i, region_j)
        print(iou)
        if iou > 0.3:
            _x = int(((region_j[0] - region_i[0]) / 2))
            _y = int(((region_j[1] - region_i[1]) / 2))
            proposals[index_j]['elem_det_region'] = [region_j[0] - _x, region_j[1] - _y,
                                                     region_j[2] - _x, region_j[3] - _y]


def get_image_patches(img: numpy.ndarray, patch_w, patch_h, resolution_w: float, resolution_h: float):
    patches = []
    origin_h, origin_w, _ = img.shape
    assert patch_w < origin_w and patch_h < origin_h
    padding_w = numpy.zeros((origin_h, origin_w % patch_w, 3), numpy.uint8) + 255
    padding_h = numpy.zeros((origin_h % patch_h, origin_w + origin_w % patch_w, 3), numpy.uint8) + 255
    img_padding = numpy.vstack((numpy.hstack((img, padding_w)), padding_h))
    h, w, _ = img_padding.shape
    step_w = int(w / patch_w)
    step_h = int(h / patch_h)
    point_count_map = {}
    # 滑动窗口生成patch
    for i in numpy.arange(0, step_w, resolution_w):
        for j in numpy.arange(0, step_h, resolution_h):
            x0 = int(i*patch_w)
            y0 = int(j*patch_h)
            x1 = int((i+1)*patch_w) if (i+1)*patch_w < origin_w else origin_w
            y1 = int((j+1)*patch_h) if (j+1)*patch_h < origin_h else origin_h
            if x1 > x0 and y1 > y0:
                patches.append({'elem_det_region': [x0, y0, x1, y1]})
                for point in [f"{x0},{y0}", f"{x0},{y1}", f"{x1},{y1}", f"{x1},{y0}"]:
                    if point not in point_count_map:
                        point_count_map[point] = 1
                    else:
                        point_count_map[point] = point_count_map[point] + 1
    return patches
