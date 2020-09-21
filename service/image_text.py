from service.image_utils import *
from dbnet_crnn.image_text import image_text


def operation_morphology(img, operation_type, k):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if operation_type == 'open':
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    else:
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return result


def get_text_contour(img, contour):
    k = 30
    img_h = img.shape[0]
    img_w = img.shape[1]
    contour_x1 = contour[0][0]
    contour_y1 = contour[0][1]
    contour_x2 = contour[2][0]
    contour_y2 = contour[2][1]
    x1 = contour_x1 - k if contour_x1 - k > 0 else contour_x1
    y1 = contour_y1 - k if contour_y1 - k > 0 else contour_y1
    x2 = contour_x2 + k if contour_x2 + k < img_w else img_w
    y2 = contour_y2 + k if contour_y2 + k < img_h else img_h
    text_contour = numpy.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return text_contour


def get_text_roi(img):
    img = cv2.imread('capture/'+img)
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    binary_open = operation_morphology(binary, 'open', 30)
    binary_open = cv2.GaussianBlur(binary_open, (5, 5), 0)
    contours = get_rectangle_contours(binary_open)
    img_roi = gray
    roi_text_list = []
    for contour in contours:
        contour = get_text_contour(img_roi, contour)
        roi = get_roi_image(img_roi, contour)
        roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        text = get_image_text(roi).replace(' ', '')
        if len(text) > 0:
            roi_text_list.append({'rect': contour.tolist(), 'pos': get_center_pos(contour), 'text': text})

    result = {
        'roi_text': roi_text_list,
        'img_shape': img.shape
    }
    return result


def get_image_text(image):
    img = cv2.imread('capture/' + image)
    h, w, _ = img.shape
    log_size = int(0.9*h)
    result = {
        'roi_text': image_text.get_text(img, log_size),
        'img_shape': img.shape
    }
    return result

