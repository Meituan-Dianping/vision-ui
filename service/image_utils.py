import cv2
import numpy


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


def get_image_text(img, engine='cnocr'):
    text = 'cnocr'
    return text


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
