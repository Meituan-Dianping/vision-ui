import numpy as np
import cv2
from PIL import Image


def crop_rect(img, rect, alpha=0.05):
    img = np.asarray(img)
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    if angle > -45:
        center, size = tuple(map(int, center)), tuple(map(int, size))
        size = (int(size[0] * (1 + alpha)), int(size[1] + size[0] * alpha))
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1]), int(rect[1][0])])
        size = (int(size[0] * (1 + alpha)), int(size[1] + size[0] * alpha))
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    img_crop = Image.fromarray(img_crop)
    return img_crop


def draw_bbox(image, result, color=(209, 206, 0)):
    img = cv2.imread(image) if isinstance(image, str) else image.copy()
    mask = img.copy()
    for point in result:
        point = point.astype(int)
        cv2.fillConvexPoly(mask, point, color)
    alpha = 0.6
    beta = 1 - alpha
    gamma = 0
    img_mask = cv2.addWeighted(img, alpha, mask, beta, gamma)
    return img_mask
