from service.image_utils import *
from dbnet_crnn.image_text import image_text


def get_image_text(image):
    img = cv2.imread('capture/' + image)
    h, w, _ = img.shape
    log_size = int(0.9*h)
    result = {
        'roi_text': image_text.get_text(img, log_size),
        'img_shape': img.shape
    }
    return result


if __name__ == '__main__':
    print(get_image_text('image_1.png'))