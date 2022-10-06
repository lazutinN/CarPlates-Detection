import cv2
import numpy as np
import urllib

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2
white_color = (255, 255, 255)
red_color = (0, 0, 255)


def set_text(image, text, position, color):
    return cv2.putText(image, text, position, font,
                       fontScale, color, thickness, cv2.LINE_AA)


def resize(image, height):
    resize_koef = height / image.shape[0]
    dim = (int(resize_koef * image.shape[1]), height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def darkening(image, koef):
    mat = np.ones(image.shape, dtype='uint8') * koef
    return cv2.subtract(image, mat)


def set_image(image1, image2, position):
    x_offset, y_offset = position
    image1[y_offset:y_offset + image2.shape[0], x_offset:x_offset + image2.shape[1]] = image2
    return image1


def read_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)