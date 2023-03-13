import json
from os import listdir
from os.path import splitext

from PIL import Image

# Переводим разметку данных в формат YOLO
folders = ['train', 'val', 'test']
for folder in folders:
    mypath = './dataset' + folder  # путь до изображений, которые нужно переразметить

    files = [f for f in listdir(mypath) if not (f.endswith('.txt'))]  # список изображений

    with open('./dataset' + '/train.json') as ann_f:
        ann = json.load(ann_f)  # сейчас информация обо всех изображениях хранится в одном json-файле
        for f in files:
            im = Image.open(mypath + "/" + f)
            w, h = im.size  # получаем размер изображения для дальнейшей нормализации координат
            # получаем текущую разметку для изображения
            nums = list(filter(lambda x: x["file"] == "train/" + f, ann))[0]["nums"]
            new_filename = splitext(f)[0] + '.txt'
            # будем записывать данные в YOLO-формате в одноименный текстовый файл
            with open(mypath + "/" + new_filename, 'w') as lbl_f:
                for num in nums:
                    box = num["box"]
                    xs = sorted([box[0][0], box[1][0], box[2][0], box[3][0]])
                    ys = sorted([box[0][1], box[1][1], box[2][1], box[3][1]])
                    x_min, x_max = xs[0], xs[3]
                    y_min, y_max = ys[0], ys[3]
                    w_norm = (x_max - x_min) * 1. / w
                    h_norm = (y_max - y_min) * 1. / h
                    x_center_norm = w_norm / 2 + x_min * 1. / w
                    y_center_norm = h_norm / 2 + y_min * 1. / h
                    # данные в формате class x_center y_center w h (все значения нормализованные)
                    lbl_f.write(
                        ' '.join(list(map(lambda x: str(x), [0, x_center_norm, y_center_norm,
                                                             w_norm, h_norm]))))
