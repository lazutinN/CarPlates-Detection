import shutil
from os import listdir, path

import easyocr
import numpy as np
import torch
from matplotlib import pyplot as plt

import sys

# Имитация подключения к БД с разрешенными номерами и данными о машине и водителе
database = [
    {'number': 'с555вм',
     'region': 777,
     'name': 'Круз, Том Томасович',
     'company': 'ООО \'Миссия невыполнима\'',
     'photo': 'https://pentad.ru/wp-content/uploads/2019/08/Tom-Kruz-v-filme-Dzhek-Richer.jpg'},
    {'number': 'н001ах',
     'region': 777,
     'name': 'Питт, Уильямович Бред',
     'company': 'ООО \'Бойцовский клуб\'',
     'photo': 'https://www.film.ru/sites/default/files/filefield_paths/leonardo-dicaprio-and-brad-pitt-are-seen-the'
              '-set-of-once-news-photo-1594220917.jpg'}
]


# Функция первичного распознавания номера
def predict(model, img_name):
    results = model(img_name)  # находим номер на изображении
    results.print()
    plt.imshow(np.squeeze(results.render()))
    plt.show()

    save_path = './runs/detect/' + img_name.rsplit('/', 1)[-1].split('.')[0]
    added_path = 'crops/plates'
    shutil.rmtree(save_path, ignore_errors=True)
    # вырезаем картинки с автомобильными номерами и сохраняем
    list(map(lambda x: x['im'],
             results.crop(save=True, save_dir=save_path, exist_ok=True)))
    files = listdir('/'.join([save_path, added_path]))

    # Подключаем модель easyOCR для распознавания символов на изображении
    reader = easyocr.Reader(['ru'])
    result = []
    # Каждый найденный номер переводим в текст
    for f in files:
        result.append(reader.readtext('/'.join([save_path, added_path, f]),
                                      allowlist='0123456789АаВвЕеКкМмНнОоРрСсТтУуХх',
                                      detail=0, paragraph=True, min_size=60)[0])
    return result


# Функция, корректирующая распознанный номер
def correct_plate_num(plate):
    plate = plate.lower()  # все к нижнему регистру
    # Из-за путаницы 0 и О явно указываем, где должны быть цифры, а где - буквы
    part1 = plate[:1].replace('0', 'о')
    part2 = plate[1:4].replace('о', '0')
    part3 = plate[4:6].replace('0', 'о')
    part4 = plate[6:].replace('o', '0')
    return part1 + part2 + part3 + part4


# Функция поиска номера в БД
def get_data(plate_num, plate_code):
    result = list(filter(lambda x: x["number"] == plate_num and x["region"] == plate_code, database))
    if len(result) == 0:
        raise Exception('Проезд запрещен:' + plate_num + str(plate_code))
    return result[0]


# Основной сценарий работы с изображением
def pipeline(model, img_name):
    preds = predict(model, img_name)
    cars_data = []
    for pred in preds:
        corrected = correct_plate_num(pred)
        try:
            cars_data.append(get_data(corrected[:6], int(corrected[6:])))
        except Exception as e:
            print(e)
    return cars_data


weights_path = './yolov5/runs/train/exp/weights/last.pt'
if not path.exists(weights_path):
    weights_path = './last.pt'

# Берем обученную нами модель
mdl = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
print(pipeline(mdl, sys.argv[1]))
