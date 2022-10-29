import sys
from os import path
import cv2
import torch
from model.utils import translator, printer, predictor

# Имитация подключения к БД с разрешенными номерами и данными о машине и водителе
database = [
    {'number': 'с555вм',
     'region': 777,
     'name': 'Cruise, Tom',
     'company': 'OOO \'Mission impossible\'',
     'photo': 'https://pentad.ru/wp-content/uploads/2019/08/Tom-Kruz-v-filme-Dzhek-Richer.jpg'},
    {'number': 'н001ах',
     'region': 777,
     'name': 'Pitt, Brad',
     'company': 'OOO \'Fighting club\'',
     'photo': 'https://www.film.ru/sites/default/files/filefield_paths/leonardo-dicaprio-and-brad-pitt-are-seen-the'
              '-set-of-once-news-photo-1594220917.jpg'}
]


# Функция поиска номера в БД
def get_data(plate_num, plate_code):
    result = list(filter(lambda x: x["number"] == plate_num and x["region"] == plate_code, database))
    if len(result) == 0:
        return None
    return result[0]


# Отрисовка результата
def print_result(img_url, data, num):
    # Изначальное изображение в качестве фона
    image = printer.read_image(img_url)
    # Затемнить фон
    image = printer.darkening(image, 80)
    # Изменить размер пропорционально с высотой 600
    image = printer.resize(image, 600)

    # Помещаем текст с распознанным номером
    image = printer.set_text(image, translator.translate(num, translator.ru_eng).upper(), (50, 50), printer.white_color)
    # Если данных о водителе нет в базе, выводим сообщение, что доступ ограничен. Иначе выводим данные и фото водителя
    if len(data) == 0:
        image = printer.set_text(image, 'Access denied', (50, 100), printer.red_color)
    else:
        image = printer.set_text(image, data[0]['name'], (50, 100), printer.white_color)
        image = printer.set_text(image, data[0]['company'], (50, 150), printer.white_color)

        # Получаем изображение водителя
        image2 = printer.read_image(data[0]['photo'])
        # Приводим фото к нужному нам размеру
        image2 = printer.resize(image2, 200)
        # Помещаем фото водителя на выводимое изображение
        image = printer.set_image(image, image2, (50, 200))

    # Выводим изображение
    cv2.imshow('Result', image)

    if cv2.waitKey() & 0xff == 27:
        quit()


# Основной сценарий работы с изображением
def pipeline(model, img_name):
    # Получаем первичное предсказание номеров
    preds = predictor.predict(model, img_name)
    cars_data = []
    # Каждый найденный номер корректируем и ищем в базе
    for pred in preds:
        corrected = predictor.correct_plate_num(pred)
        try:
            data = get_data(corrected[:6], int(corrected[6:]))
            if data is not None:
                cars_data.append(data)
        except Exception as e:
            print(corrected, e)
    print_result(img_name, cars_data, corrected)
    return cars_data, corrected


# Путь до весов, которые будем использовать
weights_path = '../yolov5/runs/train/exp/weights/last.pt'
if not path.exists(weights_path):
    weights_path = '../last.pt'

# Берем обученную нами модель
mdl = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
img_path = sys.argv[1]
pipeline(mdl, img_path)
