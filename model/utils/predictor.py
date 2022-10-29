import shutil
from os import listdir
import easyocr


# Функция первичного распознавания номера
def predict(model, img_name):
    results = model(img_name)  # находим номер на изображении
    results.render()

    save_path = './runs/detect/' + img_name.rsplit('/', 1)[-1].split('.')[0]
    added_path = 'crops/plates'
    # Чистим старые данные о предсказаниях по этой картинке, если они есть
    shutil.rmtree(save_path, ignore_errors=True)
    # вырезаем картинки с автомобильными номерами и сохраняем
    list(map(lambda x: x['im'],
             results.crop(save=True, save_dir=save_path, exist_ok=True)))
    files = []
    try:
        files = listdir('/'.join([save_path, added_path]))
    except Exception as e:
        return []

    # Подключаем модель easyOCR для распознавания символов на изображении
    reader = easyocr.Reader(['ru'])
    result = []
    # Каждый найденный номер переводим в текст
    for f in files:
        try:
            result.append(reader.readtext('/'.join([save_path, added_path, f]),
                                          allowlist='0123456789АаВвЕеКкМмНнОоРрСсТтУуХх',
                                          detail=0, paragraph=True, min_size=60)[0])
        except Exception as e:
            result.append('')
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