import pandas as pd

import torch

from os import listdir
import json
import shutil
import easyocr
import fastwer


# Функция первичного распознавания номера
def predict(model, img_name):
    results = model(img_name)  # находим номер на изображении
    results.render()

    save_path = './runs/detect/' + img_name.rsplit('/', 1)[-1].split('.')[0]
    added_path = 'crops/plates'
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


def correct(text):
    text = text.lower()
    dictionary = {
        'a': 'а',
        'b': 'в',
        'e': 'е',
        'k': 'к',
        'm': 'м',
        'h': 'н',
        'o': 'о',
        'p': 'р',
        'c': 'с',
        't': 'т',
        'y': 'у',
        'x': 'х'
    }
    for key in dictionary.keys():
        text = text.replace(key, dictionary[key])
    return text


def test(model):
    df_output = pd.DataFrame(columns=['img_filename', 'ocr_output', 'ref_text', 'cer'])
    mypath = '../../dataset/test'
    files = [f for f in listdir(mypath) if not (f.endswith('.txt'))]  # список изображений

    with open('../../dataset/train.json') as ann_f:
        ann = json.load(ann_f)
        for img in files:
            ref_txt = correct(list(filter(lambda x: x["file"] == "train/" + img, ann))[0]["nums"][0]["text"])
            preds = predict(model, mypath + '/' + img)
            corrected = '' if len(preds) == 0 else correct_plate_num(preds[0])
            cer = fastwer.score_sent(corrected, ref_txt, char_level=True)
            dictionary = {'img_filename': img, 'ocr_output': corrected, 'ref_text': ref_txt, 'cer': cer}
            df_output = df_output.append(dictionary, ignore_index=True)
    return df_output


mdl = torch.hub.load('ultralytics/yolov5', 'custom', path='../../last.pt', force_reload=True, device='cpu')
output = test(mdl)
# Overall performances
print(output.to_string())
mean_cer = output['cer'].mean()
print(f'Mean CER = {mean_cer}%')
