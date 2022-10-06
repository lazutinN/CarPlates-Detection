import pandas as pd
import torch
from os import listdir
import json
import fastwer
from model.utils import translator, predictor


def test(model):
    df_output = pd.DataFrame(columns=['img_filename', 'ocr_output', 'ref_text', 'cer'])
    test_path = '../../dataset/test'
    files = [f for f in listdir(test_path) if not (f.endswith('.txt'))]  # список тестовых изображений

    with open('../../dataset/train.json') as ann_f:
        ann = json.load(ann_f)  # файл с разметкой данных
        for img in files:
            # Переводим правильный ответ с номером в русские символы
            ref_txt = translator.translate(
                list(filter(lambda x: x["file"] == "train/" + img, ann))[0]["nums"][0]["text"], translator.eng_ru)
            # Получаем предсказания
            preds = predictor.predict(model, test_path + '/' + img)
            # Корректируем предсказания
            corrected = '' if len(preds) == 0 else predictor.correct_plate_num(preds[0])
            # Считаем CER
            cer = fastwer.score_sent(corrected, ref_txt, char_level=True)
            # Заносим данные в датафрейм
            dictionary = {'img_filename': img, 'ocr_output': corrected, 'ref_text': ref_txt, 'cer': cer}
            df_output = df_output.append(dictionary, ignore_index=True)
    return df_output


mdl = torch.hub.load('ultralytics/yolov5', 'custom', path='../../last.pt', force_reload=True, device='cpu')
output = test(mdl)
# Среднее CER
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(output.sort_values(by='cer', ascending=False))
mean_cer = output['cer'].mean()
print(f'Mean CER = {mean_cer}%')
