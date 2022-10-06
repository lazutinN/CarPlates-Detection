import os
import torch


def train(img_size,
          batch_size,
          epochs,
          data_config_file,
          weights):
    cmd = 'python ../yolov5/train.py --img {img_size} --batch {batch_size} ' \
          '--epochs {epochs} --data {data_file} --weights {weights}'
    cmd = cmd.format(img_size=img_size, batch_size=batch_size, epochs=epochs, data_file=data_config_file,
                     weights=weights)
    print("TRAINING STARTED")
    os.system(cmd)
    print("TRAINING FINISHED")


torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cpu')
train(640, 2, 1, '../dataset.yml', 'yolov5s.pt')
