from utils.constants import dataset_path
import io
import base64
from PIL import Image
import os
import cv2
import numpy as np

os.mkdir('leak_data')

with open(str(dataset_path / 'sample_submission.csv'), 'r') as f:
    lines_subm = f.readlines()

# load, rotate, flip, crop, save
for img_id, line_subm in enumerate(lines_subm[1:]):
    line_subm = line_subm.strip()
    data_s = line_subm.split(",")[2]
    # load
    im = Image.open(io.BytesIO(base64.b64decode(data_s)))
    name = "0" * (3 - len(str(img_id + 1))) + str(img_id + 1)
    im.save('tmp.png')
    img_s = cv2.imread('tmp.png', 0)
    # rotate
    img_s = np.rot90(img_s, k=3, axes=(0, 1))
    # flip
    img_s = img_s[:, ::-1]
    # crop
    img_s_512 = img_s[:512, :512]
    # save
    cv2.imwrite(f'leak_data/{name}.png', img_s_512)
