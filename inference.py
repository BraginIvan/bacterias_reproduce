from utils.gpu_memory import gpu_memory_settings
from segmentation.data_loader import Dataset
import numpy as np
import cv2
from keras.models import load_model
import os
import efficientnet.keras
from utils.constants import dataset_path

gpu_memory_settings()
versions = 3
dataset = Dataset(mode='test', dataset_path=dataset_path)
data = dataset.to_list()
imgs = data[0]
masks = data[1]


def tta_predict(m, imgs):
    bs = 6
    pred = m.predict(imgs, batch_size=bs)
    pred_h_flip = m.predict(imgs[:, :, ::-1, :], batch_size=bs)[:, :, ::-1, :]
    pred_v_flip = m.predict(imgs[:, ::-1, :, :], batch_size=bs)[:, ::-1, :, :]
    pred_hv_flips = m.predict(imgs[:, ::-1, ::-1, :], batch_size=bs)[:, ::-1, ::-1, :]
    ps = [pred, pred_h_flip, pred_v_flip, pred_hv_flips]
    return np.median(ps, axis=0)


for model_type in ['dice_ft_cee', 'dice_ft_focal1', 'dice_ft_focal2', 'dice_ft_focal3', 'dice_ft_focal4']:
    os.mkdir(model_type)
    model_names = [f'{str(version)}/{model_type}_v{str(version)}.h5' for version in range(1, versions + 1)]
    preds = []
    for model_name in model_names:
        model = load_model(model_name, compile=False)
        pred = tta_predict(model, imgs)
        print(model_name)
        preds.append(pred)
    preds = np.mean(preds, axis=0)
    for i in range(len(imgs)):
        pred = preds[i][:, :, 0]
        name = "0" * (3 - len(str(i + 1))) + str(i + 1)
        cv2.imwrite(f"{model_type}/{name}.png", ((pred * 255).astype('uint8')))
