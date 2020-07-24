import numpy as np
from glob import glob
import cv2
import keras
import json

def load(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_mask(path):
    data = load(path)
    mask = np.zeros((512, 640), dtype=np.uint8)
    name = set()
    for shape in data['shapes']:
        filtered_points = []
        name.add(shape['label'])
        for point in shape['points']:
            filtered_points.append(point)
        filtered_points = np.array([filtered_points], dtype=np.int32)
        cv2.fillPoly(mask, filtered_points, [1])
    mask = np.expand_dims(mask, axis=2)
    return mask.astype('float')

def add_weights(mask):
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 2))
    new_mask[:, :, 0] = mask[:, :, 0]
    weights = np.zeros((mask.shape[0], mask.shape[1]))
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=2)
    dilation = cv2.dilate(mask, kernel, iterations=2)
    weights[(dilation > 0.5) & (erosion < 0.5)] = 1
    new_mask[:, :, 0] = mask[:, :, 0]
    new_mask[:, :, 1] = weights
    return new_mask


def read_img(path):
    img = cv2.imread(path)
    return img

def get_pathes(mode, dataset_path):
    path = dataset_path / mode
    imgs_paths = glob(str(path / '*.png'))
    masks_json_paths = glob(str(path / '*.json'))

    imgs_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    masks_json_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return imgs_paths, masks_json_paths


class Dataset:

    def to_list(self, size=None):
        val_x = []
        val_y = []
        if size is None:
            size = len(self)
        for i in range(size):
            x, y = self[i]
            val_x.append(x)
            val_y.append(y)
        return np.array(val_x), np.array(val_y)


    def __init__(
            self,
            mode,
            dataset_path,
            augmentation=None,
    ):
        self.img_paths, masks_json_paths = get_pathes(mode, dataset_path)
        self.imgs = [read_img(img_path) for img_path in self.img_paths]

        if mode == 'test':
            self.masks = [np.zeros_like(self.imgs[0])[:, :, :1] for _ in self.img_paths]
        else:
            self.masks = [get_mask(x) for x in masks_json_paths]

        self.augmentation = augmentation

    def __getitem__(self, i):
        image = self.imgs[i].copy()
        mask = self.masks[i].copy()

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = image / 127.5 - 1

        new_mask = add_weights(mask)

        return image, new_mask

    def __len__(self):
        return len(self.imgs)


class Dataloder(keras.utils.Sequence):

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[self.indexes[j]])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
