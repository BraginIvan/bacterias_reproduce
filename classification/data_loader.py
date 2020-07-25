import numpy as np
from glob import glob
import cv2
import keras
import efficientnet.keras as efn

classes = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus', 'moraxella_catarrhalis',
           'c_kefir', 'ent_cloacae']
classes = {cl: i for i, cl in enumerate(classes)}


def load_classes(path):
    with open(str(path), 'r') as f:
        return [classes[l.strip().split(',')[1]] for l in f.readlines() if len(l) > 1]


class Dataset:
    def to_list(self):
        val_x = []
        val_y = []
        for i in range(len(self)):
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
        if mode == 'train':
            path = dataset_path / 'train'
            imgs_paths = glob(str(path / '*.png'))
            self.classes = load_classes(path / 'classes.csv')
        elif mode == 'test':
            path = dataset_path / 'test'
            imgs_paths = sorted(glob(str(path / '*.png')))
            self.classes = [1]*len(imgs_paths)
        else:
            raise AttributeError(f'mode = {mode} but expected train/val')

        imgs_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        self.imgs = [cv2.imread(path) for path in imgs_paths]
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = self.imgs[i].copy()
        cls = self.classes[i]
        if self.augmentation:
            image = self.augmentation(image=image)['image']
        image=image/127.5-1
        return image, np.eye(6, dtype=int)[cls]

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
        return [np.stack(samples, axis=0) for samples in zip(*data)]

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
