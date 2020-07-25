import cv2
import numpy as np
import keras
import os

class ShowingCallback(keras.callbacks.Callback):
    def __init__(self, test_imgs):
        super().__init__()
        self.test_imgs=test_imgs
        self.classes = ['staphylococcus_epidermidis', 'klebsiella_pneumoniae', 'staphylococcus_aureus',
                   'moraxella_catarrhalis',
                   'c_kefir', 'ent_cloacae']


    def on_epoch_end(self, batch, logs={}):
        prs = np.asarray(self.model.predict(self.test_imgs, batch_size=1))
        classes = [self.classes[i] for i in prs.argmax(axis=1)]
        scores= prs.max(axis=1)
        for i, (cl, score) in enumerate(zip(classes, scores)):
            print(i+1, cl, score)



