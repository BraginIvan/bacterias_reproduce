import keras
from keras.models import Model
from keras.layers import Dense
import efficientnet.keras as efn
from .data_loader import Dataloder, Dataset
from .augmentations import get_augmentations
from utils.constants import dataset_path
from keras import backend as K
import tensorflow as tf
from .showing_predicts import ShowingCallback
from utils.gpu_memory import gpu_memory_settings
gpu_memory_settings()


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def run():
    configs = [
        (efn.EfficientNetB1, 4, 'classification_b1.h5'),
        (efn.EfficientNetB0, 4, 'classification_b0.h5')
    ]

    for model_type, BATCH_SIZE, filename in configs:
        train_dataset = Dataset(mode='train',
                                     augmentation=get_augmentations(),
                                     dataset_path=dataset_path)
        test_dataset = Dataset(mode='test',  dataset_path=dataset_path)
        train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        base_model = model_type(weights='imagenet', input_shape=(512, 640, 3), include_top=False, pooling='avg')
        x = Dense(128, activation='relu')(base_model.output)
        x = Dense(6, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=30, monitor='loss', verbose=3),
            ShowingCallback(test_dataset.to_list()[0])
        ]

        # model.compile(optimizer=keras.optimizers.Adam(0.001), loss=[focal_loss()], metrics=['acc'])
        model.compile(optimizer=keras.optimizers.Adam(0.001), loss='categorical_crossentropy', metrics=['acc'])

        model.fit_generator(
            train_dataloader,
            steps_per_epoch=len(train_dataloader),
            epochs=350,
            callbacks=callbacks)
        model.save("cce_" + filename)

        callbacks = [
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, monitor='loss', verbose=3),
            ShowingCallback(test_dataset.to_list()[0])
        ]
        train_dataset = Dataset(mode='train', dataset_path=dataset_path)
        train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        model.compile(optimizer=keras.optimizers.Adam(0.00001), loss=[focal_loss(gamma=1.2)], metrics=['acc'])
        model.fit_generator(
            train_dataloader,
            steps_per_epoch=len(train_dataloader),
            epochs=15,
            callbacks=callbacks)
        model.save("focal_" + filename)

# Predicts of different models are slightly different.
# We have 5 submissions, lets choose different classification outputs instead of different segmentation

# B1 cee result
# 1 staphylococcus_epidermidis 0.9999982
# 2 staphylococcus_epidermidis 0.9999981
# 3 ent_cloacae 1.0
# 4 ent_cloacae 1.0
# 5 ent_cloacae 1.0
# 6 ent_cloacae 1.0
# 7 ent_cloacae 1.0
# 8 klebsiella_pneumoniae 0.9999995
# 9 klebsiella_pneumoniae 1.0
# 10 klebsiella_pneumoniae 0.9999995
# 11 klebsiella_pneumoniae 1.0
# 12 klebsiella_pneumoniae 1.0
# 13 klebsiella_pneumoniae 0.99999964
# 14 staphylococcus_aureus 0.99869436
# 15 staphylococcus_aureus 0.99959844
# 16 staphylococcus_epidermidis 0.78662664 !!!!!!!! (send both staphylococcus_epidermidis and staphylococcus_aureus)
# 17 staphylococcus_aureus 0.99971884
# 18 staphylococcus_aureus 0.9996276
# 19 staphylococcus_aureus 0.9999769
# 20 staphylococcus_aureus 0.99997234
# 21 staphylococcus_aureus 0.9997273
# 22 staphylococcus_epidermidis 0.99999857
# 23 staphylococcus_epidermidis 0.9999924
# 24 staphylococcus_epidermidis 0.99984634
# 25 staphylococcus_epidermidis 0.9999794
# 26 staphylococcus_epidermidis 0.9991202
# 27 staphylococcus_epidermidis 0.81047314 !! (send both staphylococcus_epidermidis and staphylococcus_aureus) it's public and staphylococcus_epidermidis correct
# 28 staphylococcus_epidermidis 0.99999976
# 29 staphylococcus_epidermidis 0.9999844
# 30 moraxella_catarrhalis 0.99999976
# 31 moraxella_catarrhalis 1.0
# 32 moraxella_catarrhalis 1.0
# 33 moraxella_catarrhalis 0.9999839
# 34 moraxella_catarrhalis 1.0
# 35 moraxella_catarrhalis 0.9999999
# 36 moraxella_catarrhalis 0.99999845
# 37 moraxella_catarrhalis 0.9999993
# 38 moraxella_catarrhalis 0.99999976
# 39 moraxella_catarrhalis 1.0
# 40 moraxella_catarrhalis 0.99999964
# 41 moraxella_catarrhalis 0.9999931
# 42 moraxella_catarrhalis 0.9999981
# 43 moraxella_catarrhalis 1.0
# 44 moraxella_catarrhalis 0.99999976
# 45 moraxella_catarrhalis 0.9721053
# 46 c_kefir 1.0
# 47 c_kefir 1.0
# 48 c_kefir 0.9999999
# 49 c_kefir 1.0
# 50 c_kefir 1.0
# 51 c_kefir 1.0
# 52 c_kefir 1.0
# 53 c_kefir 1.0
# 54 c_kefir 1.0
# 55 c_kefir 0.9999858
# 56 c_kefir 1.0
# 57 c_kefir 1.0
# 58 c_kefir 1.0
# 59 c_kefir 1.0
# 60 c_kefir 0.99999917
# 61 c_kefir 1.0
# 62 c_kefir 1.0
# 63 c_kefir 1.0
# 64 c_kefir 1.0
# 65 c_kefir 1.0
# 66 c_kefir 1.0
# 67 c_kefir 1.0
# 68 c_kefir 1.0
# 69 c_kefir 1.0
# 70 c_kefir 0.99992776
# 71 c_kefir 1.0
# 72 c_kefir 1.0
# 73 c_kefir 1.0
# 74 ent_cloacae 1.0
# 75 ent_cloacae 1.0
# 76 ent_cloacae 1.0
# 77 ent_cloacae 1.0
# 78 ent_cloacae 1.0
# 79 ent_cloacae 1.0
# 80 ent_cloacae 1.0
# 81 ent_cloacae 1.0
# 82 ent_cloacae 1.0
# 83 ent_cloacae 1.0
# 84 ent_cloacae 1.0
# 85 ent_cloacae 1.0
# 86 ent_cloacae 1.0
# 87 ent_cloacae 1.0
# 88 ent_cloacae 1.0
# 89 ent_cloacae 1.0
# 90 ent_cloacae 1.0
# 91 ent_cloacae 1.0
# 92 ent_cloacae 1.0
# 93 ent_cloacae 1.0
# 94 ent_cloacae 1.0
# 95 ent_cloacae 1.0
# 96 ent_cloacae 1.0
# 97 ent_cloacae 1.0
# 98 ent_cloacae 1.0
# 99 ent_cloacae 1.0
# 100 ent_cloacae 1.0
# 101 ent_cloacae 1.0
# 102 ent_cloacae 1.0
# 103 ent_cloacae 1.0
# 104 ent_cloacae 1.0
# 105 ent_cloacae 1.0
# 106 ent_cloacae 1.0
# 107 ent_cloacae 1.0


# B1 focal result
# 1 staphylococcus_epidermidis 0.9999982
# 2 staphylococcus_epidermidis 0.999998
# 3 ent_cloacae 1.0
# 4 ent_cloacae 1.0
# 5 ent_cloacae 1.0
# 6 ent_cloacae 1.0
# 7 ent_cloacae 1.0
# 8 klebsiella_pneumoniae 0.9999994
# 9 klebsiella_pneumoniae 1.0
# 10 klebsiella_pneumoniae 0.9999995
# 11 klebsiella_pneumoniae 1.0
# 12 klebsiella_pneumoniae 1.0
# 13 klebsiella_pneumoniae 0.99999964
# 14 staphylococcus_aureus 0.9989336
# 15 staphylococcus_aureus 0.9997602
# 16 staphylococcus_aureus 0.9723475 !!!!!!!!!! (send both staphylococcus_epidermidis and staphylococcus_aureus)
# 17 staphylococcus_aureus 0.9997944
# 18 staphylococcus_aureus 0.9997743
# 19 staphylococcus_aureus 0.9999827
# 20 staphylococcus_aureus 0.9999788
# 21 staphylococcus_aureus 0.9997888
# 22 staphylococcus_epidermidis 0.99999845
# 23 staphylococcus_epidermidis 0.99999166
# 24 staphylococcus_epidermidis 0.99986005
# 25 staphylococcus_epidermidis 0.9999819
# 26 staphylococcus_epidermidis 0.99913174
# 27 staphylococcus_epidermidis 0.800046 !! (send both staphylococcus_epidermidis and staphylococcus_aureus) it's public and staphylococcus_epidermidis correct
# 28 staphylococcus_epidermidis 0.99999964
# 29 staphylococcus_epidermidis 0.99998534
# 30 moraxella_catarrhalis 0.9999999
# 31 moraxella_catarrhalis 1.0
# 32 moraxella_catarrhalis 1.0
# 33 moraxella_catarrhalis 0.99998975
# 34 moraxella_catarrhalis 1.0
# 35 moraxella_catarrhalis 0.9999999
# 36 moraxella_catarrhalis 0.9999988
# 37 moraxella_catarrhalis 0.9999995
# 38 moraxella_catarrhalis 0.9999999
# 39 moraxella_catarrhalis 1.0
# 40 moraxella_catarrhalis 0.99999976
# 41 moraxella_catarrhalis 0.9999958
# 42 moraxella_catarrhalis 0.9999987
# 43 moraxella_catarrhalis 1.0
# 44 moraxella_catarrhalis 0.9999999
# 45 moraxella_catarrhalis 0.9730698
# 46 c_kefir 1.0
# 47 c_kefir 1.0
# 48 c_kefir 0.9999999
# 49 c_kefir 1.0
# 50 c_kefir 1.0
# 51 c_kefir 1.0
# 52 c_kefir 1.0
# 53 c_kefir 1.0
# 54 c_kefir 1.0
# 55 c_kefir 0.9998178
# 56 c_kefir 1.0
# 57 c_kefir 1.0
# 58 c_kefir 1.0
# 59 c_kefir 1.0
# 60 c_kefir 0.99999845
# 61 c_kefir 1.0
# 62 c_kefir 1.0
# 63 c_kefir 1.0
# 64 c_kefir 0.9999999
# 65 c_kefir 1.0
# 66 c_kefir 1.0
# 67 c_kefir 1.0
# 68 c_kefir 1.0
# 69 c_kefir 1.0
# 70 c_kefir 0.9997265
# 71 c_kefir 1.0
# 72 c_kefir 1.0
# 73 c_kefir 0.9999999
# 74 ent_cloacae 1.0
# 75 ent_cloacae 1.0
# 76 ent_cloacae 1.0
# 77 ent_cloacae 1.0
# 78 ent_cloacae 1.0
# 79 ent_cloacae 1.0
# 80 ent_cloacae 1.0
# 81 ent_cloacae 1.0
# 82 ent_cloacae 1.0
# 83 ent_cloacae 1.0
# 84 ent_cloacae 1.0
# 85 ent_cloacae 1.0
# 86 ent_cloacae 1.0
# 87 ent_cloacae 1.0
# 88 ent_cloacae 1.0
# 89 ent_cloacae 1.0
# 90 ent_cloacae 1.0
# 91 ent_cloacae 1.0
# 92 ent_cloacae 1.0
# 93 ent_cloacae 1.0
# 94 ent_cloacae 1.0
# 95 ent_cloacae 1.0
# 96 ent_cloacae 1.0
# 97 ent_cloacae 1.0
# 98 ent_cloacae 1.0
# 99 ent_cloacae 1.0
# 100 ent_cloacae 1.0
# 101 ent_cloacae 1.0
# 102 ent_cloacae 1.0
# 103 ent_cloacae 1.0
# 104 ent_cloacae 1.0
# 105 ent_cloacae 1.0
# 106 ent_cloacae 1.0
# 107 ent_cloacae 1.0


# b0 cce results
# 1 staphylococcus_epidermidis 0.9999999
# 2 staphylococcus_epidermidis 1.0
# 3 ent_cloacae 1.0
# 4 ent_cloacae 1.0
# 5 ent_cloacae 1.0
# 6 ent_cloacae 1.0
# 7 ent_cloacae 1.0
# 8 klebsiella_pneumoniae 1.0
# 9 klebsiella_pneumoniae 1.0
# 10 klebsiella_pneumoniae 1.0
# 11 klebsiella_pneumoniae 1.0
# 12 klebsiella_pneumoniae 1.0
# 13 klebsiella_pneumoniae 1.0
# 14 staphylococcus_aureus 0.99998903
# 15 staphylococcus_aureus 0.99987817
# 16 staphylococcus_aureus 0.999997
# 17 staphylococcus_aureus 0.9999684
# 18 staphylococcus_aureus 0.9910294
# 19 staphylococcus_aureus 0.99981886
# 20 staphylococcus_aureus 0.9997391
# 21 staphylococcus_aureus 0.9997594
# 22 staphylococcus_epidermidis 0.99996877
# 23 staphylococcus_epidermidis 0.99998796
# 24 staphylococcus_epidermidis 1.0
# 25 staphylococcus_epidermidis 0.99998856
# 26 staphylococcus_epidermidis 0.9998491
# 27 staphylococcus_epidermidis 0.9986374
# 28 staphylococcus_epidermidis 0.9999211
# 29 staphylococcus_epidermidis 0.99999964
# 30 moraxella_catarrhalis 0.99999785
# 31 moraxella_catarrhalis 1.0
# 32 moraxella_catarrhalis 1.0
# 33 moraxella_catarrhalis 1.0
# 34 moraxella_catarrhalis 1.0
# 35 moraxella_catarrhalis 0.9999999
# 36 moraxella_catarrhalis 0.99999666
# 37 moraxella_catarrhalis 1.0
# 38 moraxella_catarrhalis 1.0
# 39 moraxella_catarrhalis 0.99999845
# 40 moraxella_catarrhalis 1.0
# 41 moraxella_catarrhalis 1.0
# 42 moraxella_catarrhalis 0.9999993
# 43 moraxella_catarrhalis 1.0
# 44 moraxella_catarrhalis 1.0
# 45 moraxella_catarrhalis 0.9996468
# 46 c_kefir 1.0
# 47 c_kefir 1.0
# 48 c_kefir 1.0
# 49 c_kefir 1.0
# 50 c_kefir 1.0
# 51 c_kefir 1.0
# 52 c_kefir 1.0
# 53 c_kefir 1.0
# 54 c_kefir 1.0
# 55 c_kefir 1.0
# 56 c_kefir 1.0
# 57 c_kefir 1.0
# 58 c_kefir 1.0
# 59 c_kefir 1.0
# 60 c_kefir 1.0
# 61 c_kefir 1.0
# 62 c_kefir 1.0
# 63 c_kefir 1.0
# 64 c_kefir 1.0
# 65 c_kefir 1.0
# 66 c_kefir 1.0
# 67 c_kefir 1.0
# 68 c_kefir 1.0
# 69 c_kefir 1.0
# 70 c_kefir 1.0
# 71 c_kefir 1.0
# 72 c_kefir 1.0
# 73 c_kefir 1.0
# 74 ent_cloacae 1.0
# 75 ent_cloacae 1.0
# 76 ent_cloacae 1.0
# 77 ent_cloacae 1.0
# 78 ent_cloacae 1.0
# 79 ent_cloacae 1.0
# 80 ent_cloacae 1.0
# 81 ent_cloacae 1.0
# 82 ent_cloacae 1.0
# 83 ent_cloacae 1.0
# 84 ent_cloacae 1.0
# 85 ent_cloacae 1.0
# 86 ent_cloacae 1.0
# 87 ent_cloacae 1.0
# 88 ent_cloacae 1.0
# 89 ent_cloacae 1.0
# 90 ent_cloacae 1.0
# 91 ent_cloacae 1.0
# 92 ent_cloacae 1.0
# 93 ent_cloacae 1.0
# 94 ent_cloacae 1.0
# 95 ent_cloacae 1.0
# 96 ent_cloacae 1.0
# 97 ent_cloacae 1.0
# 98 ent_cloacae 1.0
# 99 ent_cloacae 1.0
# 100 ent_cloacae 1.0
# 101 ent_cloacae 1.0
# 102 ent_cloacae 1.0
# 103 ent_cloacae 1.0
# 104 ent_cloacae 1.0
# 105 ent_cloacae 1.0
# 106 ent_cloacae 1.0
# 107 ent_cloacae 1.0

#b0 focal
# 1 staphylococcus_epidermidis 1.0
# 2 staphylococcus_epidermidis 1.0
# 3 ent_cloacae 1.0
# 4 ent_cloacae 1.0
# 5 ent_cloacae 1.0
# 6 ent_cloacae 1.0
# 7 ent_cloacae 1.0
# 8 klebsiella_pneumoniae 1.0
# 9 klebsiella_pneumoniae 1.0
# 10 klebsiella_pneumoniae 1.0
# 11 klebsiella_pneumoniae 1.0
# 12 klebsiella_pneumoniae 1.0
# 13 klebsiella_pneumoniae 1.0
# 14 staphylococcus_aureus 0.99998844
# 15 staphylococcus_aureus 0.9998776
# 16 staphylococcus_aureus 0.9999962
# 17 staphylococcus_aureus 0.9999658
# 18 staphylococcus_aureus 0.98883957
# 19 staphylococcus_aureus 0.9998375
# 20 staphylococcus_aureus 0.9997124
# 21 staphylococcus_aureus 0.99981743
# 22 staphylococcus_epidermidis 0.99996173
# 23 staphylococcus_epidermidis 0.99997437
# 24 staphylococcus_epidermidis 1.0
# 25 staphylococcus_epidermidis 0.9999871
# 26 staphylococcus_epidermidis 0.99981445
# 27 staphylococcus_epidermidis 0.99900013
# 28 staphylococcus_epidermidis 0.9998913
# 29 staphylococcus_epidermidis 0.9999994
# 30 moraxella_catarrhalis 0.99999654
# 31 moraxella_catarrhalis 1.0
# 32 moraxella_catarrhalis 1.0
# 33 moraxella_catarrhalis 0.9999999
# 34 moraxella_catarrhalis 1.0
# 35 moraxella_catarrhalis 0.9999999
# 36 moraxella_catarrhalis 0.9999958
# 37 moraxella_catarrhalis 1.0
# 38 moraxella_catarrhalis 1.0
# 39 moraxella_catarrhalis 0.999997
# 40 moraxella_catarrhalis 1.0
# 41 moraxella_catarrhalis 1.0
# 42 moraxella_catarrhalis 0.9999988
# 43 moraxella_catarrhalis 1.0
# 44 moraxella_catarrhalis 1.0
# 45 moraxella_catarrhalis 0.9995864
# 46 c_kefir 1.0
# 47 c_kefir 1.0
# 48 c_kefir 1.0
# 49 c_kefir 1.0
# 50 c_kefir 1.0
# 51 c_kefir 1.0
# 52 c_kefir 1.0
# 53 c_kefir 1.0
# 54 c_kefir 1.0
# 55 c_kefir 1.0
# 56 c_kefir 1.0
# 57 c_kefir 1.0
# 58 c_kefir 1.0
# 59 c_kefir 1.0
# 60 c_kefir 1.0
# 61 c_kefir 1.0
# 62 c_kefir 1.0
# 63 c_kefir 1.0
# 64 c_kefir 1.0
# 65 c_kefir 1.0
# 66 c_kefir 1.0
# 67 c_kefir 1.0
# 68 c_kefir 1.0
# 69 c_kefir 1.0
# 70 c_kefir 1.0
# 71 c_kefir 1.0
# 72 c_kefir 1.0
# 73 c_kefir 1.0
# 74 ent_cloacae 1.0
# 75 ent_cloacae 1.0
# 76 ent_cloacae 1.0
# 77 ent_cloacae 1.0
# 78 ent_cloacae 1.0
# 79 ent_cloacae 1.0
# 80 ent_cloacae 1.0
# 81 ent_cloacae 1.0
# 82 ent_cloacae 1.0
# 83 ent_cloacae 1.0
# 84 ent_cloacae 1.0
# 85 ent_cloacae 1.0
# 86 ent_cloacae 1.0
# 87 ent_cloacae 1.0
# 88 ent_cloacae 1.0
# 89 ent_cloacae 1.0
# 90 ent_cloacae 1.0
# 91 ent_cloacae 1.0
# 92 ent_cloacae 1.0
# 93 ent_cloacae 1.0
# 94 ent_cloacae 1.0
# 95 ent_cloacae 1.0
# 96 ent_cloacae 1.0
# 97 ent_cloacae 1.0
# 98 ent_cloacae 1.0
# 99 ent_cloacae 1.0
# 100 ent_cloacae 1.0
# 101 ent_cloacae 1.0
# 102 ent_cloacae 1.0
# 103 ent_cloacae 1.0
# 104 ent_cloacae 1.0
# 105 ent_cloacae 1.0
# 106 ent_cloacae 1.0
# 107 ent_cloacae 1.0
