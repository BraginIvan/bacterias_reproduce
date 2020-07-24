from utils.gpu_memory import gpu_memory_settings
gpu_memory_settings()
from .losses import DiceLoss, IOUScore, WeightedBinaryCELoss, BinaryFocalLoss
from .augmentation import get_augmentations
import keras
import segmentation_models as sm
from .data_loader import Dataset, Dataloder
import os
BATCH_SIZE = 4
IMG_WIGHT = 640
IMG_HEIGHT = 512

def run(config, version):
    if not os.path.exists(str(version)):
        os.mkdir(str(version))
    train_dataset = Dataset(mode='train',
                            dataset_path=config['dataset_path'],
                            augmentation=get_augmentations())

    train_data = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = sm.Unet(backbone_name='efficientnetb1',
                    input_shape=(IMG_HEIGHT, IMG_WIGHT, 3),
                    classes=1,
                    activation='sigmoid',
                    encoder_features=['block6a_expand_activation', 'block4a_expand_activation',
                                      'block3a_expand_activation',
                                      'block2a_expand_activation', 'input_1'],
                    decoder_filters=(256, 128, 64, 32, 32),
                    encoder_freeze=True
                    )

    dice_loss = DiceLoss(beta=1.0, per_image=True)
    focal_loss = BinaryFocalLoss(gamma=5)
    cee_loss = WeightedBinaryCELoss()

    if not os.path.exists(f'{str(version)}/dice_tmp1_v{str(version)}.h5'):
        # warmup decoder
        optim = keras.optimizers.Adam()
        model.compile(optim, dice_loss)
        model.fit_generator(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=10)
        for layer in model.layers:
            layer.trainable = True

        # training model 100 epoches
        optim = keras.optimizers.Adam()
        model.compile(optim, dice_loss, [IOUScore()])
        model.fit_generator(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=100,
        )
        model.save(f'{str(version)}/dice_tmp1_v{str(version)}.h5')


    if not os.path.exists(f'{str(version)}/dice_tmp2_v{str(version)}.h5'):
        model.load_weights(f'{str(version)}/dice_tmp1_v{str(version)}.h5')

        # training model with lr decay
        optim = keras.optimizers.Adam()
        model.compile(optim, dice_loss, [IOUScore()])
        model.fit_generator(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=200,
            callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=6, monitor='iou_score0.5', verbose=3,
                                                         mode='max')],
        )
        model.save(f'{str(version)}/dice_tmp2_v{str(version)}.h5')


    if not os.path.exists(f'{str(version)}/dice_ft_cee_v{str(version)}.h5'):
        model.load_weights(f'{str(version)}/dice_tmp2_v{str(version)}.h5')
        optim = keras.optimizers.Adam()
        model.compile(optim, dice_loss+cee_loss, [IOUScore()])
        model.fit_generator(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=100,
            callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=3, monitor='iou_score0.5', verbose=3,
                                                         mode='max')],
        )
        model.save(f'{str(version)}/dice_ft_cee_v{str(version)}.h5')

    if not os.path.exists(f'{str(version)}/dice_ft_focal1_v{str(version)}.h5'):
        model.load_weights(f'{str(version)}/dice_ft_cee_v{str(version)}.h5')
        optim = keras.optimizers.Adam()
        model.compile(optim, dice_loss*0.01+focal_loss+cee_loss, [IOUScore()])
        model.fit_generator(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=100,
            callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=3, monitor='iou_score0.5', verbose=3,
                                                         mode='max')])
        model.save(f'{str(version)}/dice_ft_focal1_v{str(version)}.h5')

    if not os.path.exists(f'{str(version)}/dice_ft_focal2_v{str(version)}.h5'):
        model.load_weights(f'{str(version)}/dice_ft_focal1_v{str(version)}.h5')

        optim = keras.optimizers.Adam()
        model.compile(optim, dice_loss*0.001+focal_loss+cee_loss*0.1, [IOUScore()])
        model.fit_generator(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=100,
            callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=3, monitor='iou_score0.5', verbose=3,
                                                         mode='max')])
        model.save(f'{str(version)}/dice_ft_focal2_v{str(version)}.h5')

    if not os.path.exists(f'{str(version)}/dice_ft_focal3_v{str(version)}.h5'):
        model.load_weights(f'{str(version)}/dice_ft_focal2_v{str(version)}.h5')
        optim = keras.optimizers.Adam()
        model.compile(optim, dice_loss*0.0001+focal_loss+cee_loss*0.01, [IOUScore()])
        model.fit_generator(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=100,
            callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=3, monitor='iou_score0.5', verbose=3,
                                                         mode='max')])
        model.save(f'{str(version)}/dice_ft_focal3_v{str(version)}.h5')

    if not os.path.exists(f'{str(version)}/dice_ft_focal4_v{str(version)}.h5'):
        model.load_weights(f'{str(version)}/dice_ft_focal3_v{str(version)}.h5')

        optim = keras.optimizers.Adam()
        model.compile(optim, focal_loss, [IOUScore()])
        model.fit_generator(
            train_data,
            steps_per_epoch=len(train_data),
            epochs=100,
            callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=3, monitor='iou_score0.5', verbose=3,
                                                         mode='max')])
        model.save(f'{str(version)}/dice_ft_focal4_v{str(version)}.h5')