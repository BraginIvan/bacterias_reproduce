from .losses import DiceLoss
from .augmentation import get_augmentations
import keras
import segmentation_models as sm
from .data_loader import Dataset, Dataloder

BATCH_SIZE = 4
IMG_WIGHT = 640
IMG_HEIGHT = 512

def run(config, version):
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
                    encoder_freeze=True,
                    )

    dice_loss = DiceLoss(beta=1.0, per_image=True)

    # warmup decoder
    optim = keras.optimizers.Adam()
    model.compile(optim, dice_loss)
    model.fit_generator(
        train_data,
        steps_per_epoch=len(train_data),
        epochs=10)
    for layer in model.layers:
        layer.trainable = True

    # training model
    optim = keras.optimizers.Adam()
    model.compile(optim, dice_loss)
    model.fit_generator(
        train_data,
        steps_per_epoch=len(train_data),
        epochs=200,
        callbacks=[keras.callbacks.ReduceLROnPlateau(factor=0.7, patience=7, monitor='iou_score0.5', verbose=3,
                                                     mode='max')]
    )

    model.save(f'dice_v{str(version + 1)}.h5')
