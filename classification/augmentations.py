import albumentations as A


def get_augmentations():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1),
        A.PadIfNeeded(min_height=512, min_width=640, p=1.0),
        A.CenterCrop(height=512, width=640, p=1.0),
        A.ShiftScaleRotate(rotate_limit=90, shift_limit=0.5, scale_limit=0.1,p=0.1)

    ]

    return A.Compose(train_transform)
