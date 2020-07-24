from segmentation_models.base import Loss
from segmentation_models.base import functional as F
from segmentation_models.base import Metric
from segmentation_models.base import functional as F
from segmentation_models.base import Loss
from segmentation_models.base import Loss
import keras

# we have additional channel in masks so we have to rewrite loss fuctions
class DiceLoss(Loss):

    def __init__(self, beta=1, class_weights=None, class_indexes=None, per_image=False):
        super().__init__(name='dice_loss')
        self.beta = beta
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.per_image = per_image
        self.smooth = 1e-5

    def __call__(self, gt, pr):
        gt = gt[:, :, :, :1]
        loss = 1-F.f_score(
            gt,
            pr,
            beta=self.beta,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=None,
            **self.submodules
        )
        return loss




class BinaryFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.):
        super().__init__(name='binary_focal_loss')
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, gt, pr):
        gt = gt[:, :, :, :1]
        return F.binary_focal_loss(gt, pr, alpha=self.alpha, gamma=self.gamma, **self.submodules)






class WeightedBinaryCELoss(Loss):
    def __init__(self):
        super().__init__(name='weighted_binary_crossentropy')

    def __call__(self, gt, pr):
        weights = gt[:, :, :, 1:]
        gt = gt[:, :, :, :1]
        per_pixel_loss = keras.backend.binary_crossentropy(gt, pr)
        return keras.backend.mean(per_pixel_loss * weights)


class IOUScore(Metric):

    def __init__(self,threshold=0.5, limit = None):
        name ='iou_score' + str(threshold)
        if limit is not None:
            name = 'iou_score' + str(threshold) + '_' + str(limit)

        super().__init__(name=name)
        self.class_weights =  1
        self.class_indexes = None
        self.threshold = threshold
        self.per_image = True
        self.smooth = 1e-5
        self.limit = limit

    def __call__(self, gt, pr):
        gt = gt[:, :, :, :1]
        if self.limit is not None:
            gt=gt[:,:self.limit, :self.limit, :]
            pr=pr[:,:self.limit, :self.limit, :]

        return F.iou_score(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            smooth=self.smooth,
            per_image=self.per_image,
            threshold=self.threshold,
            **self.submodules
        )