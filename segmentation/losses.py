from segmentation_models.base import Loss
from segmentation_models.base import functional as F

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