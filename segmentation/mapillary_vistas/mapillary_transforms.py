"""Small MMSeg transforms used by the local Mapillary experiment."""

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class UseResizedShapeForEvaluation(BaseTransform):
    """Keep validation logits/labels at the aspect-preserving resize shape.

    Mapillary source images can be very large. MMSeg's normal post-processing
    restores all 124 logit planes to source resolution before argmax, which
    exceeds an 8 GB GPU. The validation label is resized with nearest-neighbor
    first, and this transform tells the segmentor that the resized shape is the
    evaluation target. Target-domain inference still restores the categorical
    argmax mask to source size on CPU.
    """

    def transform(self, results: dict) -> dict:
        results['ori_shape'] = results['img_shape']
        return results

