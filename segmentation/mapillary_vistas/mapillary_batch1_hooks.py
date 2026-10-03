"""Hooks required for batch-one DeepLabV3+ whole-image training."""

from mmengine.hooks import Hook
from mmseg.registry import HOOKS


@HOOKS.register_module()
class FreezeASPPImagePoolNormHook(Hook):
    """Keep the 1x1 ASPP image-pooling normalization in evaluation mode.

    All other decoder normalization layers still train normally.  The global
    pooling branch has only one value per channel at batch one, where batch
    normalization cannot estimate a variance.
    """

    priority = 'VERY_HIGH'

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        image_pool_conv = model.decode_head.image_pool[1]
        image_pool_conv.norm.eval()
