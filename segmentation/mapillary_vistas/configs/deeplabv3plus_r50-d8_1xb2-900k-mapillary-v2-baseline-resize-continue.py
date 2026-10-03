"""Strict continuation of the baseline-resize run from iteration 450k to 900k.

The runner resumes ``iter_450000.pth`` from the existing work directory.
The PolyLR schedule deliberately retains its original 450k endpoint: its saved
state has already reached ``eta_min=1e-4``, so the additional iterations are a
low-learning-rate continuation rather than a newly restarted schedule.
"""

_base_ = ['./deeplabv3plus_r50-d8_1xb2-450k-mapillary-v2-baseline-resize.py']

# ``--resume`` must discover ``last_checkpoint`` rather than treat the
# original ADE20K initialization checkpoint as the resume source.
load_from = None

train_cfg = dict(type='IterBasedTrainLoop', max_iters=900000, val_interval=4000)

# Kept identical to the resumed scheduler state stored in iter_450000.pth.
param_scheduler = [
    dict(type='PolyLR', eta_min=0.0001, power=0.9, begin=0, end=450000, by_epoch=False),
]
