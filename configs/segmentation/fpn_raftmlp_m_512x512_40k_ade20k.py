_base_ = [
    './_base_/backbones/raftmlp.py',
    './_base_/models/fpn_r50.py',
    './_base_/datasets/ade20k.py',
    './_base_/schedules/schedule40k.py',
    './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='SegRaftMLPMedium',
        init_cfg=dict(type='Pretrained', checkpoint='/datasets/weights/imagenet-raft-mlp-cross-mlp-emb-m/last_model_0.pt')),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150))

log_config = dict(
    interval=50,
    hooks=[
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='raftmlp-fpn',
                name='imagenet-raft-mlp-cross-mlp-emb-m'
            )
        )
    ])
