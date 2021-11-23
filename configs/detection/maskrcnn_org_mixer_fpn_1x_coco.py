_base_ = [
    './_base_/backbones/raftmlp.py',
    './_base_/models/mask_rcnn_r50_fpn.py',
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='DetOrgMixer',
        init_cfg=dict(type='Pretrained', checkpoint='/datasets/weights/imagenet-org-mixer/last_model_0.pt')),
    neck=dict(
        type='FPN',
        in_channels=[768, 768, 768, 768],
        out_channels=256,
        num_outs=5))

log_config = dict(
    interval=50,
    hooks=[
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='raftmlp-maskrcnn',
                name='imagenet-org-mixer'
            )
        )
    ])
