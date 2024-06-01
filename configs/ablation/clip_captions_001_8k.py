_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/ade20k_vpd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_8k.py'
]

model = dict(
    type='CustomVPD',
    sd_path='/home/yudzheltovskaya/custom_VPD/checkpoints/v1-5-pruned-emaonly.ckpt',
    class_embedding_path='/home/yudzheltovskaya/custom_VPD/checkpoints/class_embeddings.pth',
    caption_type='clip_captions',
    clip_captions_path='/home/yudzheltovskaya/custom_VPD/checkpoints/clip_captions_001.json',
    neck=dict(
        type='FPN',
        in_channels=[320, 790, 1430, 1280],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        num_classes=150,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),
)

lr_config = dict(policy='poly', power=1, min_lr=0.0, by_epoch=False,
                warmup='linear',
                 warmup_iters=150,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.00016, weight_decay=0.005,
        paramwise_cfg=dict(custom_keys={'unet': dict(lr_mult=0.1),
                                        'encoder_vq': dict(lr_mult=0.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=4, workers_per_gpu=2)

checkpoint_config = None
evaluation = dict(interval=10000, metric='mIoU', save_best = 'mIoU', pre_eval=True)
