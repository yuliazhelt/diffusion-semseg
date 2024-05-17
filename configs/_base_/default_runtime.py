# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='MMSegWandbHook',
        init_kwargs={
            'project': "thesis_vpd",
            # name added after config parsing
        },
        num_eval_images=0, # > 0 works only in online wandb mode
        by_epoch=False),
        dict(type='TextLoggerHook', by_epoch=False),
    ]
)
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
