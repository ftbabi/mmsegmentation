# model settings
_base_ = [
    '../_base_/models/fastfcn_r50-d32_jpu_psp.py',
    '../_base_/datasets/pascal_voc12.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    # Modify here
    auxiliary_head=dict(num_classes=20),
    decode_head=dict(
        _delete_=True,
        type='EncHead',
        in_channels=[512, 1024, 2048],
        in_index=(0, 1, 2),
        channels=512,
        num_codes=32,
        use_se_loss=True,
        add_lateral=False,
        dropout_ratio=0.1,
        # Modify here
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_se_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# Batch-size = 8, using 2 GPUS
data = dict(samples_per_gpu=4, workers_per_gpu=4)
dist_params = dict(port='29507')