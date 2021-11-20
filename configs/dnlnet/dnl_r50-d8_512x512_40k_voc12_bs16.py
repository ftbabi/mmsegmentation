_base_ = [
    '../_base_/models/dnl_r50-d8.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))

dist_params = dict(port='295052')

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0005)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,)