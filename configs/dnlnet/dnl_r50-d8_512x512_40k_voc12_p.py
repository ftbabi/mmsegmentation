_base_ = [
    '../_base_/models/dnl_r50-d8.py', '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21, abl_mode='p'),
    auxiliary_head=dict(num_classes=21))

dist_params = dict(port='295051')
find_unused_parameters = True