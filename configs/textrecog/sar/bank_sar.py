_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adamw_600e_cos.py'
]

data_root = "/usr/videodate/yehc/2021A_T1_Task1/"
dict_file = data_root + "train/dict_file.txt"
label_convertor = dict(
    type='AttnConvertor', dict_file=dict_file, with_unknown=True)

model = dict(
    type='SARNet',
    backbone=dict(type='ResNet31OCR'),
    encoder=dict(
        type='SAREncoder',
        enc_bi_rnn=False,
        enc_do_rnn=0.1,
        enc_gru=False,
    ),
    decoder=dict(
        type='ParallelSARDecoder',
        enc_bi_rnn=False,
        dec_bi_rnn=False,
        dec_do_rnn=0,
        dec_gru=False,
        pred_dropout=0.1,
        d_k=512,
        pred_concat=True),
    loss=dict(type='SARLoss'),
    label_convertor=label_convertor,
    max_seq_len=30)

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=256,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiRotateAugOCR',
        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=48,
                min_width=48,
                max_width=256,
                keep_aspect_ratio=True,
                width_downsample_ratio=0.25),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'resize_shape', 'valid_ratio'
                ]),
        ])
]

dataset_type = 'OCRDataset'

train_prefix = data_root + "train/"

train_ann_file = data_root + 'train/train_gt.txt'

train = dict(
    type=dataset_type,
    img_prefix=train_prefix,
    ann_file=train_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

test_prefix = data_root + "train/"

test_ann_file = data_root + 'train/val_gt.txt'

test = dict(
    type=dataset_type,
    img_prefix=test_prefix,
    ann_file=test_ann_file,
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

data = dict(
    samples_per_gpu=40,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset', datasets=[train],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset', datasets=[test], pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset', datasets=[test], pipeline=test_pipeline))

evaluation = dict(interval=5, metric='acc')
checkpoint_config = dict(interval=5)
resume_from = "./work_dirs/bank_sar/latest.pth"
