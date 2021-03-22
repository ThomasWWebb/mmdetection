dataset_type = 'OPIXrayDataset'
data_root = 'datasets/opixray/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_transforms = [
    dict(type='Equalize',
    mode='cv',
    by_channels=False)]
    #dict(type='Blur')]
    #dict(type='JpegCompression', quality_lower=10, quality_upper=11)]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='custom_bboxMixUp', mixUp_prob=0.5),
    #dict(type='custom_CutMix', cutMix_prob=0.5, class_targets={1:3, 3:1}),
    #dict(type='custom_RandomCrop',crop_type='relative_range', crop_size=(0.75, 0.75)),
    dict(type='Resize', img_scale=(1225, 954), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Albu', transforms=albu_transforms),
    #dict(type='Rotate',level=10),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1225, 954),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test2017.json',
        img_prefix=data_root + 'test2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
