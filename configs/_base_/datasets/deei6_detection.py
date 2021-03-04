dataset_type = 'DEEI6Dataset'
data_root = 'datasets/deei6/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_transforms = [
    #dict(type='Equalize',mode='cv',by_channels=False)
    #dict(type='Blur')
    #dict(type='JpegCompression', quality_lower=10, quality_upper=11)
    ]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='custom_MixUp', mixUp_prob=0.5),
    #dict(type='custom_bboxMixUp', mixUp_prob=0.5),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='Albu', transforms=albu_transforms),
    #dict(type='custom_RandomCrop',crop_type='relative_range', crop_size=(0.75, 0.75)),
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
        img_scale=(1333, 800),
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
        ann_file=data_root + 'annotations/deei6_l_train.json',
        img_prefix=data_root + 'image/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/deei6_l_val.json',
        img_prefix=data_root + 'image/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/deei6_l_test.json',
        img_prefix=data_root + 'image/test/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
