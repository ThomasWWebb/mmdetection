...
# dataset settings
dataset_type = 'CocoDataset'
classes = ('Folding_Knife', "Straight_Knife", "Scissor", "Utility_Knife", "Multi-tool_Knife")
...
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../drive/My Drive/L4 Project/OPIXray/image/train'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../drive/My Drive/L4 Project/OPIXray/image/test'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='../drive/My Drive/L4 Project/OPIXray/image/test')
)