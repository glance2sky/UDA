
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='MixVisionTransformer2',
                  embed_dims=64,
                  num_heads=[1,2,5,8],
                  num_layers=[3,6,4,3],
                  ),
    decode_head=dict(type='DAFormerHead',
                     in_channels=[64, 128, 320, 512],
                     in_index=[0, 1, 2, 3],
                     channels=256,
                     dropout_ratio=0.1,
                     num_classes=19,
                     align_corners=False,
                     norm_cfg=None,
                     decoder_params=dict(
                        embed_dims=256,
                        embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                        embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                         fusion_cfg=dict(
                             type='aspp',
                             sep=True,
                             dilations=(1,6,12,18),
                             pool=False,
                             act_cfg=dict(type='ReLU'),
                             norm_cfg=norm_cfg,
                         )
                     ),
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                     )),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)