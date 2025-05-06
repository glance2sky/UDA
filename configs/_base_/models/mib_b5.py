norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    # init_cfg=dict(type='pretrained',checkpoint='pretrained/mit_b5.pth'),
    backbone=dict(type='mit_b5', style='pytorch'),
    # hy_conv=dict(type='conv_layer'),
    decode_head=dict(
        type='DAFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(
            embed_dims=256,
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            fusion_cfg=dict(
                type='aspp',
                sep=True,
                act_cfg=dict(type='ReLU'),
                dilations=(1, 6, 12, 18),
                pool=False,
                norm_cfg=norm_cfg),
        ),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='whole')
    # model training and testing settings
)