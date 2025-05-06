from projects.Adabins.configs.adabins.adabins_efficient_b5_4x16_25e_NYU_416x544 import norm_cfg



model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(type='MixVisionTransformer',
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
                             norm_cfg=None,
                         )
                     ),
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                     )),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)