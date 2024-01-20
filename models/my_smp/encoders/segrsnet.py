import torch #line:1
import torch .nn as nn #line:2
import torch .nn .functional as F #line:3
from ._base import EncoderMixin #line:4
from timm .models .layers import make_divisible #line:5
from timm .models .resnet import ResNet #line:6
class RadixSoftmax (nn .Module ):#line:9
    def __init__ (OOOO0OOO000O0O000 ,OO000000OO00O00OO ,O0O000O0O000O0OOO ):#line:10
        super (RadixSoftmax ,OOOO0OOO000O0O000 ).__init__ ()#line:11
        OOOO0OOO000O0O000 .radix =OO000000OO00O00OO #line:12
        OOOO0OOO000O0O000 .cardinality =O0O000O0O000O0OOO #line:13
    def forward (O0O00OO00O0OO00O0 ,O0OOO0O00OO000000 ):#line:15
        OO0000O0O0O0O00OO =O0OOO0O00OO000000 .size (0 )#line:16
        if O0O00OO00O0OO00O0 .radix >1 :#line:17
            O0OOO0O00OO000000 =O0OOO0O00OO000000 .view (OO0000O0O0O0O00OO ,O0O00OO00O0OO00O0 .cardinality ,O0O00OO00O0OO00O0 .radix ,-1 ).transpose (1 ,2 )#line:18
            O0OOO0O00OO000000 =F .softmax (O0OOO0O00OO000000 ,dim =1 )#line:19
            O0OOO0O00OO000000 =O0OOO0O00OO000000 .reshape (OO0000O0O0O0O00OO ,-1 )#line:20
        else :#line:21
            O0OOO0O00OO000000 =torch .sigmoid (O0OOO0O00OO000000 )#line:22
        return O0OOO0O00OO000000 #line:23
class SegRSBlock (nn .Module ):#line:26
    def __init__ (O000OO00OO0O000O0 ,OO0O000OO00O0O0OO ,out_channels =None ,kernel_size =3 ,stride =1 ,padding =None ,dilation =1 ,groups =1 ,bias =False ,radix =2 ,rd_ratio =0.25 ,rd_channels =None ,rd_divisor =8 ,act_layer =nn .GELU ,norm_layer =None ,drop_layer =None ,**O0O0OOO00OOOO0000 ,):#line:45
        super (SegRSBlock ,O000OO00OO0O000O0 ).__init__ ()#line:46
        out_channels =out_channels or OO0O000OO00O0O0OO #line:47
        O000OO00OO0O000O0 .radix =radix #line:48
        O000O00OOOO000O0O =out_channels *radix #line:49
        if rd_channels is None :#line:50
            OOOOOOOO000OOO000 =make_divisible (OO0O000OO00O0O0OO *radix *rd_ratio ,min_value =32 ,divisor =rd_divisor )#line:53
        else :#line:54
            OOOOOOOO000OOO000 =rd_channels *radix #line:55
        padding =kernel_size //2 if padding is None else padding #line:57
        O000OO00OO0O000O0 .conv =nn .Conv2d (OO0O000OO00O0O0OO ,O000O00OOOO000O0O //radix ,kernel_size ,stride ,padding ,dilation ,groups =groups *radix ,bias =bias ,**O0O0OOO00OOOO0000 ,)#line:68
        O000OO00OO0O000O0 .bn0 =norm_layer (O000O00OOOO000O0O //radix )if norm_layer else nn .Identity ()#line:69
        O000OO00OO0O000O0 .drop =drop_layer ()if drop_layer is not None else nn .Identity ()#line:70
        O000OO00OO0O000O0 .act0 =act_layer ()#line:71
        O000OO00OO0O000O0 .fc1 =nn .Conv2d (out_channels //radix ,OOOOOOOO000OOO000 ,1 ,groups =groups )#line:72
        O000OO00OO0O000O0 .bn1 =norm_layer (OOOOOOOO000OOO000 )if norm_layer else nn .Identity ()#line:73
        O000OO00OO0O000O0 .act1 =act_layer ()#line:74
        O000OO00OO0O000O0 .fc2_0 =nn .Conv2d (OOOOOOOO000OOO000 ,out_channels //radix ,1 ,groups =groups )#line:75
        O000OO00OO0O000O0 .fc2_1 =nn .Conv2d (OOOOOOOO000OOO000 ,out_channels //radix ,1 ,groups =groups )#line:76
        O000OO00OO0O000O0 .rsoftmax =RadixSoftmax (radix ,groups )#line:77
        O000OO00OO0O000O0 .pool_h =nn .AdaptiveAvgPool2d ((None ,1 ))#line:79
        O000OO00OO0O000O0 .pool_w =nn .AdaptiveAvgPool2d ((1 ,None ))#line:80
        O000OO00OO0O000O0 .att_norm =nn .GroupNorm (out_channels //radix //radix ,out_channels //radix )#line:84
        O000OO00OO0O000O0 .conv2 =nn .Conv2d (out_channels //radix ,out_channels //radix ,kernel_size =3 ,stride =1 ,padding =1 ,)#line:92
        O000OO00OO0O000O0 .gn2 =nn .GroupNorm (out_channels //radix //radix ,out_channels //radix )#line:93
        O000OO00OO0O000O0 .conv3 =nn .Conv2d (out_channels //radix ,out_channels ,kernel_size =3 ,stride =1 ,padding =1 )#line:97
        O000OO00OO0O000O0 .gn3 =nn .GroupNorm (out_channels //radix ,out_channels )#line:98
        O000OO00OO0O000O0 .act3 =act_layer ()#line:99
        O000OO00OO0O000O0 .squeeze_projector =nn .Conv2d (2 ,1 ,kernel_size =1 ,padding =0 )#line:101
        O000OO00OO0O000O0 .attn_conv =nn .Conv2d (out_channels ,out_channels ,1 )#line:102
        O000OO00OO0O000O0 .agp =nn .AdaptiveAvgPool2d ((1 ,1 ))#line:104
        O000OO00OO0O000O0 .softmax =nn .Softmax (-1 )#line:105
    def forward (OO0O000O0O0O00O0O ,OOOOOO0O0O0O0OO00 ):#line:107
        OOOOOO0O0O0O0OO00 =OO0O000O0O0O00O0O .conv (OOOOOO0O0O0O0OO00 )#line:108
        OOOOOO0O0O0O0OO00 =OO0O000O0O0O00O0O .bn0 (OOOOOO0O0O0O0OO00 )#line:109
        OOOOOO0O0O0O0OO00 =OO0O000O0O0O00O0O .drop (OOOOOO0O0O0O0OO00 )#line:110
        OOOOOO0O0O0O0OO00 =OO0O000O0O0O00O0O .act0 (OOOOOO0O0O0O0OO00 )#line:111
        OO000O000OOOO0OOO ,OOOOOO00O0OOOO0OO ,OOOO0O00O000OO000 ,OO000O00O0OOOO0OO =OOOOOO0O0O0O0OO00 .shape #line:113
        if OO0O000O0O0O00O0O .radix >1 :#line:114
            OOO0000O0OOOO00OO =OOOOOO0O0O0O0OO00 .reshape ((OO000O000OOOO0OOO *OO0O000O0O0O00O0O .radix ,-1 ,OOOO0O00O000OO000 ,OO000O00O0OOOO0OO ))#line:115
            OOOOOO0O0O0O0OO00 =OOOOOO0O0O0O0OO00 .reshape ((OO000O000OOOO0OOO ,OO0O000O0O0O00O0O .radix ,OOOOOO00O0OOOO0OO //OO0O000O0O0O00O0O .radix ,OOOO0O00O000OO000 ,OO000O00O0OOOO0OO ))#line:116
        else :#line:117
            OOO0000O0OOOO00OO =OOOOOO0O0O0O0OO00 #line:118
        OOO0O00O000OOOOOO =OO0O000O0O0O00O0O .pool_h (OOO0000O0OOOO00OO )#line:120
        O0O0OO0O00OOO0000 =OO0O000O0O0O00O0O .pool_w (OOO0000O0OOOO00OO ).permute (0 ,1 ,3 ,2 )#line:121
        OOO0000O00O0OOOOO =torch .cat ([OOO0O00O000OOOOOO ,O0O0OO0O00OOO0000 ],dim =2 )#line:123
        OOO0000O00O0OOOOO =OO0O000O0O0O00O0O .fc1 (OOO0000O00O0OOOOO )#line:124
        OOO0O00O000OOOOOO ,O0O0OO0O00OOO0000 =torch .split (OOO0000O00O0OOOOO ,[OOOO0O00O000OO000 ,OO000O00O0OOOO0OO ],dim =2 )#line:125
        O0O0OO0O00OOO0000 =O0O0OO0O00OOO0000 .permute (0 ,1 ,3 ,2 )#line:126
        O00OOOO00OOO00OO0 =OO0O000O0O0O00O0O .fc2_0 (OOO0O00O000OOOOOO ).sigmoid ()#line:127
        O0000O0OO00OOOO00 =OO0O000O0O0O00O0O .fc2_1 (O0O0OO0O00OOO0000 ).sigmoid ()#line:128
        O0O00O0OO000OO0O0 =O00OOOO00OOO00OO0 *O0000O0OO00OOOO00 *OOO0000O0OOOO00OO #line:129
        O0O00O0OO000OO0O0 =OO0O000O0O0O00O0O .att_norm (O0O00O0OO000OO0O0 )#line:130
        OOO000000OOOOOOO0 =O0O00O0OO000OO0O0 #line:132
        OOOO0O0OOO0O0OO0O =OO0O000O0O0O00O0O .conv2 (O0O00O0OO000OO0O0 )#line:133
        OO00O00O000OO00OO =OO0O000O0O0O00O0O .softmax (OO0O000O0O0O00O0O .agp (OOO000000OOOOOOO0 ).reshape (OO000O000OOOO0OOO *OO0O000O0O0O00O0O .radix ,-1 ,1 ).permute (0 ,2 ,1 ))#line:134
        O000OO0OOOO00000O =OOOO0O0OOO0O0OO0O .reshape (OO000O000OOOO0OOO *OO0O000O0O0O00O0O .radix ,OOOOOO00O0OOOO0OO //OO0O000O0O0O00O0O .radix ,-1 )#line:135
        O0O00OO000OO00O0O =OO0O000O0O0O00O0O .softmax (OO0O000O0O0O00O0O .agp (OOOO0O0OOO0O0OO0O ).reshape (OO000O000OOOO0OOO *OO0O000O0O0O00O0O .radix ,-1 ,1 ).permute (0 ,2 ,1 ))#line:136
        O0O0OOO0O0O0000O0 =OOO000000OOOOOOO0 .reshape (OO000O000OOOO0OOO *OO0O000O0O0O00O0O .radix ,OOOOOO00O0OOOO0OO //OO0O000O0O0O00O0O .radix ,-1 )#line:137
        O0O0OO00OOO000O00 =((torch .matmul (OO00O00O000OO00OO ,O000OO0OOOO00000O )+torch .matmul (O0O00OO000OO00O0O ,O0O0OOO0O0O0000O0 )).reshape (OO000O000OOOO0OOO *OO0O000O0O0O00O0O .radix ,1 ,OOOO0O00O000OO000 ,OO000O00O0OOOO0OO ).sigmoid ())#line:142
        O0O00O0OO000OO0O0 =O0O00O0OO000OO0O0 *O0O0OO00OOO000O00 #line:143
        OOOO0O0O0OO0OOO00 =torch .mean (OOOO0O0OOO0O0OO0O ,dim =1 ,keepdim =True )#line:146
        O00O0OO0000O0OO00 =torch .max (OOOO0O0OOO0O0OO0O ,dim =1 ,keepdim =True )[0 ]#line:147
        OOO0O0O000O000000 =torch .cat ([OOOO0O0O0OO0OOO00 ,O00O0OO0000O0OO00 ],dim =1 )#line:148
        OOO0O0O000O000000 =OO0O000O0O0O00O0O .squeeze_projector (OOO0O0O000O000000 ).sigmoid ()#line:149
        O0O00O0OO000OO0O0 =O0O00O0OO000OO0O0 *OOO0O0O000O000000 #line:151
        O0O00O0OO000OO0O0 =O0O00O0OO000OO0O0 .reshape ((OO000O000OOOO0OOO ,-1 ,OOOO0O00O000OO000 ,OO000O00O0OOOO0OO ))#line:152
        O0O00O0OO000OO0O0 =OO0O000O0O0O00O0O .attn_conv (O0O00O0OO000OO0O0 )#line:153
        if OO0O000O0O0O00O0O .radix >1 :#line:154
            OOOOOO0O0O0O0OO00 =OOOOOO0O0O0O0OO00 .sum (dim =1 )#line:155
        else :#line:156
            OOOOOO0O0O0O0OO00 =OOOOOO0O0O0O0OO00 .reshape ((OO000O000OOOO0OOO ,OOOOOO00O0OOOO0OO ,OOOO0O00O000OO000 ,OO000O00O0OOOO0OO ))#line:157
        OOOOOO0O0O0O0OO00 =OO0O000O0O0O00O0O .conv3 (OOOOOO0O0O0O0OO00 )#line:159
        OOOOOO0O0O0O0OO00 =OO0O000O0O0O00O0O .gn3 (OOOOOO0O0O0O0OO00 )#line:160
        OOOOOO0O0O0O0OO00 =OO0O000O0O0O00O0O .act3 (OOOOOO0O0O0O0OO00 )#line:161
        OOOOOO0O0O0O0OO00 =OOOOOO0O0O0O0OO00 *O0O00O0OO000OO0O0 #line:163
        return OOOOOO0O0O0O0OO00 #line:164
class SegRSNetBottleneck (nn .Module ):#line:167
    ""#line:168
    expansion =4 #line:170
    def __init__ (OO0O00OOO00O000OO ,O0O0O00OOOOO0OOOO ,O00O0O0O0000OOO00 ,stride =1 ,downsample =None ,radix =4 ,cardinality =1 ,base_width =64 ,avd =False ,avd_first =False ,is_first =False ,reduce_first =1 ,dilation =1 ,first_dilation =None ,act_layer =nn .GELU ,norm_layer =nn .BatchNorm2d ,attn_layer =None ,aa_layer =None ,drop_block =None ,drop_path =None ,att_dim =128 ,reduction =16 ,):#line:195
        super (SegRSNetBottleneck ,OO0O00OOO00O000OO ).__init__ ()#line:196
        assert reduce_first ==1 #line:197
        assert attn_layer is None #line:198
        assert aa_layer is None #line:199
        assert drop_path is None #line:200
        O0O0OOO0OOO0OOO00 =int (O00O0O0O0000OOO00 *(base_width /64.0 ))*cardinality #line:202
        first_dilation =first_dilation or dilation #line:203
        if avd and (stride >1 or is_first ):#line:204
            OO000O00O00O0OOO0 =stride #line:205
            stride =1 #line:206
        else :#line:207
            OO000O00O00O0OOO0 =0 #line:208
        OO0O00OOO00O000OO .radix =radix #line:209
        OO0O00OOO00O000OO .conv1 =nn .Conv2d (O0O0O00OOOOO0OOOO ,O0O0OOO0OOO0OOO00 ,kernel_size =1 ,bias =False )#line:211
        OO0O00OOO00O000OO .bn1 =nn .GroupNorm (O0O0OOO0OOO0OOO00 //radix ,O0O0OOO0OOO0OOO00 )#line:212
        OO0O00OOO00O000OO .act1 =act_layer ()#line:213
        OO0O00OOO00O000OO .avd_first =(nn .AvgPool2d (3 ,OO000O00O00O0OOO0 ,padding =1 )if OO000O00O00O0OOO0 >0 and avd_first else None )#line:218
        if OO0O00OOO00O000OO .radix >=1 :#line:220
            OO0O00OOO00O000OO .conv2 =SegRSBlock (O0O0OOO0OOO0OOO00 ,O0O0OOO0OOO0OOO00 ,kernel_size =3 ,stride =stride ,padding =first_dilation ,dilation =first_dilation ,groups =cardinality ,radix =radix ,norm_layer =norm_layer ,drop_layer =drop_block ,)#line:232
            OO0O00OOO00O000OO .bn2 =nn .Identity ()#line:233
            OO0O00OOO00O000OO .drop_block =nn .Identity ()#line:234
            OO0O00OOO00O000OO .act2 =nn .Identity ()#line:235
        else :#line:236
            OO0O00OOO00O000OO .conv2 =nn .Conv2d (O0O0OOO0OOO0OOO00 ,O0O0OOO0OOO0OOO00 ,kernel_size =3 ,stride =stride ,padding =first_dilation ,dilation =first_dilation ,groups =cardinality ,bias =False ,)#line:246
            OO0O00OOO00O000OO .bn2 =norm_layer (O0O0OOO0OOO0OOO00 )#line:247
            OO0O00OOO00O000OO .drop_block =drop_block ()if drop_block is not None else nn .Identity ()#line:248
            OO0O00OOO00O000OO .act2 =act_layer ()#line:249
        OO0O00OOO00O000OO .avd_last =(nn .AvgPool2d (3 ,OO000O00O00O0OOO0 ,padding =1 )if OO000O00O00O0OOO0 >0 and not avd_first else None )#line:254
        OO0O00OOO00O000OO .conv3 =nn .Conv2d (O0O0OOO0OOO0OOO00 ,O00O0O0O0000OOO00 *4 ,kernel_size =1 ,bias =False )#line:256
        OO0O00OOO00O000OO .bn3 =norm_layer (O00O0O0O0000OOO00 *4 )#line:257
        OO0O00OOO00O000OO .act3 =act_layer ()#line:258
        OO0O00OOO00O000OO .downsample =downsample #line:259
    def forward (O00000OO0O00O0000 ,OOO0O0OOO0OO00000 ):#line:261
        OO0OOO0O0OO0000OO =OOO0O0OOO0OO00000 #line:262
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .conv1 (OOO0O0OOO0OO00000 )#line:264
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .bn1 (O0OO00OOOO000OO0O )#line:265
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .act1 (O0OO00OOOO000OO0O )#line:266
        if O00000OO0O00O0000 .avd_first is not None :#line:268
            O0OO00OOOO000OO0O =O00000OO0O00O0000 .avd_first (O0OO00OOOO000OO0O )#line:269
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .conv2 (O0OO00OOOO000OO0O )#line:271
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .bn2 (O0OO00OOOO000OO0O )#line:272
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .drop_block (O0OO00OOOO000OO0O )#line:273
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .act2 (O0OO00OOOO000OO0O )#line:274
        if O00000OO0O00O0000 .avd_last is not None :#line:276
            O0OO00OOOO000OO0O =O00000OO0O00O0000 .avd_last (O0OO00OOOO000OO0O )#line:277
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .conv3 (O0OO00OOOO000OO0O )#line:279
        O0OO00OOOO000OO0O =O00000OO0O00O0000 .bn3 (O0OO00OOOO000OO0O )#line:280
        if O00000OO0O00O0000 .downsample is not None :#line:282
            OO0OOO0O0OO0000OO =O00000OO0O00O0000 .downsample (OOO0O0OOO0OO00000 )#line:283
        O0O0OO0OO0O0O0O00 =O0OO00OOOO000OO0O +OO0OOO0O0OO0000OO #line:285
        O0O0OO0OO0O0O0O00 =O00000OO0O00O0000 .act3 (O0O0OO0OO0O0O0O00 )#line:286
        return O0O0OO0OO0O0O0O00 #line:287
class SegRSEncoder (ResNet ,EncoderMixin ):#line:290
    def __init__ (O00O00OOO0OOOOOOO ,OOOO00OOO000O0OOO ,depth =5 ,**OO0000O00OOOOO0OO ):#line:291
        super ().__init__ (**OO0000O00OOOOO0OO )#line:292
        O00O00OOO0OOOOOOO ._depth =depth #line:293
        O00O00OOO0OOOOOOO ._out_channels =OOOO00OOO000O0OOO #line:294
        O00O00OOO0OOOOOOO ._in_channels =3 #line:295
        del O00O00OOO0OOOOOOO .fc #line:297
        del O00O00OOO0OOOOOOO .global_pool #line:298
    def get_stages (O0OOO0O00O0OOO000 ):#line:300
        return [nn .Identity (),nn .Sequential (O0OOO0O00O0OOO000 .conv1 ,O0OOO0O00O0OOO000 .bn1 ,O0OOO0O00O0OOO000 .act1 ),nn .Sequential (O0OOO0O00O0OOO000 .maxpool ,O0OOO0O00O0OOO000 .layer1 ),O0OOO0O00O0OOO000 .layer2 ,O0OOO0O00O0OOO000 .layer3 ,O0OOO0O00O0OOO000 .layer4 ,]#line:308
    def make_dilated (O000O0OOOOO00O0O0 ,*OOO000OO0O000O0O0 ,**OOOO00O0000OO0OOO ):#line:310
        raise ValueError ("SegRSNet encoders do not support dilated mode")#line:311
    def forward (O00O0O0OO00OO00OO ,OOOO000O0O0000O00 ):#line:313
        OOO0O0O0OOOOO0O0O =O00O0O0OO00OO00OO .get_stages ()#line:314
        O00O000000O0O0OO0 =[]#line:316
        for OO0OOO00O00O00000 in range (O00O0O0OO00OO00OO ._depth +1 ):#line:317
            OOOO000O0O0000O00 =OOO0O0O0OOOOO0O0O [OO0OOO00O00O00000 ](OOOO000O0O0000O00 )#line:318
            O00O000000O0O0OO0 .append (OOOO000O0O0000O00 )#line:319
        return O00O000000O0O0OO0 #line:321
    def load_state_dict (O00O0OO0O00000OOO ,O0O000O00000OOOOO ,**O00O00O0OOO00OOO0 ):#line:323
        O0O000O00000OOOOO .pop ("fc.bias",None )#line:324
        O0O000O00000OOOOO .pop ("fc.weight",None )#line:325
        super ().load_state_dict (O0O000O00000OOOOO ,**O00O00O0OOO00OOO0 )#line:326
segrsnet_weights = {
    "segrsnet-14": {
        "whu": "https://drive.google.com/file/d/1zz8jQjkIGZ1n5zl13jMbEIBGJIyISPKV",
        "mass": "https://drive.google.com/file/d/1Fwgy0TJW02F8oO6xm_2JCsZjvLLakOj8",
        "deepglobe": "https://drive.google.com/file/d/1S8ZwGuVxQJybtF005ftnu1HLsDYzFJDw",
    },
    "segrsnet-26": {
        "whu": "https://drive.google.com/file/d/17926BDB5VFAuUr1IY3W8nJ2YrH1_jb7h",
        "mass": "https://drive.google.com/file/d/1R6Q396QgVy9ESwqDFFL9VyPD31bao6lc",
        "deepglobe": "https://drive.google.com/file/d/1YpLckvr5fcTZp0EDGqfGG1rWn-vXJQX1",
    },
    "segrsnet-50": {
        "whu": "https://drive.google.com/file/d/1RZvnv-8mjWl_RNr2Z8BfLJVKs6CImluz",
        "mass": "https://drive.google.com/file/d/1D94LoMLAaW2khlohSPl9NQgTJH7B5VhS",
        "deepglobe": "https://drive.google.com/file/d/13-6kn6iFUXAi0FqDXwv9pJDnhwv_mg3c",
    },
    "segrsnet-101": {
        "whu": "https://drive.google.com/file/d/1pG3lt5E4CPARXrKmrvq-VqwGtFm1f8pz",
        "mass": "https://drive.google.com/file/d/1zr7RskKoxaeFlVjOFLsBuUlafduq_dSC",
    },
}
segrsnet_resolutions ={"segrsnet-14":{"whu":256 ,"mass":256 ,"deepglobe":1024 },"segrsnet-26":{"whu":256 ,"mass":384 ,"deepglobe":1024 },"segrsnet-50":{"whu":384 ,"mass":384 ,"deepglobe":1024 },"segrsnet-101":{"whu":384 ,"mass":384 ,},}#line:359
pretrained_settings ={}#line:361
for model_name ,sources in segrsnet_weights .items ():#line:362
    pretrained_settings [model_name ]={}#line:363
    for source_name ,source_url in sources .items ():#line:364
        pretrained_settings [model_name ][source_name ]={"url":source_url ,"input_size":[3 ,224 ,224 ],"input_range":[0 ,1 ],"mean":[0.485 ,0.456 ,0.406 ],"std":[0.229 ,0.224 ,0.225 ],"num_classes":1000 ,}#line:372
segrsnet_encoders ={"segrsnet-14":{"encoder":SegRSEncoder ,"pretrained_settings":pretrained_settings ["segrsnet-14"],"params":{"out_channels":(3 ,64 ,256 ,512 ,1024 ,2048 ),"block":SegRSNetBottleneck ,"layers":[1 ,1 ,1 ,1 ],"stem_type":"deep","stem_width":32 ,"avg_down":True ,"base_width":64 ,"cardinality":1 ,"block_args":{"radix":4 ,"avd":True ,"avd_first":False },},},"segrsnet-26":{"encoder":SegRSEncoder ,"pretrained_settings":pretrained_settings ["segrsnet-26"],"params":{"out_channels":(3 ,64 ,256 ,512 ,1024 ,2048 ),"block":SegRSNetBottleneck ,"layers":[2 ,2 ,2 ,2 ],"stem_type":"deep","stem_width":32 ,"avg_down":True ,"base_width":64 ,"cardinality":1 ,"block_args":{"radix":2 ,"avd":True ,"avd_first":False },},},"segrsnet-50":{"encoder":SegRSEncoder ,"pretrained_settings":pretrained_settings ["segrsnet-50"],"params":{"out_channels":(3 ,64 ,256 ,512 ,1024 ,2048 ),"block":SegRSNetBottleneck ,"layers":[3 ,4 ,6 ,3 ],"stem_type":"deep","stem_width":32 ,"avg_down":True ,"base_width":64 ,"cardinality":1 ,"block_args":{"radix":2 ,"avd":True ,"avd_first":False },},},"segrsnet-101":{"encoder":SegRSEncoder ,"pretrained_settings":pretrained_settings ["segrsnet-101"],"params":{"out_channels":(3 ,128 ,256 ,512 ,1024 ,2048 ),"block":SegRSNetBottleneck ,"layers":[3 ,4 ,23 ,3 ],"stem_type":"deep","stem_width":64 ,"avg_down":True ,"base_width":64 ,"cardinality":1 ,"block_args":{"radix":2 ,"avd":True ,"avd_first":False },},},}#line:435
