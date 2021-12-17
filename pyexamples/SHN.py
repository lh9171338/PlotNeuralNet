import sys
sys.path.append('../')
from pycore.blocks import *


def to_EncoderLayer(name, to, offset, s_filer, n_filer, size):
    return [
        to_Pool(name='maxpool_{}'.format(name), offset=offset, to='({}-east)'.format(to), width=1, height=size[0],
                depth=size[1], opacity=0.5),
        to_ConvRes(name='res_{}'.format(name), offset='(0, 0, 0)', to='(maxpool_{}-east)'.format(name), s_filer=s_filer,
                   n_filer=n_filer, width=size[2], height=size[0], depth=size[1]),
    ]


def to_DecoderLayer(name, to, offset1, offset2, offset3, s_filer, n_filer, size1, size2, opacity=0.5):
    return [
        to_ConvRes(name='res_{}'.format(name), offset=offset1, to='({}-east)'.format(to), s_filer=s_filer,
                   n_filer=n_filer, width=size1[2], height=size1[0], depth=size1[1], opacity=opacity),
        to_UnPool(name='upsample_{}'.format(name), offset=offset2, to='(res_{}-east)'.format(name), width=1, height=size2[0],
                  depth=size2[1], opacity=opacity),
        to_Sum(name='sum_{}'.format(name), offset=offset3, to='(upsample_{}-east)'.format(name), radius=1.5),
        to_connection(of='upsample_{}'.format(name), to='sum_{}'.format(name)),
    ]


def to_Shortcut(name, to, offset, s_filer, n_filer, size):
    return [
           to_ConvRes(name='res_{}'.format(name), offset=offset, to='({}-east)'.format(to), s_filer=s_filer,
                      n_filer=n_filer, width=size[2], height=size[0], depth=size[1]),
    ]


def to_connection_ur(of, to):
    return r"""
\draw [copyconnection]  (""" + of + """-northeast)  
-- node {\copymidarrow} (""" + to + """-west-|""" + of + """-east)
-- node {\copymidarrow} (""" + to + """-west);
"""


def to_connection_rd(of, to):
    return r"""
\draw [copyconnection]  (""" + of + """-east)  
-- node {\copymidarrow} (""" + of + """-east-|""" + to + """-north)
-- node {\copymidarrow} (""" + to + """-north);
"""


def to_connection_dru(of, to, pos=1.25):
    return r"""
\path (""" + of + """-northeast) -- (""" + of + """-southeast) coordinate[pos=""" + str(pos) + """] (""" + of + """-down) ;
\draw [copyconnection]  (""" + of + """-southeast)  
-- node {\copymidarrow} (""" + of + """-down)
-- node {\copymidarrow} (""" + of + """-down-|""" + to + """-south)
-- node {\copymidarrow} (""" + to + """-south);
"""


arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_Conv(name='conv', offset='(30, 10, 0)', to='(0, 0, 0)', s_filer='', n_filer='', width=4, height=8, depth=8, caption='Conv'),
    to_ConvRes(name='res', offset='(1, 0, 0)', to='(conv-east)', s_filer='', n_filer='', width=4, height=8, depth=8,
            caption='Residual block'),
    to_Pool(name='maxpool', offset='(1, 0, 0)', to='(res-east)', width=4, height=8, depth=8, caption='MaxPool'),
    to_UnPool(name='upsample', offset='(1, 0, 0)', to='(maxpool-east)', width=4, height=8, depth=8, caption='Up sampling'),

    to_Feature(name='input', offset='(0, 0, 0)', to='(0, 0, 0)', s_filer=512, n_filer=3, width=0.5, height=64, depth=64, caption='Input'),

    # Shallow feature extraction
    to_ConvRelu(name='conv1', offset='(1, 0, 0)', to='(input-east)', s_filer=256, n_filer=64, width=1, height=32, depth=32),
    to_ConvRes(name='layer1', offset='(1, 0, 0)', to='(conv1-east)', s_filer=256, n_filer=128, width=2, height=32, depth=32),
    to_Pool(name='maxpool', offset='(1, 0, 0)', to='(layer1-east)', width=1, height=16, depth=16),
    to_ConvRes(name='layer2', offset='(0, 0, 0)', to='(maxpool-east)', s_filer=128, n_filer=256, width=4, height=16, depth=16),
    to_ConvRes(name='layer3', offset='(1, 0, 0)', to='(layer2-east)', s_filer=128, n_filer=256, width=4, height=16, depth=16),

    # Encoder
    *to_EncoderLayer(name='encoder1', to='layer3', offset='(1.5, 0, 0)', s_filer=64, n_filer=256, size=(8, 8, 4)),
    *to_EncoderLayer(name='encoder2', to='res_encoder1', offset='(1.2, 0, 0)', s_filer=32, n_filer=256, size=(4, 4, 4)),
    *to_EncoderLayer(name='encoder3', to='res_encoder2', offset='(1, 0, 0)', s_filer=16, n_filer=256, size=(2, 2, 4)),
    *to_EncoderLayer(name='encoder4', to='res_encoder3', offset='(0.75, 0, 0)', s_filer=8, n_filer=256, size=(1, 1, 4)),

    # Skip
    *to_Shortcut(name='shortcut1', to='res_encoder4', offset='(0.5, 9, 0)', s_filer=128, n_filer=256, size=(16, 16, 4)),
    *to_Shortcut(name='shortcut2', to='res_encoder4', offset='(0.5, 5, 0)', s_filer=64, n_filer=256, size=(8, 8, 4)),
    *to_Shortcut(name='shortcut3', to='res_encoder4', offset='(0.5, 2.5, 0)', s_filer=32, n_filer=256, size=(4, 4, 4)),
    *to_Shortcut(name='shortcut4', to='res_encoder4', offset='(0.5, 1, 0)', s_filer=16, n_filer=256, size=(2, 2, 4)),
    *to_Shortcut(name='shortcut5', to='res_encoder4', offset='(0.5, 0, 0)', s_filer=8, n_filer=256, size=(1, 1, 4)),

    # Decoder
    *to_DecoderLayer(name='decoder4', to='res_shortcut5', offset1='(0.5, 0, 0)', offset2='(0, 0, 0)', offset3='(0.5, 0, 0)',
                     s_filer=8, n_filer=256, size1=(1, 1, 4), size2=(2, 2, 4)),
    *to_DecoderLayer(name='decoder3', to='sum_decoder4', offset1='(0.75, 0, 0)', offset2='(0, 0, 0)', offset3='(0.75, 0, 0)',
                     s_filer=16, n_filer=256, size1=(2, 2, 4), size2=(4, 4, 4)),
    *to_DecoderLayer(name='decoder2', to='sum_decoder3', offset1='(1, 0, 0)', offset2='(0, 0, 0)', offset3='(1, 0, 0)',
                     s_filer=32, n_filer=256, size1=(4, 4, 4), size2=(8, 8, 4)),
    *to_DecoderLayer(name='decoder1', to='sum_decoder2', offset1='(1.2, 0, 0)', offset2='(0, 0, 0)', offset3='(1.2, 0, 0)',
                     s_filer=64, n_filer=256, size1=(8, 8, 4), size2=(16, 16, 4)),

    #
    to_ConvRes(name='layer4', offset='(1, 0, 0)', to='(sum_decoder1-east)', s_filer=128, n_filer=256, width=4, height=16,
               depth=16),
    to_ConvRelu(name='conv2', offset='(1, 0, 0)', to='(layer4-east)', s_filer=128, n_filer=256, width=4, height=16,
                depth=16),
    to_Conv(name='conv3', offset='(1, 0, 0)', to='(conv2-east)', s_filer=128, n_filer=256, width=4, height=16,
                depth=16),
    to_Sum(name='sum1', offset='(1, 0, 0)', to='(conv3-east)', radius=1.5),
    to_Feature(name='feature', offset='(1, 0, 0)', to='(sum1-east)', s_filer=128, n_filer=256, width=4, height=16, depth=16, caption='Feature'),

    # Connection
    to_connection(of='input', to='conv1'),
    to_connection(of='conv1', to='layer1'),
    to_connection(of='layer1', to='maxpool'),
    to_connection(of='layer2', to='layer3'),

    to_connection(of='layer3', to='maxpool_encoder1'),
    to_connection(of='res_encoder1', to='maxpool_encoder2'),
    to_connection(of='res_encoder2', to='maxpool_encoder3'),
    to_connection(of='res_encoder3', to='maxpool_encoder4'),
    to_connection(of='res_encoder4', to='res_shortcut5'),
    to_connection(of='res_shortcut5', to='res_decoder4'),

    to_connection(of='sum_decoder4', to='res_decoder3'),
    to_connection(of='sum_decoder3', to='res_decoder2'),
    to_connection(of='sum_decoder2', to='res_decoder1'),

    to_connection(of='sum_decoder1', to='layer4'),
    to_connection(of='layer4', to='conv2'),
    to_connection(of='conv2', to='conv3'),
    to_connection(of='conv3', to='sum1'),
    to_connection_dru('layer3', 'sum1', pos=2),
    to_connection('sum1', 'feature'),

    to_connection_ur(of='layer3', to='res_shortcut1'),
    to_connection_rd(of='res_shortcut1', to='sum_decoder1'),
    to_connection_ur(of='res_encoder1', to='res_shortcut2'),
    to_connection_rd(of='res_shortcut2', to='sum_decoder2'),
    to_connection_ur(of='res_encoder2', to='res_shortcut3'),
    to_connection_rd(of='res_shortcut3', to='sum_decoder3'),
    to_connection_ur(of='res_encoder3', to='res_shortcut4'),
    to_connection_rd(of='res_shortcut4', to='sum_decoder4'),

    to_end()
]

if __name__ == '__main__':
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

