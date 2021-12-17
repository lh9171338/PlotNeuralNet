import sys
sys.path.append('../')
from pycore.blocks import *


def to_connection_rdr(of, to, pos=0.5):
    return r"""
\path (""" + of + """-east) -- (""" + to + """-west|-""" + of + """-west) coordinate[pos=""" + str(pos) + """] (""" + of + """-mid) ;
\draw [copyconnection]  (""" + of + """-east)  
-- node {\copymidarrow}(""" + of + """-mid)
-- node {\copymidarrow}(""" + to + """-west-|""" + of + """-mid)
-- node {\copymidarrow} (""" + to + """-west);
"""


def to_EncoderLayer(name, to, offset, s_filer, n_filer, size, opacity=0.5):
    return [
        to_Pool(name='pool_{}'.format(name), offset=offset, to='({}-east)'.format(to), width=1, height=size[0],
                depth=size[1], opacity=opacity),
        to_ConvConvRelu(name='ccr_{}'.format(name), offset='(0, 0, 0)', to='(pool_{}-east)'.format(name), s_filer=s_filer, n_filer=(n_filer, n_filer), width=(size[2], size[2]), height=size[0], depth=size[1]),
    ]


def to_DecoderLayer(name, to, offset1, offset2, offset3, s_filer, n_filer, size, opacity=0.5):
    return [
        to_UnPool(name='unpool_{}'.format(name), offset=offset1, to='({}-east)'.format(to), width=1, height=size[0], depth=size[1], opacity=opacity),
        to_Conv(name='conv_{}'.format(name), offset='(0, 0, 0)', to='(unpool_{}-east)'.format(name), s_filer=s_filer, n_filer=n_filer, width=size[2], height=size[0], depth=size[1]),
        to_Concat(name='cat_{}'.format(name), offset=offset2, to='(conv_{}-anchor)'.format(name)),
        to_ConvConvRelu(name='uccr_{}'.format(name), offset=offset3, to='(cat_{}-east)'.format(name), s_filer=s_filer, n_filer=(n_filer, n_filer), width=(size[2], size[2]), height=size[0], depth=size[1]),
    ]


def to_Bottleneck(name, to, offset, s_filer, n_filer, size, opacity=0.5, caption=''):
    return [
        to_Pool(name='pool_{}'.format(name), offset=offset, to='({}-east)'.format(to), width=1, height=size[0],
                depth=size[1], opacity=opacity),
        to_ConvConvRelu(name='ccr_{}'.format(name), offset='(0, 0, 0)', to='(pool_{}-east)'.format(name), s_filer=s_filer, n_filer=(n_filer, n_filer), width=(size[2], size[2]), height=size[0], depth=size[1], caption=caption),
    ]


arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Encoder
    to_ConvConvRelu(name='ccr_b1', offset='(0, 0, 0)', to='(0, 0, 0)', s_filer=512, n_filer=(64, 64), width=(2.5, 2.5),
                    height=40, depth=40),

    *to_EncoderLayer('b2', offset='(1.2, -10, 0)', to='ccr_b1', s_filer=256, n_filer=128, size=(32, 32, 3.5)),
    *to_EncoderLayer('b3', offset='(1.2, -8.5, 0)', to='ccr_b2', s_filer=128, n_filer=256, size=(25, 25, 4.5)),
    *to_EncoderLayer('b4', offset='(1.2, -6.5, 0)', to='ccr_b3', s_filer=64, n_filer=512, size=(16, 16, 6)),

    # Bottleneck
    *to_Bottleneck(name='b5', offset='(1.2, -3.0, 0)', to='ccr_b4', s_filer=32, n_filer=1024, size=(8, 8, 8), caption='Bottleneck'),

    # Decoder
    *to_DecoderLayer(name='b4', to='ccr_b5', offset1='(1, 0, 0)', offset2='(0, 3, 0)', offset3='(1.4, 0, 0)',
                     s_filer=64, n_filer=512, size=(16, 16, 6)),
    *to_DecoderLayer(name='b3', to='uccr_b4', offset1='(1, 0, 0)', offset2='(0, 6.5, 0)', offset3='(1.5, 0, 0)',
                     s_filer=128, n_filer=256, size=(25, 25, 4.5)),
    *to_DecoderLayer(name='b2', to='uccr_b3', offset1='(1, 0, 0)', offset2='(0, 8.5, 0)', offset3='(1.8, 0, 0)',
                     s_filer=256, n_filer=128, size=(32, 32, 3.5)),
    *to_DecoderLayer(name='b1', to='uccr_b2', offset1='(1, 0, 0)', offset2='(0, 10, 0)', offset3='(2, 0, 0)',
                     s_filer=512, n_filer=64, size=(40, 40, 2.5)),

    # Classifier
    to_SoftMax(name='out', offset='(2, 0, 0)', to='(uccr_b1-east)', s_filer=512,  width=1, height=40, depth=40, caption='SoftMax'),

    # Connection
    to_connection_rdr(of='ccr_b1', to='pool_b2'),
    to_connection_rdr(of='ccr_b2', to='pool_b3'),
    to_connection_rdr(of='ccr_b3', to='pool_b4'),
    to_connection_rdr(of='ccr_b4', to='pool_b5'),

    to_connection(of='ccr_b1', to='cat_b1'),
    to_connection(of='ccr_b2', to='cat_b2'),
    to_connection(of='ccr_b3', to='cat_b3'),
    to_connection(of='ccr_b4', to='cat_b4'),

    to_connection(of='cat_b1', to='uccr_b1'),
    to_connection(of='cat_b2', to='uccr_b2'),
    to_connection(of='cat_b3', to='uccr_b3'),
    to_connection(of='cat_b4', to='uccr_b4'),

    to_connection(of='conv_b1', to='cat_b1', direction='tb'),
    to_connection(of='conv_b2', to='cat_b2', direction='tb'),
    to_connection(of='conv_b3', to='cat_b3', direction='tb'),
    to_connection(of='conv_b4', to='cat_b4', direction='tb'),

    to_connection(of='ccr_b5', to='unpool_b4'),
    to_connection(of='uccr_b4', to='unpool_b3'),
    to_connection(of='uccr_b3', to='unpool_b2'),
    to_connection(of='uccr_b2', to='unpool_b1'),
    to_connection(of='uccr_b1', to='out'),

    to_end()
]

if __name__ == '__main__':
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

