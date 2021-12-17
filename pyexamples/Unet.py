import sys
sys.path.append('../')
from pycore.blocks import *


def to_EncoderLayer(name, to, offset, s_filer, n_filer, size, opacity=0.5):
    return [
        to_Pool(name='pool_{}'.format(name), offset=offset, to='({}-east)'.format(to), width=1, height=size[0],
                depth=size[1], opacity=opacity),
        to_ConvConvRelu(name='ccr_{}'.format(name), offset='(0, 0, 0)', to='(pool_{}-east)'.format(name), s_filer=s_filer, n_filer=(n_filer, n_filer), width=(size[2], size[2]), height=size[0], depth=size[1]),
    ]


def to_DecoderLayer(name, to, offset, s_filer, n_filer, size, opacity=0.5):
    return [
        to_UnPool(name='unpool_{}'.format(name), offset=offset, to='({}-east)'.format(to), width=1, height=size[0], depth=size[1], opacity=opacity),
        to_Conv(name='conv_{}'.format(name), offset='(0, 0, 0)', to='(unpool_{}-east)'.format(name), s_filer='', n_filer=n_filer, width=size[2], height=size[0], depth=size[1]),
        to_ConvRes(name='ccr_res_{}'.format(name), offset='(0, 0, 0)', to="(conv_{}-east)".format(name), s_filer='', n_filer=n_filer, width=size[2], height=size[0], depth=size[1], opacity=opacity),
        to_ConvConvRelu(name='uccr_{}'.format(name), offset='(0, 0, 0)', to='(ccr_res_{}-east)'.format(name), s_filer=s_filer, n_filer=(n_filer, n_filer), width=(size[2], size[2]), height=size[0], depth=size[1]),
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
    to_ConvConvRelu(name='ccr_b1', s_filer=512, n_filer=(64, 64), offset='(0, 0, 0)', to='(0, 0, 0)', width=(2.5, 2.5),
                    height=40, depth=40),
    *to_EncoderLayer(name='b2', to='ccr_b1', offset='(2, 0, 0)', s_filer=256, n_filer=128, size=(32, 32, 3.5)),
    *to_EncoderLayer(name='b3', to='ccr_b2', offset='(1.5, 0, 0)', s_filer=128, n_filer=256, size=(25, 25, 4.5)),
    *to_EncoderLayer(name='b4', to='ccr_b3', offset='(1, 0, 0)', s_filer=64, n_filer=512, size=(16, 16, 6)),

    # Bottleneck
    *to_Bottleneck(name='b5', to='ccr_b4', offset='(0.75, 0, 0)', s_filer=32, n_filer=1024, size=(8, 8, 8), caption='Bottleneck'),

    # Decoder
    *to_DecoderLayer(name='b4', to='ccr_b5', offset='(1.2, 0, 0)', s_filer=64, n_filer=512, size=(16, 16, 6)),
    *to_DecoderLayer(name='b3', to='uccr_b4', offset='(1.5, 0, 0)', s_filer=128, n_filer=256, size=(25, 25, 4.5)),
    *to_DecoderLayer(name='b2', to='uccr_b3', offset='(1, 0, 0)', s_filer=256, n_filer=128, size=(32, 32, 3.5)),
    *to_DecoderLayer(name='b1', to='uccr_b2', offset='(1.5, 0, 0)', s_filer=512, n_filer=64, size=(40, 40, 2.5)),

    # Classifier
    to_ConvSoftMax(name='out', s_filer=512, offset='(0.75, 0, 0)', to='(uccr_b1-east)', width=1, height=40, depth=40,
                   caption='SoftMax'),

    # Connection
    to_connection(of='ccr_b1', to='pool_b2'),
    to_connection(of='ccr_b2', to='pool_b3'),
    to_connection(of='ccr_b3', to='pool_b4'),
    to_connection(of='ccr_b4', to='pool_b5'),

    to_connection(of='ccr_b5', to='unpool_b4'),
    to_connection(of='uccr_b4', to='unpool_b3'),
    to_connection(of='uccr_b3', to='unpool_b2'),
    to_connection(of='uccr_b2', to='unpool_b1'),
    to_connection(of='uccr_b1', to='out'),

    to_skip(of='ccr_b1', to='ccr_res_b1', pos=1.25),
    to_skip(of='ccr_b2', to='ccr_res_b2', pos=1.25),
    to_skip(of='ccr_b3', to='ccr_res_b3', pos=1.25),
    to_skip(of='ccr_b4', to='ccr_res_b4', pos=1.25),

    to_end() 
    ]


if __name__ == '__main__':
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')
    
