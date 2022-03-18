import sys
sys.path.append('..')
from pycore.blocks import *


def to_Layer(name, offset, to, dim, size, opacity=0.8):
    return [
        to_Fc(name='fc_{}'.format(name), offset=offset, to='({}-east)'.format(to), dim='', width=size[0], height=size[1],
                depth=size[2]),
        to_Relu(name='relu_{}'.format(name), offset='(0, 0, 0)', to='(fc_{}-east)'.format(name), width=1, height=size[1],
                depth=size[2], opacity=opacity),
        to_Bn1d(name='bn_{}'.format(name), offset='(0, 0, 0)', to='(relu_{}-east)'.format(name), dim=dim, width=1, height=size[1],
                depth=size[2], opacity=opacity),
    ]


arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    to_Fc(name='fc', offset='(13, 4, 0)', to='(0, 0, 0)', dim='', width=2, height=2, depth=8, caption='Fc'),
    to_Relu(name='relu', offset='(1, 0, 0)', to='(fc-east)', width=2, height=2, depth=8, opacity=1, caption='Relu'),
    to_Bn1d(name='bn', offset='(1, 0, 0)', to='(relu-east)', dim='', width=2, height=2, depth=8, opacity=1, caption='Bn'),
    to_Sigmoid1d(name='sigmoid', offset='(1, 0, 0)', to='(bn-east)', dim='', width=2, height=2, depth=8, opacity=1, caption='Sigmoid'),

    to_Feature1d(name='input', dim=5, offset='(0, 0, 0)', to='(0, 0, 0)', width=2, height=2, depth=16, caption='Input'),

    *to_Layer(name='layer1', offset='(2, 0, 0)', to='input', dim=512, size=(2, 2, 96)),
    *to_Layer(name='layer2', offset='(1.8, 0, 0)', to='bn_layer1', dim=256, size=(2, 2, 64)),
    *to_Layer(name='layer3', offset='(1.8, 0, 0)', to='bn_layer2', dim=96, size=(2, 2, 48)),
    *to_Layer(name='layer4', offset='(1.8, 0, 0)', to='bn_layer3', dim=16, size=(2, 2, 32)),
    *to_Layer(name='layer5', offset='(1.8, 0, 0)', to='bn_layer4', dim=2, size=(2, 2, 8)),
    to_Fc(name='layer6', offset='(1.8, 0, 0)', to='(bn_layer5-east)', dim='', width=2, height=2, depth=4),
    to_Sigmoid1d(name='sigmoid', offset='(0, 0, 0)', to='(layer6-east)', dim=1, width=1, height=2, depth=4, opacity=1),

    to_Feature1d(name='output', dim=1, offset='(2, 0, 0)', to='(sigmoid-east)', width=2, height=2, depth=4, caption='Score'),

    # Connection
    to_connection(of='input', to='fc_layer1'),
    to_connection(of='bn_layer1', to='fc_layer2'),
    to_connection(of='bn_layer2', to='fc_layer3'),
    to_connection(of='bn_layer3', to='fc_layer4'),
    to_connection(of='bn_layer4', to='fc_layer5'),
    to_connection(of='bn_layer5', to='layer6'),
    to_connection(of='sigmoid', to='output'),

    to_end()
]

if __name__ == '__main__':
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')

