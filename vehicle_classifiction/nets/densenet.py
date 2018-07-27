"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): 
    return tf.truncated_normal_initializer(stddev=stddev)

# 复合函数，包括归一(bn)、激活(relu)、池化(conv)和dropout
def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):      # num_outputs: 输出通道数，kernel_size：卷积核大小
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current

# 层连接函数(xl= H([x0,x1,...xl-1]))，新生成的层tmp加入到输入层net，形成新的输入层，经过layers次连接后，最后得到的net作为输出层
# layers次连接后net的输出通道数：growth0 + growth * layers
def block(net, layers, growth, scope='block'):                                               # layers: dense block包含的网络层数 
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],scope=scope + '_conv1x1' + str(idx)) # bn-relu-conv2d-dropput(1,1)
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],scope=scope + '_conv3x3' + str(idx))     # bn-relu-conv2d-dropput(3,3)
        net = tf.concat(axis=3, values=[net, tmp])                                                # 原net与tmp连接,作为新的net
    return net

# 稠密链接函数
def densenet(images, num_classes=1001, is_training=False, dropout_keep_prob=0.8,scope='densenet'):    
  
    growth = 24                 # 增长率 growth rate
    compression_rate = 0.5
    
    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):       
        with slim.arg_scope(bn_drp_scope(is_training=is_training,keep_prob=dropout_keep_prob)) as ssc:
            
            #  初始卷积层的输出通道数是2*growth = 48,先是(7,7)卷积，图片变为(112 *112)；再(3,3)池化，图片变为(55 * 55)
            current = end_points['pre_conv2'] = slim.conv2d(images, 2*growth, [7, 7], stride=2, padding='same', scope='pre_conv2')
            current = end_points['pre_pool2'] = slim.max_pool2d(current, [3, 3], stride=2, scope='pre_pool2')
      
            #  4个block,每个block包括(1,1)卷积和(3,3)卷积,前3个block之后(1,1)卷积和(2,2)池化作为过渡层,第4个block之后（6,6）池化
            #  图片变为(27 * 27)
            current = end_points['block1'] = block(current, 6, growth, scope='block1')         
            current = end_points['transition1_conv2'] = bn_act_conv_drp(current, growth, [1, 1], scope='transition1_conv2')
            current = end_points['transition1_pool2'] = slim.avg_pool2d(current, [2, 2], stride=2, scope='transition1_pool2') 
            
            #  图片变为(13 * 13)                                               
            current = end_points['block2'] = block(current, 12, growth, scope='block2')      
            current = end_points['transition2_conv2'] = bn_act_conv_drp(current, growth, [1, 1], scope='transition2_conv2')
            current = end_points['transition2_pool2'] = slim.avg_pool2d(current, [2, 2], stride=2, scope='transition2_pool2') 
            
            #  图片变为(6 * 6) 
            current = end_points['block3'] = block(current, 24, growth, scope='block3')
            current = end_points['transition3_conv2'] = bn_act_conv_drp(current, growth, [1, 1], scope='transition3_conv2')
            current = end_points['transition3_pool2'] = slim.avg_pool2d(current, [2, 2], stride=2, scope='transition3_pool2') 

            #  图片变为(1 * 1) 
            current = end_points['block4'] = block(current, 16, growth, scope='block4')	      
            current = end_points['global_pool2'] = slim.avg_pool2d(current, [6, 6], scope='global_pool2') 
            
            #  全连接                                                                                                        
            current = end_points['PreLogitsFlatten'] = slim.flatten(current, scope='PreLogitsFlatten')
            logits = end_points['Logits'] = slim.fully_connected(current, num_classes, activation_fn=None, scope='Logits')

            # softmax分类
            end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')

    return logits, end_points

def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope([slim.batch_norm], scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope([slim.dropout],is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc

def densenet_arg_scope(weight_decay=0.004):
    with slim.arg_scope([slim.conv2d],
                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN',uniform=False),
                    activation_fn=None, biases_initializer=None, padding='same',stride=1) as sc:
         return sc

densenet.default_image_size = 224
