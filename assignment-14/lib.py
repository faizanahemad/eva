from tensorflow.keras.layers import BatchNormalization, Layer,InputSpec
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.initializers import Initializer
from tensorflow.python.keras.backend import _regular_normalize_batch_in_training
import tensorflow.keras.backend as K
import tensorflow
import tensorflow as tf
import math
import gc

import logging

logger = logging.getLogger("App")
logger.setLevel(logging.INFO)
# logger.warning('This is a warning')

###### BatchNorm to be Used #####
# BatchNorm = BatchNormalizationF16

BatchNorm = tf.keras.layers.BatchNormalization
    
    

from tensorflow.keras.backend import set_session
from tensorflow.keras.backend import clear_session
from tensorflow.keras.backend import get_session
import tensorflow
import tensorflow as tf

# Reset Keras Session
def reset_keras(config_builder=None):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    _ = gc.collect() # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    # config = tensorflow.ConfigProto(log_device_placement=True, allow_soft_placement=True,)
    config = tf.compat.v1.ConfigProto() if config_builder is None else config_builder()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))
    

def init_pytorch(shape, dtype=tf.float32, partition_info=None):
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

class ConvBN(tf.keras.Model):
    def __init__(self, c_out, kernel_size=3, bn=False, strides=1, activation="relu", spatial_dropout=0.0, use_thin_conv=False):
        super().__init__()
        if use_thin_conv:
            tf.keras.layers.SeparableConv2D(filters=c_out,depth_multiplier=2, kernel_size=kernel_size,
                                            strides=strides, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
        else:
            self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=kernel_size,
                                               strides=strides, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
        self.bn = bn
        self.activation = activation
        self.spatial_dropout = spatial_dropout
        self.sd = tf.keras.layers.SpatialDropout2D(self.spatial_dropout)
        
        if bn:
            self.bn = BatchNorm(momentum=0.9, epsilon=1e-7)

    def call(self, inputs):
        if self.bn:
            res = self.bn(self.conv(inputs))
        else:
            res = self.conv(inputs)
            
        if self.spatial_dropout>0:
            res = self.sd(res)
            
        if self.activation=="relu":
            res = tf.nn.relu(res)
        return res
    
    

class ResBlk(tf.keras.Model):
    def __init__(self, c_out, pool, res = False, bn=False, 
                 no_activation_first_conv=False, residual_dropout=0.0,spatial_dropout=0.0, use_thin_conv=False):
        super().__init__()
        self.conv_bn = ConvBN(c_out, bn=bn, spatial_dropout=spatial_dropout,
                              activation="linear" if no_activation_first_conv else "relu")
        self.pool = pool
        self.res = res
        self.residual_dropout = residual_dropout
        if self.res:
            self.res1 = ConvBN(c_out, bn=bn, use_thin_conv=use_thin_conv)
            self.res2 = ConvBN(c_out, bn=bn, use_thin_conv=use_thin_conv)

    def call(self, inputs):
        h = self.pool(self.conv_bn(inputs))
        if self.res:
            p_1 = tf.random.uniform([1],minval=0,maxval=1,dtype=tf.dtypes.float32,)
            cond = tf.keras.backend.less(p_1,self.residual_dropout)
            cond = tf.broadcast_to(cond,[inputs.shape[0]])
            h = tf.keras.backend.switch(cond,h,h + self.res2(self.res1(h)))
            
#             p_1 = np.random.rand()
#             if p_1 >= self.residual_dropout:
#                 print("Trying Residual")
#                 h = h + self.res2(self.res1(h))
#             else:
#                 h = h
#                 print("Not Trying Residual")
        return h
    
    

    

# Batch Size
# Maxpool after 1st conv

class FNet(tf.keras.Model):
    def __init__(self, start_kernels=64, weight=0.125, sparse_bn = True, thin_block = False, 
                 enable_skip=False, enable_pool_before_skip=False, no_activation_first_conv=False, 
                 residual_dropout=0.0, spatial_dropout=0.0):
        super().__init__()
        c = start_kernels
        pool = tf.keras.layers.MaxPooling2D()
        bn = not sparse_bn
        self.init_conv_bn = ConvBN(c, kernel_size=3, bn=bn)
        self.enable_skip = enable_skip
        self.enable_pool_before_skip = enable_pool_before_skip
        if enable_skip:
            self.skip = ConvBN(c*2,kernel_size=1, strides=1, bn=bn, activation="relu")
            self.avg_pool = tf.keras.layers.AveragePooling2D()
            self.max_pool = pool
        
        
        self.blk1 = ResBlk(c*2, pool, res = True, bn=bn,use_thin_conv=thin_block,
                           no_activation_first_conv=no_activation_first_conv,
                           residual_dropout=residual_dropout, spatial_dropout=spatial_dropout,)
        self.blk2 = ResBlk(c*4, pool, bn=bn,use_thin_conv=thin_block,
                           no_activation_first_conv=no_activation_first_conv, 
                           residual_dropout=residual_dropout, spatial_dropout=spatial_dropout,)
        self.blk3 = ResBlk(c*8, pool, res = True, bn=bn,use_thin_conv=thin_block, 
                           no_activation_first_conv=no_activation_first_conv, 
                           residual_dropout=residual_dropout, spatial_dropout=spatial_dropout,)
        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
        self.weight = weight
        self.sparse_bn = sparse_bn
        self.concat = tf.keras.layers.Concatenate()
        if sparse_bn:
            self.bn1 = BatchNorm(momentum=0.9, epsilon=1e-7)
            self.bn2 = BatchNorm(momentum=0.9, epsilon=1e-7)
            self.bn3 = BatchNorm(momentum=0.9, epsilon=1e-7)
            self.bn4 = BatchNorm(momentum=0.9, epsilon=1e-7)

    def call(self, x, y):
        h = self.init_conv_bn(x)
        h = self.bn1(h) if self.sparse_bn else h
        
        h = self.blk1(h)
        h = self.bn2(h) if self.sparse_bn else h

        if self.enable_skip:  
            if self.enable_pool_before_skip:
                k = self.pool(self.skip(self.avg_pool(h)))
            else:
                k = self.pool(self.skip(h))
       
        h = self.blk2(h)
        h = self.bn3(h) if self.sparse_bn else h
        
        h = self.blk3(h)
        h = self.pool(h)
        if self.enable_skip:
            h = self.concat([h,k])
        h = self.bn4(h) if self.sparse_bn else h
        
        h = self.linear(h) * self.weight
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
        loss = tf.reduce_sum(ce)
        correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
        return loss, correct
    
def make_plots():
    plt.figure(figsize=(12,4))
    plt.title("Train acc = %.3f, Test acc = %.3f"%(train_accs[-1],test_accs[-1]))
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.show()


    plt.figure(figsize=(12,4))
    plt.title("Train acc = %.3f, Test acc = %.3f"%(train_accs[-1],test_accs[-1]))
    plt.plot(list(range(3,len(train_accs))),train_accs[3:])
    plt.plot(list(range(3,len(test_accs))),test_accs[3:])
    plt.show()
    return


from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

class CIFAR10Policy(object):
    def __init__(self, proba=1.0, fillcolor=(128, 128, 128),
                 enabled_policies=[("rotate",5, 15,),("shearX",0.1, 0.2,),("shearY",0.1, 0.2,)], 
                 log=False):
        
        self.enabled_policies = enabled_policies
        self.proba = proba
        self.policies = [SubPolicy(policy[0],policy[1],policy[2], fillcolor) for policy in enabled_policies]
        def log_fn_enabled(*args):
            print(*args)


        def log_fn_disabled(*args):
            pass
        self.log_fn = log_fn_enabled if log else log_fn_disabled


    def __call__(self, img, policy_idx=None, magnitude_idx=None):
        p_1 = np.random.rand()
        if p_1 > self.proba:
            return img
        policy_idx = random.randint(0, len(self.policies) - 1) if policy_idx is None else policy_idx
        policy = self.policies[policy_idx]
        magnitude_idx = np.random.randint(0,10) if magnitude_idx is None else magnitude_idx
        self.log_fn("Policy = ",self.enabled_policies[policy_idx], "Magnitude ID = ",magnitude_idx, "Magnitude = ",policy.ranges[magnitude_idx])
        img = Image.fromarray(np.uint8(img),'RGB')
        img = policy(img,magnitude_idx)
        img = np.array(img)
        return img


class SubPolicy(object):
    def __init__(self, operation1, magnitude_start,magnitude_end, fillcolor=(128, 128, 128)):
        mst = magnitude_start
        mend = magnitude_end
        self.ranges = np.round(np.linspace(mst, mend, 10), 0).astype(np.int) if operation1=="posterize" else np.linspace(mst, mend, 10)

        
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, tuple(list(fillcolor) + [128])), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
        }

        self.operation1 = func[operation1]
        self.op1_name = operation1
        
        

    def __call__(self, img, magnitude_idx=None):
        magnitude1 = self.ranges[magnitude_idx]
        img = self.operation1(img, magnitude1)
        return img
    
    
def get_numpy_wrapper(fn,name=None, Tout=None):
    def wrapper(x,y):
        x = tf.numpy_function(fn,[x],Tout=x.dtype if Tout is None else Tout)
        return x,y
    return wrapper

def get_multimapper(mappers):
    def mapper(x,y):
        for m_fn in mappers:
            x,y = m_fn(x,y)
        return x,y
    return mapper


def get_hue_aug(max_delta):
    return lambda x,y: (tf.image.random_hue(x,max_delta=max_delta),y)


hflip_mapper = lambda x, y: (tf.image.random_flip_left_right(tf.image.random_crop(x, [32, 32, 3])), y)


def msg(*args):
    assert len(args)>0
    message = ""
    for arg in args:
        arg = str(arg)
        message =message + " " + arg
    return message
    

    
