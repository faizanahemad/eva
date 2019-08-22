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

#custom initializers to force float32
class Ones32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(1, shape=shape, dtype='float32')

class Zeros32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(0, shape=shape, dtype='float32')
    


class BatchNormalizationF16(Layer):

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(BatchNormalizationF16, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (
            initializers.get(moving_variance_initializer))
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return tf.nn.batch_normalization(#K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    #axis=self.axis,
                    self.epsilon)#epsilon=self.epsilon)
            else:
                return tf.nn.batch_normalization(#K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    #axis=self.axis,
                    self.epsilon)#epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = _regular_normalize_batch_in_training(#K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalizationF16, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
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
    def __init__(self, c_out, kernel_size=3, bn=False, strides=1):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=kernel_size,strides=strides, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
        self.bn = bn
        if bn:
            self.bn = BatchNorm(momentum=0.9, epsilon=1e-5)

    def call(self, inputs):
        if self.bn:
            res = tf.nn.relu(self.bn(self.conv(inputs)))
        else:
            res = tf.nn.relu(self.conv(inputs))
        return res
    
class ThinConvBN(tf.keras.Model):
    def __init__(self, c_out, kernel_size=3, bn=False, strides=1):
        super().__init__()
        self.conv = tf.keras.layers.SeparableConv2D(filters=c_out,depth_multiplier=1, kernel_size=kernel_size,strides=strides, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
        self.bn = bn
        if bn:
            self.bn = BatchNorm(momentum=0.9, epsilon=1e-5)

    def call(self, inputs):
        if self.bn:
            res = tf.nn.relu(self.bn(self.conv(inputs)))
        else:
            res = tf.nn.relu(self.conv(inputs))
        return res
    

class ResBlk(tf.keras.Model):
    def __init__(self, c_out, pool, res = False, bn=False):
        super().__init__()
        self.conv_bn = ConvBN(c_out, bn=bn)
        self.pool = pool
        self.res = res
        if self.res:
            self.res1 = ConvBN(c_out, bn=bn)
            self.res2 = ConvBN(c_out, bn=bn)

    def call(self, inputs):
        h = self.pool(self.conv_bn(inputs))
        if self.res:
            h = h + self.res2(self.res1(h))
        return h
    
    
class ResBlkThin(tf.keras.Model):
    def __init__(self, c_out, pool, res = False, bn=False ):
        super().__init__()
        self.conv_bn = ThinConvBN(c_out, bn=bn)
        self.pool = pool
        self.res = res
        if self.res:
            self.res1 = ThinConvBN(c_out, bn=bn)
            self.res2 = ConvBN(c_out, bn=bn)

    def call(self, inputs):
        h = self.pool(self.conv_bn(inputs))
        if self.res:
            h = h + self.res2(self.res1(h))
        return h
    

    

# Batch Size
# Maxpool after 1st conv

class FNet(tf.keras.Model):
    def __init__(self, start_kernels=64, weight=0.125, sparse_bn = True, thin_block = False, 
                 enable_skip=False, enable_pool_before_skip=False):
        super().__init__()
        c = start_kernels
        pool = tf.keras.layers.MaxPooling2D()
        bn = not sparse_bn
        self.init_conv_bn = ConvBN(c, kernel_size=3, bn=bn)
        self.enable_skip = enable_skip
        self.enable_pool_before_skip = enable_pool_before_skip
        if enable_skip:
            self.skip = ConvBN(c*2,kernel_size=1, strides=1, bn=bn)
            self.max_pool = tf.keras.layers.MaxPooling2D()
        
        
        self.blk1 = ResBlkThin(c*2, pool, res = True, bn=bn) if thin_block else ResBlk(c*2, pool, res = True, bn=bn)
        self.blk2 = ResBlk(c*4, pool, bn=bn)
        self.blk3 = ResBlkThin(c*8, pool, res = True, bn=bn) if thin_block else ResBlk(c*8, pool, res = True, bn=bn)
        self.pool = tf.keras.layers.GlobalMaxPool2D()
        self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
        self.weight = weight
        self.sparse_bn = sparse_bn
        if sparse_bn:
            self.bn1 = BatchNorm(momentum=0.9, epsilon=1e-4)
            self.bn2 = BatchNorm(momentum=0.9, epsilon=1e-4)
            self.bn3 = BatchNorm(momentum=0.9, epsilon=1e-4)
            self.bn4 = BatchNorm(momentum=0.9, epsilon=1e-4)

    def call(self, x, y):
        h = self.init_conv_bn(x)
        h = self.bn1(h) if self.sparse_bn else h
        
        h = self.blk1(h)
        h = self.bn2(h) if self.sparse_bn else h
        
        h = self.blk2(h)
        h = self.bn3(h) if self.sparse_bn else h
        
        if self.enable_skip:  
            if self.enable_pool_before_skip:
                k = self.pool(self.skip(self.max_pool(h)))
            else:
                k = self.pool(self.skip(h))
        h = self.blk3(h)
        h = self.pool(h)
        if self.enable_skip:
            h = tf.keras.layers.concatenate([h,k])
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
    

    
