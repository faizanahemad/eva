def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)

def tf_cutout(x: tf.Tensor, area: int = 81, c: int = 3) -> tf.Tensor:
    """
    Cutout data augmentation. Randomly cuts a h by w whole in the image, and fill the whole with zeros.
    :param x: Input image.
    :param h: Height of the hole.
    :param w: Width of the hole
    :param c: Number of color channels in the image. Default: 3 (RGB).
    :return: Transformed image.
    """
    minimum = tf.cast(tf.reduce_min(x),dtype=tf.int32)
    maximum = tf.cast(tf.reduce_max(x),dtype=tf.int32)
    h = np.random.randint(3,np.sqrt(area)+1)
    w = int(np.ceil(area/h))
    shape = tf.shape(x)
    x0 = tf.random.uniform([], 0, shape[0] + 1 - h, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, shape[1] + 1 - w, dtype=tf.int32)
    
    slic = tf.cast(tf.random.uniform([h, w, c],minval=minimum,maxval=maximum, dtype=tf.int32),dtype=tf.uint8)
    x = replace_slice(x,slic, [x0, y0, 0])
    return x

def get_tf_cutout_eraser(area=81):
    def cutout_mapper(x,y):
        return tf_cutout(x,area,3),y
    return cutout_mapper



def batch_cut(imgs,cutout_proba=1.0, copy=False):
    cutout_fn = get_cutout_eraser(p=1.0, pixel_level=True)
    if copy:
        imgs = np.copy(imgs)
    for i,im in enumerate(imgs):
        imgs[i] = cutout_fn(im, proba=cutout_proba)
    return imgs

def cifar10_augs(imgs,cifar10_proba=1.0,policy_idx=None,magnitude_idx=None):
    policy = CIFAR10Policy(fillcolor=tuple(train_mean.astype(int)),log=True)
    new_images = []
    for i,im in enumerate(imgs):
        p_1 = np.random.rand()
        if p_1 < cifar10_proba:
            im = policy(im,policy_idx=policy_idx,magnitude_idx=magnitude_idx)
        new_images.append(im)
    new_images = np.stack(new_images)
    return new_images