# Image Normalization

- Dividing by 255 is Pixel normalization, its like changing the units but it does not change the distribution
- Image Normalization may not be needed if you are doing just divide by 255 and using Batch Norm. 
- In case of day vs night images you may find normalizing each image per channel is useful
```python
import numpy as np
for image in images:
    # shape of image = (255,255,3)
    # normalize across channels
    for i in range(image.shape[2]):
        image[:,:,i] = (image[:,:,i] - np.mean(image[:,:,i]))/np.std(image[:,:,i])

``` 

# Learnings
- Image Normalization increases acc
- Batch Normalization increases acc
- L2 Norm is not very good to increase acc, it helps in regularization only
- With BN and L2 applied Lr needs to be tuned again for good performance
- Higher values of LR can be used BN and L2

# TODOS
TODO: check image normalization code for per channel is correct

# References
### BatchNorm
- [Before end layer](https://github.com/alexgkendall/SegNet-Tutorial/issues/9)
- [Before FC layer](https://stats.stackexchange.com/questions/361700/lack-of-batch-normalization-before-last-fully-connected-layer)

### L2 Reg
- [L2 Reg vs BatchNorm](https://www.reddit.com/r/MLQuestions/comments/7hy8cz/why_does_batch_norm_plus_l2_regularization_make/)
- [L2 vs BN](https://medium.com/@SeoJaeDuk/archived-post-l2-regularization-versus-batch-and-weight-normalization-9d1d96e59391)
- [L2 vs BN Paper](https://arxiv.org/abs/1706.05350)

### Image Normalization
- [Different Types of Image Norms](https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current)
- [SO Ref](https://stackoverflow.com/questions/33610825/normalization-in-image-processing)
- [CS231n Data prep ref](http://cs231n.github.io/neural-networks-2/#datapre)
- [Fast Ai Forum on Image Norm](https://forums.fast.ai/t/images-normalization/4058/10)

### Image Preprocessing
- [ImageDataGenerator](https://keras.io/preprocessing/image/)
- [How to use predict_generator with ImageDataGenerator?](https://stackoverflow.com/questions/45806669/keras-how-to-use-predict-generator-with-imagedatagenerator)
- [Predictions from ImageDataGenerator](https://github.com/keras-team/keras/issues/2702)