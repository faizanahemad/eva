# Image Normalization
- Dividing by 255 is Pixel normalization, its like changing the units but it does not change the distribution
- See types of Image Normalization
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

# TODOS
TODO: check image normalization code for per channel is correct

# References
## Image Normalization
- [Different Types of Image Norms](https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current)
- [SO Ref](https://stackoverflow.com/questions/33610825/normalization-in-image-processing)
- [CS231n Data prep ref](http://cs231n.github.io/neural-networks-2/#datapre)
- [Fast Ai Forum on Image Norm](https://forums.fast.ai/t/images-normalization/4058/10)
## Batch Normalization