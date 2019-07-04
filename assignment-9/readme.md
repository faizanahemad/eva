
## Ideas to Explore
- Cutout for internal layers as a dropout strategy
- Patch Gaussian cutout
- Cutout, Hue, Jitter, Mixup, Label Smoothing, RICAP,  random shifts to the pictures in RGB
colorspace
- RICAP enhancement
    - First we train a network with cutout
    - Next we use gradcam on each training image to detect its important parts
    - Now we combine multiple training images like in RICAP
    - But here we use the labels not by just image part size, rather part size * importance of area given by gradcam.
    - This will do better label smoothing.
    - Use multiple parts of same image, like put the same image in corners and different image in center.
    
- **Pachout**: Instead of cutout use proper occlusions by taking segments of other images.
    - Segments are again choosen based on the grad cam process above
    - Here we just keep labels same as main image but reduce the label value by Label smoothing
    

- Gaussian Patch noise circle with centre as most noisy and ripple.
- Use all above techniques simultaneously with some proba for each and mutually exclusive for Cutout, guassian and Ricap
- Prediction smoothing vs Label smoothing 

### Idea Dustbin
- Patch Gaussian and cutout of varying shapes like circle, ellipse, trapezium, parallelogram, square, rectangle  
- RICAP Labels with non rectangular sections


## Key Terms
- ablation study


## References
- [Image Augmentation Strategies - Fast.ai](https://hackernoon.com/introduction-to-image-augmentations-using-the-fastai-library-692dfaa2da42)
- [When Conventional Wisdom Fails: Revisiting Data Augmentation](https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509)
- [Channel Shuffle Code 1](https://github.com/scheckmedia/keras-shufflenet/blob/master/shufflenet.py) and [Channel Shuffle Code 2](https://github.com/minhto2802/keras-shufflenet/blob/master/shufflenet.py)
- [Bag of Tricks for Image Classification](https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/)
