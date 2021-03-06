# Architectural Basics

### High Level Process

- Deciding Architecture
- Regularization
- Optimising Training Process 
    - Optimisers 
    - LR and LR scheduling 
    - epochs
    - ES
    - ReduceLR
- Augmentation

**Note:** List Down the learnings in each file at the end.

**Imp Note** Use Train, Validation, Test Split (i.e. 3 splits), Do not optimise model on Test Split. None of our decisions should be based on performance on test set. Since in real world you will not see the test set beforehand.


### Structuring our Approach
#### 1st File: Candidate Architectures**

    - Receptive field of full image
    - MaxPool vs strides=2
    - Stop at 4x4 or 5x5 or 6x6 or 7x7
    - Start with 16 vs 32 vs 64 (64 filters was a winner in assignment 3)
    - 10 Epochs
    
_Learnings_

    - Stopping at 5x5, 6x6 and 8x8 works well
    - Starting with 32 filters was good for Mnist
    - Strides=2 and MaxPool both work well, Mnist is not enough to conclude
    
#### 2nd File: Regularization**

    - Overfit here with more epochs (100 epochs)
    - Then try BN vs Dropout
    - Use both BN & Dropout
    
_Learning_

    - Dropout can be added just before prediction layer as well.
    - **Imp:** After adding dropout, number of dead kernels reduced. As such dropout is not only good for regularization, but also in adding variety in learning

Before Dropout
![Before Dropout](https://github.com/faizanahemad/eva/raw/master/assignment-4/1st_layer_32x32_before_dropout.png)

After Dropout (See more kernels active compared to above)
![After Dropout](https://github.com/faizanahemad/eva/raw/master/assignment-4/1st_layer_32x32_after_dropout.png)

#### 3rd File: Improvements (Optimiser and EarlyStopping)**

    - Adam vs Fine tuned SGD (SGD may be better since recent papers use this)
    - Use ES with validation data. Find right num epochs.
    - ReduceLRonPlateau
    - Play with BatchSize
    
_Learning_

    - NAdam Optimiser does not work well
    - Of All optimisers only Adam works well with less fiddling
    - SGD is the most difficult to tune. Also in my experiments SGD never gave perf same as Adam. SGD was Always below.
    - Batch size effects validation_acc, with right batch size it is possible to reach 99.4
    
 
#### 4th File: Improvements part 2 (play with LR)**
 
    - Use custom Loss Fn
        - We penalize filters which are highly correlated
    - Use SpatialDropout2D
    
_Learning_
    
    - Custom loss fn does not give provable gains on MNist with our own architectures.
    - SpatialDroopout2D does not perform any better than normal dropout.


### Small thoughts on over fitting and Mnist

- Over fitting your training data is good if you know that your test set looks like your training data
- In case of Mnist this is true.
- As a result Over fitting to training set improves performance of Test set in Mnist. In case of other data sets this assumption may fall flat.  

# Thinking and Ordering the Points of Consideration

### Order in My mind

- Preprocessing/Augmentation
- Architecture
- Early Validation
- Regularization
- Optimising Training Process

### **Arranging Points in Assignment in the Given thought order**

**Preprocessing**

1. Image Normalization
    - Preprocessing and augmentation steps should be done before modelling. 
    - Also our augmentation steps will determine the total number of examples and hence model complexity

**Architecure**

2. Receptive Field
    - Based on this layers and MaxPool location is decided.
3. Kernels and how do we decide the number of kernels?
    - We decide this based on how complex our images are. 
    - Also based on how different our different our classes are, we may need more kernels.
    - For example if you want to distinguish sedan and hatchback cars then you need more kernels, but if you want to distinguish between car and monkey then you need less kernels.
4. 3x3 Convolutions,
    - We use these in all layers, for perf reasons other sizes are not favored.
    - In last layer we may use bigger sizes, since a 3x3 on a 5x5 image sees middle pixel 9 times, and peripheral pixels only 1 time
5. How many layers,
    - Depends on receptive field
    - Required model complexity
6. 1x1 Convolutions,
    - Useful to combine multiple channels in a weighted average fashion
    - since they don't look at side pixels in same channel so they do a good job of looking across channels for combining
7. Concept of Transition Layers,
    - Decreasing size of channel
    - Increasing receptive field twice
    - MaxPool and conv with strides=2 can be used
8. Position of Transition Layer,
    - After 2 or 3 or 4 conv layers once we have made a receptive field of 9x9 or 11x11
    - It is data dependent
    - Maintain distance from last layer, since we want to pass all info to last layer 
    - [Striding vs Pooling](https://stackoverflow.com/questions/44666390/max-pool-layer-vs-convolution-with-stride-performance)
    - [Resnet vs Fishnet](https://stats.stackexchange.com/questions/387482/pooling-vs-stride-for-downsampling)
    - [Geoff Hinton on MaxPooling](https://mirror2image.wordpress.com/2014/11/11/geoffrey-hinton-on-max-pooling-reddit-ama/)
9. MaxPooling,
    - Useful to provide Translation Invariance (Doubtful in new studies)
    - Reduces dimension of channel and allows use to use less layers and params
10. Position of MaxPooling,
    - After 2/3/4 conv layers once we have formed a significant feature, or 11x11 receptive field.
11. The distance of MaxPooling from Prediction,
    - at least 2/3 layers away, else information loss happens
12. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
    - Once you just have few pixels like less than 7x7.
    - In 5x5 if you convolve 3x3 middle pixel is seen 9 times, for 7x7 middle pixel is seen 4 times. So This causes the mid pixel to become too important and hence issues.
    - The info in side pixels is drowned since middle pixel takes all attention
    - Use Padding
    - Use Strides=2 towards end may also help. 
13. SoftMax

**Early Validation**

14. How do we know our network is not going well, comparatively, very early
    - Use validation split and see if our validation acc is decreasing
15. When to add validation checks
    - If your final network does not do well, we need to see if our network was able to reduce validation acc during training.

**Regularization**

16. DropOut
    - Do not add after final layer since that layer has just n neurons as classes. So dropping means you will lose accuracy
    - Dropout may be added before final layer, test in your arch. 
    - Dropout values are good between 0.05-0.2 in my experiments
    - Sometimes its suggested to use 0.5 dropout with a deep network, reduce dropout slowly, this works if you have hardware capacity
    - For constrained hardware, 1st minimise your params, then overfit and use dropout.
17. When do we introduce DropOut, or when do we know we have some overfitting
    - Use dropout when gap between train and test acc is too much, like 0.5% or more.
18. Batch Normalization,
    - Normalizes each channel
    - If channel 1 and channel 2 are there, 32 examples per batch, then for these 32 examples each channels mean and variance is found
19. The distance of Batch Normalization from Prediction, 
    - Batch Norm should not be right before prediction layer, we want prediction layer to have all info unchanged
**Optimising Training**

20. Batch Size, and effects of batch size
    - Larger batch sizes are good usually
    - After some batch size increase the validation acc starts decreasing.
    - Lower batch sizes help avoid local minimas. [Ref](https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent) 
    - If you notice the image, full batch gradient descent follows a very linear path while SGD and mini batch follow more snake like path. Now lower your batch size more snake like your path. If full batch descent (blue line) faces a local minima in its path, it will get stuck there always. While SGD and mini batch may escape that local minima sometimes. So you will notice better performance with SGD/mini batch.
    - ![Gradient Descent Image](https://cdn-images-1.medium.com/max/1600/1*PV-fcUsNlD9EgTIc61h-Ig.png)

21. Number of Epochs and when to increase them
    - How many times we pass over our entire dataset
    - Increase when you have underfitting
    - Increase when your validation loss is still decreasing
    - Use EarlyStopping and ReduceLRonPlateau to find right epochs 
22. Adam vs SGD 
    - Adam is easier to tune, out of box provides good results
    - [UC Berkeley Study on Adam vs SGD](https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/)
    - All high performing architectures use SGD these days
    - [Why SGD better Quora](https://www.quora.com/Why-do-the-state-of-the-art-deep-learning-models-like-ResNet-and-DenseNet-use-SGD-with-momentum-over-Adam-for-training)
23. Learning Rate,
    - The rate at which weights are changes once gradients are found
    - Lower provides better results but needs more epochs
    - Using BatchNorm allows higher learning rates.
24. LR schedule and concept behind it
    - Reducing LR as it reaches the minima can help in reducing oscillations near the minima.
    - Try the decay param in Adam before trying LR scheduling.
25. ReduceLRonPlateau & EarlyStopping
    - Helps in finding good minimas
    - ReduceLRonPlateau reduces Lr when learning is stuck
    - Early stopping stops training once we are not improving any validation acc.
26. Gradient capping/clipping  


# References
- Optimisers
    - [Keras Optimisers](https://keras.io/optimizers/)
    - [AdaBound](https://medium.com/syncedreview/iclr-2019-fast-as-adam-good-as-sgd-new-optimizer-has-both-78e37e8f9a34)
    - [Adam vs SGD](https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/)
- [Layer vs Batch Norm](https://datascience.stackexchange.com/questions/12956/paper-whats-the-difference-between-layer-normalization-recurrent-batch-normal)
    - LayerNorm is not good for ConvNets
    - [LayerNorm Explanation](http://mlexplained.com/2018/01/13/weight-normalization-and-layer-normalization-explained-normalization-in-deep-learning-part-2/)
    - [Implementation in Keras](https://gist.github.com/udibr/7f46e790c9e342d75dcbd9b1deb9d940) 
- [How Does Batch Normalization Help Optimization? (No, It Is Not About Internal Covariate Shift)](https://shaoanlu.wordpress.com/2018/07/12/notes-for-paper-how-does-batch-normalization-help-optimization-no-it-is-not-about-internal-covariate-shift/)
    - BN allows wider range of LR parameters
    - BN reparametrizes the underlying optimization problem to make it more stable and smooth
- [Checkpointing Tutorial](https://machinelearningmastery.com/check-point-deep-learning-models-keras/)
- [Keras Custom Losses](https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618)
- [Batch Size and Model Generalisation](https://arxiv.org/abs/1609.04836)
    - Models with bigger batch sizes don't generalize well.
- [How Does dropout make robust models](https://datascience.stackexchange.com/questions/37021/why-does-adding-a-dropout-layer-in-keras-improve-machine-learning-performance-g/37024#37024)
        
