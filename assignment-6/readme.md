# Results 
### **Assignment 6A**

- 84.4% using 56k Parameters (Round 8 - Experiment 3)
- One Cycle LR and cyclic LR are good techniques and helped get last 5%
    - One Cycle LR seemed easier to use and provided more gains
- DepthWise Convs along with normal convs produced best results. Make the layer with less kernels as normal conv. 
- Winning Architecture (Round 8 - Experiment 3)
    - Block 1
        - `Conv(32)->DepthConv(64)->DepthConv(128)->DepthConv(256, padded)`
        - `MaxPool(2)->Transition(Conv(32,1))`
    - Block 2
        - `Conv(32, padded)->DepthConv(64, padded)->DepthConv(128, padded)`
        - `MaxPool(2, padded)->Transition(Conv(32,1))`
    - Block 3
        - `Conv(32, padded)->DepthConv(64, padded)->DepthConv(128)`
        - `Transition(Conv(32,1))`
    - Output Block
        - `Conv(10,5)`
        - `Flatten()`
        - `Activation('softmax')`
        
- Training:
    - OneCycleLR
    - Random cutout regularization
    
### Assignment 6B
98.5k params, 50 epochs, One Cycle LR

```bash
Train
Score =  [0.578230784702301, 0.8286399998474121]
Balanced Accuracy = 82.89%, Accuracy = 82.89%

Test
Score =  [0.5842084721565246, 0.8293000004768372]
Balanced Accuracy = 80.66%, Accuracy = 80.66%
```


# References
- [One cycle LR](https://github.com/titu1994/keras-one-cycle)
- [Improvements in Deep Learning Optimisers after Adam](https://medium.com/vitalify-asia/whats-up-with-deep-learning-optimizers-since-adam-5c1d862b9db0)
- [Keract for visualizing Activations](https://github.com/philipperemy/keract)
- [Making Convolutions Faster](https://towardsdatascience.com/speeding-up-convolutional-neural-networks-240beac5e30f)
- [Padding Options](https://www.corvil.com/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow)
- [Padding Options Ref 2](https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t)
- [Separable Conv Intro](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)
- [Grouped Convolutions - Different from what we learnt](https://blog.yani.io/filter-group-tutorial/)
- [Functional API keras guide](https://keras.io/getting-started/functional-api-guide/) and [Cifar 10 resnet keras](https://keras.io/examples/cifar10_resnet/)
- [Pooling API Keras](https://keras.io/layers/pooling/) and [Global Average Pooling Explanation](https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer)
- Global Average Pooling vs Global Max Pooling [Ref](https://www.researchgate.net/post/Differences_between_Global_Max_Pooling_and_Global_Average_pooling)

        In cases where I want to extract the maximum representable features from my input image, Average pooling fails in that case and returns less accuracy. 
        This happens when I have a binary classification problem where images in both the classes look almost the same, but there are very minute differences between them, so averaging out the features in that case fails right? 
        [ For example, A Siberian Husky and Alaskan Malamute ]
- [Types on Convolutions](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)
- [Inter convertibility of FC and Conv Layers](http://cs231n.github.io/convolutional-networks/#fc)

