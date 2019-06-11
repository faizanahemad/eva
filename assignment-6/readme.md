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

