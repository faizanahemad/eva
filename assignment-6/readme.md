# Results 
- **Assignment 6A:** 84.4% using 56k Parameters (Round 8 - Experiment 3)
- One Cycle LR and cyclic LR are good techniques and helped get last 5%
    - One Cycle LR seemed easier to use and provided more gains
- DepthWise Convs along with normal convs produced best results. Make the layer with less kernels as normal conv. 
- Winning Architecture
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


# References
- [One cycle LR](https://github.com/titu1994/keras-one-cycle)
