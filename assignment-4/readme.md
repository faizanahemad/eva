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
- **1st File: Candidate Architectures**
    - Receptive field of full image
    - MaxPool vs 1x1 with stride=2
    - Stop at 4x4 or 5x5 or 6x6 or 7x7
    - Start with 16 vs 32 vs 64 (64 filters was a winner in assignment 3)
    - 10 Epochs
    - Total candidate architectures in this file = `2x4x4 = 32`
    - We will promote 8 architectures from here
    
    
- **2nd File: Regularization**
    - Overfit here with more epochs (100 epochs)
    - Then try BN vs Dropout
    - Use both BN & Dropout
    - Total Candidates from this file = `8x3 = 24` (8 from prev, 3 BN/Dropout/BN+Dropout)
    - Again we will promote only 8 to next file based on performance and diversity
    

- **3rd File: Improvements (Optimiser and EarlyStopping)**
    - Adam vs Fine tuned SGD (SGD may be better since recent papers use this)
    - Use ES with validation data. Find right num epochs. 
    - Retrain model with all data with right num epochs
    - Total Candidates from this file: `8x2 = 16`
    - We will take 4 from Adam and 4 from SGD = 8.
 
 - **4th File: Improvements part 2 (play with LR)**
    - Initial LR and LR scheduling vs Use LR decay in Adam and SGD
    - Use ReduceLRonPlateau


#### Small thoughts on over fitting and Mnist

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

**Architecure**

2. Receptive Field
3. Kernels and how do we decide the number of kernels?
4. 3x3 Convolutions,
5. How many layers,
6. 1x1 Convolutions,
7. Concept of Transition Layers,
8. Position of Transition Layer,
9. MaxPooling,
10. Position of MaxPooling,
11. The distance of MaxPooling from Prediction,
12. When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
13. SoftMax

**Early Validation**

14. How do we know our network is not going well, comparatively, very early
15. When to add validation checks

**Regularization**

16. DropOut
17. When do we introduce DropOut, or when do we know we have some overfitting
18. Batch Normalization,
19. The distance of Batch Normalization from Prediction, (Batch Normalization should be added before the predication layer)

**Optimising Training**

20. Batch Size, and effects of batch size
21. Number of Epochs and when to increase them
22. Adam vs SGD 
23. Learning Rate,
24. LR schedule and concept behind it
25. ReduceLRonPlateau
26. EarlyStopping


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
        
