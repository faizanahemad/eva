# Assignment 1: Filters/Kernels and Basic CNN

## [Assignment Notebook Link](https://colab.research.google.com/drive/1vonfE5AXxV26NHC_venQKnAEDQ8SUw6i)

- https://colab.research.google.com/drive/1vonfE5AXxV26NHC_venQKnAEDQ8SUw6i

- [EVA_S1.ipynb](https://github.com/faizanahemad/eva/blob/master/assignment-1/EVA_S1.ipynb) file in this folder

## Questions and Answers 

### What are Channels and Kernels (according to EVA)?

Kernels (also filters) are matrices that we multiply element wise to each channel of an image to create new channel. Each kernel is used to detect 1 type of feature (on low level - edges), (on high level - nose/eye). Kernels we use are 3x3 matrices and they are dot multiplied with a sliding window on the image.

Channels are basically used to represent images (Normal images have RGB / 3 channels). Each channel represents info of 1 type. As such after our filter is applied and a new channel is created, the new channel represents the type of information our filter extracted. For example if we used a horizontal edge detector filter, then the channel created will contain horizontal edges.

For each Convolutional layer, number of channels = number of filters, each filter creates a channel.

### Why should we only (well mostly) use 3x3 Kernels?

- Computational Reason: 2 3x3 kernels will reduce a 5X5 image to 1x1, so will a 5x5 kernel, the number of parameters for a 5x5 kernel = 25, but for 2 3x3 = 2x9=18, so fewer parameters, faster training.

- We need to use odd kernels (symetric kernels) because they have a line of symetry, which helps us do vertical, horizontal edges etc.

- Nvidia started developed hardware acceleration for 3x3 kernels, so they are way faster now.

- From a personal perspective, I hate hyper-parameter tuning. As such fixing kernel size reduces one parameter to tune. New/hard problems require better architectures and solutions, not excessive hyper-parameter tuning. So more parameters that are fixed, more I am happy.

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

Logic: each 3x3 reduces the previous channel to `channel - 2 ` so 199->3x3->197. based on that lets write it out.

```
199->(3x3)->197->(3x3)->195->(3x3)->193->(3x3)->191->(3x3)->189->(3x3)->187->(3x3)->185->(3x3)->183->(3x3)->181->(3x3)->179->(3x3)->177->(3x3)->175->(3x3)->173->(3x3)->171->(3x3)->169->(3x3)->167->(3x3)->165->(3x3)->163->(3x3)->161->(3x3)->159->(3x3)->157->(3x3)->155->(3x3)->153->(3x3)->151->(3x3)->149->(3x3)->147->(3x3)->145->(3x3)->143->(3x3)->141->(3x3)->139->(3x3)->137->(3x3)->135->(3x3)->133->(3x3)->131->(3x3)->129->(3x3)->127->(3x3)->125->(3x3)->123->(3x3)->121->(3x3)->119->(3x3)->117->(3x3)->115->(3x3)->113->(3x3)->111->(3x3)->109->(3x3)->107->(3x3)->105->(3x3)->103->(3x3)->101->(3x3)->99->(3x3)->97->(3x3)->95->(3x3)->93->(3x3)->91->(3x3)->89->(3x3)->87->(3x3)->85->(3x3)->83->(3x3)->81->(3x3)->79->(3x3)->77->(3x3)->75->(3x3)->73->(3x3)->71->(3x3)->69->(3x3)->67->(3x3)->65->(3x3)->63->(3x3)->61->(3x3)->59->(3x3)->57->(3x3)->55->(3x3)->53->(3x3)->51->(3x3)->49->(3x3)->47->(3x3)->45->(3x3)->43->(3x3)->41->(3x3)->39->(3x3)->37->(3x3)->35->(3x3)->33->(3x3)->31->(3x3)->29->(3x3)->27->(3x3)->25->(3x3)->23->(3x3)->21->(3x3)->19->(3x3)->17->(3x3)->15->(3x3)->13->(3x3)->11->(3x3)->9->(3x3)->7->(3x3)->5->(3x3)->3->(3x3)->1

Total iterations =  100
```

Code to generate this:

```python
ctr = 0
for i in range(199,0,-2):
  ctr = ctr + 1
  if i>1:
    print(str(i)+'->(3x3)->', end='')
  else:
    print(str(i), end='')  
print("\nTotal iterations = ",ctr)
```
