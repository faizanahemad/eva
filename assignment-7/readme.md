# 7A
For Finding overall RF, We first find receptive field calculation of Inception module.
Notice That inception module retains input dimension. 

As a result we can calculate the Paddings and strides, and hence calculate RF for the inception module.

![Inception](Inception.png)

So we find rules for inception module.
```python
Rout = Rin, Rin + 2*Jin, Rin + 4*Jin
Jout = Jin
```

Now lets try for inception, 1st few layers.
![](Inception-Table.png)

Final Receptive Field is calculated by our program ([rf_finder.py](rf_finder.py) and [RF-Finder.ipynb](RF-Finder.ipynb)).

**List of final Receptive field**
```bash
[267,283,299,315,331,347,363,379,395,411,427,443,459,475,491,
 507,523,539,555,571,587,603,619,635,651,667,683,699,715,731,
 747,763,779,795,811,827,843,859,875,891,907]
```

Notice that this network has multiple receptive fields at the end.

**Note:** My code assumes that `Jump` value is only one after each layer, while multiple RF is allowed.

# 7B
### Approach
Instead of writing `Lambda` and `concatenate` everytime, we made custom functions which help us.
Example:
```python
def concat_s2d(inputs):
  final_inputs = None
  if type(inputs)==list:
    modified_inputs = []
    mh,mw = 1e8,1e8 # min height and width
    
    for inpt in inputs:
      s = K.int_shape(inpt)
      h,w = s[-3],s[-2]
      mh = min(mh,h)
      mw = min(mw,w)
      modified_inputs.append((inpt,h,w))
    final_inputs = []
    for inpt,h,w in modified_inputs:
      assert h%mh==0 and w%mw==0 and h/mh == w/mw
      if int(h/mh)>1:
        inp = Lambda(lambda x: tf.space_to_depth(x, block_size=int(h/mh)))(inpt)
      else:
        inp = inpt
      final_inputs.append(inp) 
  inputs = concatenate(final_inputs) if type(inputs)==list and len(inputs)>1 else inputs
  inputs = inputs[0] if type(inputs)==list and len(inputs)==1 else inputs
  return inputs

def conv_layer(inputs, n_kernels=32, kernel_size=(3,3), dropout=0.15,dilation_rate=1, padding='same', 
               skip_1=True,skip_2=True, enable_transition = True,transition_layer_kernels = 32):
  inputs = concat_s2d(inputs)
  inputs = transition_layer(inputs, transition_layer_kernels) if enable_transition else inputs
  out = Conv2D(n_kernels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                kernel_regularizer=l2(2e-4),
                dilation_rate=dilation_rate)(inputs)
  out = BatchNormalization()(out)
  out = Activation("relu")(out)
  out = Dropout(dropout)(out) if dropout>0 else out
  return out
```
In `concat_s2d` function, it 
- takes an array of multiple outputs, 
- concats them by doing appropriate `space_to_depth` conversions

In `conv_layer` 
- includes capability to do bottleneck (1x1), Dropout, BN.
- uses `concat_s2d` to handle array of incoming inputs


### Results
```bash
Train
Score =  [0.2631824646568298, 0.9369200000190735]
Balanced Accuracy = 93.84%, Accuracy = 93.84%

Validation 
Score =  [0.5835708273649216, 0.8553]
Balanced Accuracy = 84.09%, Accuracy = 84.09%
```