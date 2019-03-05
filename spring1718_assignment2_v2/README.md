# Assignment 2

## Requirement

Original link about assignment 2:
- [http://cs231n.github.io/assignments2018/assignment2/](http://cs231n.github.io/assignments2018/assignment2/)

## Notes for important places

### Batch normalization

An excellent online material with detailed derivation of numerical backward method of batch normalization: 
- [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

Analytical backward method of batch normalization: 
- [What does the gradient flowing through batch normalization looks like](http://cthorey.github.io./backpropagation/)

### Layer normalization

Forward pass of layer normalization is very similar to the batch normalization, except that the mean and variance are computed per row (per datapoint) of the input *N* by *D* matrix.

In the backward pass of layer normalization, gradient of *gamma* and *beta* are exactly the same as that in batch normalization, while gradient of *x* is different as:

<a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;dx_{ij}&space;=&space;\frac{1}{D}(\sigma_i^2&space;&plus;&space;\epsilon)^{-\frac{1}{2}}&space;[D&space;\cdot&space;dy_{ij}&space;\cdot&space;\gamma_j&space;-&space;\sum_l&space;dy_{il}&space;\cdot&space;\gamma_l&space;-&space;(h_{ij}&space;-&space;\mu_i)(\sigma_i^2&space;&plus;&space;\epsilon)^{-1}&space;\sum_l&space;dy_{il}&space;\cdot&space;\gamma_l&space;(h_{il}&space;-&space;\mu_i)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\LARGE&space;dx_{ij}&space;=&space;\frac{1}{D}(\sigma_i^2&space;&plus;&space;\epsilon)^{-\frac{1}{2}}&space;[D&space;\cdot&space;dy_{ij}&space;\cdot&space;\gamma_j&space;-&space;\sum_l&space;dy_{il}&space;\cdot&space;\gamma_l&space;-&space;(h_{ij}&space;-&space;\mu_i)(\sigma_i^2&space;&plus;&space;\epsilon)^{-1}&space;\sum_l&space;dy_{il}&space;\cdot&space;\gamma_l&space;(h_{il}&space;-&space;\mu_i)]" title="\LARGE dx_{ij} = \frac{1}{D}(\sigma_i^2 + \epsilon)^{-\frac{1}{2}} [D \cdot dy_{ij} \cdot \gamma_j - \sum_l dy_{il} \cdot \gamma_l - (h_{ij} - \mu_i)(\sigma_i^2 + \epsilon)^{-1} \sum_l dy_{il} \cdot \gamma_l (h_{il} - \mu_i)]" /></a>

Here *dy* is the upstream derivatives (i.e., the input *dout* in backward normalization). Note that the mean and variance are computed per row here. It's not hard to get this final formula based on the derivation of batch normalization from the online material shown above.