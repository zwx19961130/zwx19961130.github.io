---
layout: post
title: "Image Phase Correlation"
date: 2022-06-06
description: "Image Processing"
tag: Phase
---

Phase correlation as described by http://en.wikipedia.org/wiki/Phase_correlation, taken from https://github.com/michaelting/Phase_Correlation/blob/master/phase_corr.py.

```python
def phase_correlation(a, b):
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.absolute(R)
    r = np.abs(np.fft.ifft2(R))
    return r
```

Here is an example: We take two similar images, but of different phases and plot the phase correlation (a black image with a single white dot at the appropriate phase difference).

```python
from scipy import misc
from matplotlib import pyplot
import numpy as np

#Get two images with snippet at different locations
im1 = np.mean(misc.face(), axis=-1) #naive colour flattening  

im2 = np.zeros_like(im1)    
im2[:200,:200] = im1[200:400, 500:700]

corrimg = phase_correlation(im1, im2)
r,c = np.unravel_index(corrimg.argmax(), corrimg.shape)

pyplot.imshow(im1)
pyplot.plot([c],[r],'ro')
pyplot.show()

pyplot.imshow(im2)
pyplot.show()

pyplot.figure(figsize=[8,8])
pyplot.imshow(corrimg, cmap='gray')

pyplot.show()
```


![png](../images/posts/2022-06-06/vSnbu.png)


Determine the location of the peak in r.

```python
r = np.fft.fftshift(r)

plt.title('Cross Correlation Map')
plt.imshow(r)
plt.grid()
plt.show()
```



![png](../images/posts/2022-06-06/2022-06-06-1.png)



We can see the cross-correlation peak at (47, 37); usually, the peak would not be so well defined.
```python
[py,px] = np.argwhere(r==r.max())[0]

cx,cy = 57,57
shift_x = cx - px
shift_y = cy - py

print(f'Shift measured X:{shift_x}, Y:{shift_y}')
```


> Shift measured X:10, Y:20

