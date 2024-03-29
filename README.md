# Neural Upscale 2x

![headline](https://github.com/TheMrKingent/NeuralUpscale/assets/44241033/faed46c7-0767-4c39-af96-4e39bd59b1ea)

Upscale images by a factor of 2 via deep convolutional neural network in Python: API and pre-trained models, based on the [ESPC](https://arxiv.org/pdf/1609.05158.pdf) architecture by Shi et al. 2016 [1].

Two pre-trained models are included: Luminance and RGB, both of which were trained on approximately 33k various images (~6.5Gb). You can, however, use any Tensorflow model which is compatible with the architecture, or even train your own, in which case `UpscalingUtilities` package and `Training.ipynb` notebook may be useful.

Requirements: `Tensorflow`, `PIL`. Tested on Python 3.10.


***
### Quick Guide

1. Import necessary utilities, all dependencies are loaded automatically:

```python
from UpscalingUtilities import *
from Neural_Upscale_2x import *
```

2. Load desired model. Pre-trained models are either `Luminance_Model` or `RGB_Model`. Say you want to use the first one:

```python
model = load_upscaling_model('../Luminance_Model')
```

3. Load desired image using `Tensorflow`'s function, which returns a `PIL` image instance:

```python
img = load_img('../img.png')
```

4. Initialize an upscaler object using the appropriate function. Either `luminanceUpscaler()` or `rgbUpscaler()`. In this example:

```python
upscaler = luminanceUpscaler(model)
```

5. Upscale! Returns an image upscaled by a factor of 2 in both width and height (4 times as many pixels) as a `PIL` image instance, which you can save as usual:

```python
img_2x = upscaler.upscale(img)
img_2x.save('../title.png')
```

***
![LargeMatrix](https://user-images.githubusercontent.com/44241033/195892182-c8f0f600-652b-45cb-8160-ab25f7d9714b.PNG)

***
### References

<a id="1">[1]</a> 
Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 1874–1883, 2016.

