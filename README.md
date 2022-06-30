# Neural Upscale 2x

Upscale images by a factor of 2 via deep convolutional neural network on Python: API and pre-trained models, based on the [ESPC](https://arxiv.org/pdf/1609.05158.pdf) architecture by Shi et al. 2016 [1].

Two pre-trained models are included: Luminance, RGB. You can, however, use any `Tensorflow` model which is compatible with the architecture, or even train your own, in which case `Utilities` package may be useful.

Requirements: `Tensorflow`, `PIL`.


***
### Quick Guide

1. Import necessary utilities, all dependencies are loaded automatically:

```python
from UpscalingUtilities import *
from Neural_Upscale_2x import *
```

2. Load desired model. Pre-trained models are either `Luminance_Model` or `RGB_Model`, contained in the respective directories. Say you want to use the first:

```python
model = load_upscaling_model('../Luminance_Model')
```

3. Initialize an upscaler object using the appropriate function. Either `luminanceUpscaler()` or `rgbUpscaler()`. In this example:

```python
upscaler = luminanceUpscaler(model)
```

4. Load desired image using `Tensorflow`'s function, which returns a `PIL` image instance:

```python
img = load_img(../img.png)
```

5. Upscale! Returns a `PIL` image instance, which you can save as usual:

```python
2x_img = upscaler.upscale(img)
2x_img.save('../title.png')
```

***
![LargeMatrix](https://user-images.githubusercontent.com/44241033/176714502-6836391c-3b6f-46ef-905f-afc3008da0fc.PNG)

***
### References

<a id="1">[1]</a> 
Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P Aitken, Rob Bishop, Daniel Rueckert, and Zehan Wang. Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pages 1874–1883, 2016.


