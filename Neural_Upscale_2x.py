
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array

import PIL
import numpy as np



class rgbUpscaler:
    
    def __init__(self, model):
        
        self.model = model
        self.modelType = 'rgb'


    def upscale(self, img):
        '''
        Upscale image using RGB neural net model.
            
            Args:
                img: RGB input image.
            
            Return: RGB upscaled image.               
        '''
    
        # Rgb to model compatible input
        array_y = img_to_array(img) / 255
        model_input = np.expand_dims(array_y, axis=0)
        
        # Run model
        out_img = self.model.predict(model_input)
        
        # Model output to rbg
        out_img = out_img * 255
        out_img = out_img[0]
        out_img = out_img.clip(0, 255)
        out_img = array_to_img(out_img)
        
        return out_img
            


class luminanceUpscaler:
    
    def __init__(self, model):
        
        self.model = model
        self.modelType = 'luminance'
        
        
    def upscale(self, img):
        '''
        Upscale image using luminance neural net model.
            
            Args:
                img: RGB input image.
            
            Return: RGB upscaled image.               
        '''
    
        # Rgb to Luminance to model compatible input
        YCbCr = img.convert('YCbCr')
        y, cb, cr = YCbCr.split()
        array_y = img_to_array(y) / 255
        model_input = np.expand_dims(array_y, axis=0)

        # Run model
        out_img_y = self.model.predict(model_input)
        
        # Model output to luminance to rbg
        out_img_y = out_img_y[0] * 255
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
        out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
        out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert("RGB")
        
        return out_img