# UpscalingUtilities
# Author: Mattia Lamberti

'''
Various utilities for Neural Upscale 2X.
Also useful to train and test models.
'''



import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import os
import numpy as np



# -----------------------------------------------------------------------------
# Preprocess utilities

def scale(input_image):
    '''Scales RGB values from 0,255 to 0,1.'''
   
    return (input_image) / 255.0


def rgb_to_luminance(input_image):
    '''Converts to yuv and extract luminance channel y.'''
   
    yuv = tf.image.rgb_to_yuv(input_image)
    last_dimension_axis = len(yuv.shape) - 1
    y, u, v = tf.split(yuv, 3, axis=last_dimension_axis)
    return y


def resize_to_half(input_image, training_gt_size):
    '''Resizes image to half the size.'''
    
    return tf.image.resize(input_image, [training_gt_size//2, training_gt_size//2], method="area")


def associate_gt(input_image, training_gt_size):
    '''Associates halved samples to ground truth. Return tuple: (halved_train_sample, ground_truth).'''
    
    return (resize_to_half(input_image, training_gt_size), input_image)



# -----------------------------------------------------------------------------
# Preprocess routines

def preprocess_dataset_rgb_model(dataset, training_gt_size):
    '''Full preprocess routine for RGB model. Output is a model compatible dataset.'''
    
    scaled_dataset = dataset.map(scale)
    return scaled_dataset.map(lambda x: associate_gt(x, training_gt_size))


def preprocess_dataset_luminance_model(dataset, training_gt_size):
    '''Full preprocess routine for Luminance model. Output is a model compatible dataset.'''
    
    scaled_dataset = dataset.map(scale)
    scaled_lum_dataset =scaled_dataset.map(rgb_to_luminance)
    return scaled_lum_dataset.map(lambda x: associate_gt(x, training_gt_size))



# -----------------------------------------------------------------------------
# Inverse PSNR loss function

def tensor_log10(x):
    '''Computes log10 of tensor's elements.'''
    
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def inverse_PSNR(lowres_img, gt_img):
    '''Computes inverse PSNR (1/PSNR) between two images.'''
    
    squared_diff = tf.square(gt_img - lowres_img)
    mse = tf.reduce_mean(squared_diff)
    psnr = 10 * tensor_log10(1 / mse)
    return 1 / psnr



# -----------------------------------------------------------------------------
# Model utilities

def load_upscaling_model(path):
    '''Quick model load and setup from path.'''
    
    model = keras.models.load_model(path, compile = False)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=inverse_PSNR)
    return model



# -----------------------------------------------------------------------------
# Testing utilities

def plot_results(img, zoom, x1, x2, y1, y2, suplot_loc, loc1, loc2, title=None, save=False, dpi=100):
    '''Plots the result with zoom-in area.'''
    
    plt.rcParams["figure.dpi"] = dpi
    
    img_array = img_to_array(img)
    img_array = img_array.astype("uint8") / 255

    # Create a new figure with a default subplot.
    fig, ax = plt.subplots()
    im = ax.imshow(img_array[::-1], origin="lower")

    # Zoom-factor: zoom, location: suplot_loc
    axins = zoomed_inset_axes(ax, zoom, loc=suplot_loc)
    axins.imshow(img_array[::-1], origin="lower")

    # Apply the x-limits.
    axins.set_xlim(x1, x2)
    # Apply the y-limits.
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)
    ax.set_axis_off()
    fig.add_axes(ax)

    # Make the line.
    mark_inset(ax, axins, loc1=loc1, loc2=loc2, fc="none", ec="blue")    
        
    if save == True:
        plt.savefig(title + ".png", dpi='figure', bbox_inches='tight')
        
    plt.show()


def downscale(img):
    '''Downscales image using bicubic interpolation.'''
    
    shrinked = tf.image.resize(img, (img.size[1] // 2, img.size[0] // 2), method='bicubic')
    return array_to_img(shrinked)


def bicubic_upscale(img):
    '''Upscales image using bicubic interpolation.'''
    
    enlarged = tf.image.resize(img, (img.size[1] * 2, img.size[0] * 2), method='bicubic')
    return array_to_img(enlarged)


def compare(img, upscaler, title=None, zoom=2, x1=200, x2=300, y1=100, y2=200, suplot_loc=2, loc1=1, loc2=3, save=False, dpi=100):
    '''Plots Groud truth, downscaled + Bicubic upscale, downscaled + Neural net upscale. Also returns PSNR.
    
    upscaler arg can eventually be a list of upscaler objects, in which case one image per upscaler will be plotted.
    Useful for performance evaluation.'''

    img = tf.image.resize(img, (img.size[1] //2 * 2, img.size[0] //2 * 2))
    img = array_to_img(img)

    psnr = []    
    shrinked = downscale(img)
    
    # Plot ground truth
    plot_results(img, title='Ground truth',
                 zoom=zoom, x1= x1, x2=x2, y1=y1, y2=y2, suplot_loc=suplot_loc, loc1=loc1, loc2=loc2, save=save, dpi=dpi)
    
    # Plot bicubic
    bic_img = bicubic_upscale(shrinked)
    plot_results(bic_img, title='Bicubic',
                 zoom=zoom, x1= x1, x2=x2, y1=y1, y2=y2, suplot_loc=suplot_loc, loc1=loc1, loc2=loc2, save=save, dpi=dpi)
    psnr.append(tf.image.psnr(img_to_array(img), img_to_array(bic_img), max_val=255).numpy())

    # Plot NN
    if type(upscaler) != list:
        upscaler = [upscaler]        
    for i in range(len(upscaler)):
        nn_img = upscaler[i].upscale(shrinked)
        psnr.append(tf.image.psnr(img_to_array(img), img_to_array(nn_img), max_val=255).numpy())
        plot_results(nn_img, title=title[i],
                        zoom=zoom, x1= x1, x2=x2, y1=y1, y2=y2, suplot_loc=suplot_loc, loc1=loc1, loc2=loc2, save=save, dpi=dpi)
        
    return psnr


def dataset_psnr(test_path, upscaler, limit_data=None):
    '''Computes average PSNR of a given dataset between ground truth and Neural net upscaled images.
       PSNR is computed on final reconstructed images, which makes it comparable
       across different upscaling methods (e.g. different number of channels).'''

    image_names = os.listdir(test_path)
    psnr_vector = np.array([])
    
    # Limit samples if specified
    if limit_data == None: upper = len(image_names)
    else: upper = limit_data
    
    for i in range(upper):
            
        image_name = image_names[i]
        if image_name.endswith('jpg') or image_name.endswith('png'):
            
            img = load_img(test_path+'/'+image_name)
            
            # Adjust resolution to an even number
            img = tf.image.resize(img, (img.size[1] // 2 * 2, img.size[0] // 2 * 2))
            img = array_to_img(img)
            
            shrinked = downscale(img)
            nn = upscaler.upscale(shrinked)

            psnr = tf.image.psnr(img_to_array(nn), img_to_array(img), max_val=255).numpy()
            psnr_vector = np.append(psnr_vector, psnr)
            
    return psnr_vector
    
    
def dataset_psnr_bicubic(test_path, limit_data=None):
    '''Computes PSNR of a given dataset between ground truth and bicubic upscaled images.
    Useful for benchmarking.'''
    
    image_names = os.listdir(test_path)
    psnr_vector = np.array([])
    
    # Limit samples if specified
    if limit_data == None: upper = len(image_names)
    else: upper = limit_data
    
    for i in range(upper):
            
        image_name = image_names[i]
        if image_name.endswith('jpg') or image_name.endswith('png'):
            
            img = load_img(test_path+'/'+image_name)
            
            # Adjust resolution to an even number
            img = tf.image.resize(img, (img.size[1] // 2 * 2, img.size[0] // 2 * 2))
            img = array_to_img(img)
            
            shrinked = downscale(img)
            bic_img = bicubic_upscale(shrinked)

            psnr = tf.image.psnr(img_to_array(bic_img), img_to_array(img), max_val=255).numpy()
            psnr_vector = np.append(psnr_vector, psnr)
            
    return psnr_vector
