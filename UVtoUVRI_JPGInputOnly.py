# -*- coding: utf-8 -*-
"""
Created on Mon May 25 02:34:05 2020

@author: cosmi
"""


"""
Experimental UV Absorption Index program using UV-Pass Filter on DJI Mavic 2 Pro 
JPEG 16-bit combo images taken using Ultraviolet-Only Pass Filter 
Useful for Batch Processing Multiple Images

%(c)-J. Campbell MuonRay Enterprises 2020 
This Python script was created using the Spyder Editor
"""
from scipy import misc

import warnings
warnings.filterwarnings('ignore')
import imageio
import numpy as np
from matplotlib import pyplot as plt  # For image viewing

#!/usr/bin/python
import getopt
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

#dng reading requires libraw to work

# Open an image
image = misc.imread('DJI_0880.JPG')

# Get the red band from the rgb image, and open it as a numpy matrix
#NIR = image[:, :, 0]
         
#ir = np.asarray(NIR, float)


ir = (image[:,:,0]).astype('float')


# Get one of the IR image bands (all bands should be same)
#blue = image[:, :, 2]

#r = np.asarray(blue, float)

r = (image[:,:,2]).astype('float')


g = (image[:,:,1]).astype('float')

#(NIR + Green)
irg = np.add(ir, g)
       
       
L=0.5;
       
rplusb = np.add(ir, r)
rplusbplusg = np.add(ir, r, g)
rminusb = np.subtract(ir, r)
oneplusL = np.add(1, L)
# Create a numpy matrix of zeros to hold the calculated UVRI values for each pixel
uvri = np.zeros(r.size)  # The UVRI image will be the same size as the input image

# Calculate UV Reflectance Index

uvri = np.true_divide(np.subtract(g, rminusb), np.add(g, rplusb))


# Display the results
output_name = 'UVReflectanceIndex.jpg'

#a nice selection of grayscale colour palettes
cols1 = ['blue', 'green', 'yellow', 'red']
cols2 =  ['gray', 'gray', 'red', 'yellow', 'green']
cols3 = ['gray', 'blue', 'green', 'yellow', 'red']

cols4 = ['black', 'gray', 'blue', 'green', 'yellow', 'red']

def create_colormap(args):
    return LinearSegmentedColormap.from_list(name='custom1', colors=cols4)

#colour bar to match grayscale units
def create_colorbar(fig, image):
        position = fig.add_axes([0.125, 0.19, 0.2, 0.05])
        norm = colors.Normalize(vmin=-1., vmax=1.)
        cbar = plt.colorbar(image,
                            cax=position,
                            orientation='horizontal',
                            norm=norm)
        cbar.ax.tick_params(labelsize=6)
        tick_locator = ticker.MaxNLocator(nbins=3)
        cbar.locator = tick_locator
        cbar.update_ticks()
        cbar.set_label("UVRI", fontsize=10, x=0.5, y=0.5, labelpad=-25)

fig, ax = plt.subplots()
image = ax.imshow(uvri, cmap=create_colormap(colors))
plt.axis('off')

create_colorbar(fig, image)

extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig(output_name, dpi=600, transparent=True, bbox_inches=extent, pad_inches=0)
        # plt.show()