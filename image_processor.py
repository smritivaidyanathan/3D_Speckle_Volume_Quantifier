import numpy as np
import skimage.io as io
import skimage.feature
import nd2reader
from skimage import io
from skimage import filters, segmentation, measure, exposure
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter
import napari
import matplotlib.pyplot as plt
import os



def nd2converter(nd2_file_path, channel_num):
    nd2 = nd2reader.ND2Reader(nd2_file_path)
    print(nd2.sizes['z'])
    arr = []
    for z in range (nd2.sizes['z']):
        arr.append(nd2.get_frame_2D(channel_num, 0, z, 0, 0, 0))
    arr = np.array(arr, dtype='uint16')
    io.imsave(f'tiff_images/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff', arr)
    return (f'tiff_images/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff')


def imageProcessor(path):
    # Load your 3D stack image
    image = np.stack(io.imread(path), axis = 0)

    # Apply Gaussian smoothing to reduce noise.
    sigma = 1.0
    smoothed_image = gaussian_filter(image, sigma=sigma)


    # Thresholding
    binary_image3 = smoothed_image > 632



    # 3D Region Segmentation _ just using connectivity, like a fill bucket tool!
    labeled_image = measure.label(binary_image3, connectivity=None)  # Adjust the connectivity as needed
    print(labeled_image)

    # Volume Calculation
    regions = measure.regionprops(labeled_image)
    speckle_volumes = []
    for region in regions:
        volume = region.area  # In voxels
        speckle_volumes.append(volume)
    return speckle_volumes

def viewSegmentation (path):
    image = np.stack(io.imread(path), axis = 0)

    # Apply Gaussian smoothing to reduce noise.
    sigma = 1.0
    smoothed_image = gaussian_filter(image, sigma=sigma)


    # Thresholding
    binary_image3 = smoothed_image > 632



    # 3D Region Segmentation _ just using connectivity, like a fill bucket tool!
    labeled_image = measure.label(binary_image3, connectivity=None)  # Adjust the connectivity as needed

    # Volume Calculation
    regions = measure.regionprops(labeled_image)
        #Convert voxel volume to physical units if your image is properly calibrated
    viewer = napari.Viewer()

    # Add the original 3D image

    viewer.add_image(binary_image3)
    viewer.add_labels(labeled_image)

    # for region in regions:
    #     viewer.add_labels(labeled_image, name=f"Region {region.label}")

    viewer.show()

    # Volume Calculation
    regions = measure.regionprops(labeled_image)
    speckle_volumes = []
    for region in regions:
        volume = region.area  # In voxels
        speckle_volumes.append(volume)

    # Create a histogram of speckle volumes
    plt.hist(speckle_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400))
    plt.xlabel('Speckle Volume (pixels)')
    plt.ylabel('Frequency (number)')
    plt.title('Distribution of Speckle Volumes')
    plt.grid(True)


    # Show the plot
    plt.show()

    napari.run()



def loopThroughAllImages(path_to_nd2, channel_num):
    viewSegmentation(nd2converter(path_to_nd2, channel_num))

path_to_folder = "/volumes/Research/BM_LarschanLab/mukulika/5_13_21_GFPlines+CLAMP(red)" + "/Hrp38GFP_CLAMP(red)_con001.nd2"
loopThroughAllImages(path_to_folder, 0)