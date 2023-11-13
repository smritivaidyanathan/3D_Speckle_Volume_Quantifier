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



def nd2converter(nd2_file_path, channel):
    nd2 = nd2reader.Nd2(nd2_file_path)
    channel = nd2.select(channels=channel)
    io.imsave(f'tiff_images/_channel_{channel}.tiff', nd2[0])


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

# print(imageProcessor("Hrp38GFP_CLAMP(red)_con001.nd2 - C=2.tif"))
viewSegmentation("tiff_images/hi.tif")

path_to_folder = "/volumes/Research-1/BM_LarschanLab/mukulika/5_13_21_GFPlines+CLAMP(red)"
nd2converter(path_to_folder+"/Hrp38GFP_CLAMP(red)_con.nd2", "405")