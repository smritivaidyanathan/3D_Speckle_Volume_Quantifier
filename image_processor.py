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
    print(f"{nd2.sizes['z']} z-stacks in {nd2_file_path} found.")
    arr = []
    print("Adding z-stacks...")
    for z in range (nd2.sizes['z']):
        arr.append(nd2.get_frame_2D(channel_num, 0, z, 0, 0, 0))
        progress = round(float(z/nd2.sizes['z']) * 100)
        print(f"{round(float(z/nd2.sizes['z']) * 100)}% done.")
    arr = np.array(arr, dtype='uint16')
    print("Done adding z-stacks.")

    z, h, w = np.shape(arr)
    if (z == nd2.sizes['z']):
        print(f"Success! Tiff with {z} z-stacks of {w}x{h} size images was created")
    else:
        print(f"Uh oh! Something went wrong. Tiff with {z} z-stacks of {w}x{h} size images was created. Is this what you expected?")
        return
    io.imsave(f'tiff_images/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff', arr)
    return (f'tiff_images/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff')


def imageProcessor(path, speckle_volumes):
    # Load your 3D stack image
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
    speckle_volumes = []
    regions = measure.regionprops(labeled_image)
    for region in regions:
        volume = region.area  # In voxels
        speckle_volumes.append(volume)

    # Create a histogram of speckle volumes
    plt.hist(speckle_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400))
    plt.xlabel('Speckle Volume (pixels)')
    plt.ylabel('Frequency (number)')
    plt.title(f"Distribution of Speckle Volumes N = {len(speckle_volumes)}")
    plt.grid(True)


    # Show the plot
    plt.show()

    napari.run()



def loopThroughAllImages(path_to_nd2, nd2_image_names, channel_num):
    speckle_volumes = []
    for nd2 in nd2_image_names:
        tiff_path = f"tiff_images/{os.path.basename(path_to_nd2 + nd2)}_C-{channel_num}.tiff"
        if os.path.exists(tiff_path):
            numvol = len(speckle_volumes)
            print("\nTiff file " +tiff_path + " already exists.")
            speckle_volumes = imageProcessor(tiff_path, speckle_volumes)
            print("Identified " + str(len(speckle_volumes) - numvol) + " volumes from " + nd2)
        else:
            numvol = len(speckle_volumes)
            print("\nTiff file " + tiff_path + " does not exist yet. Converting from nd2.") 
            speckle_volumes = imageProcessor(nd2converter(path_to_nd2 + nd2, channel_num), speckle_volumes)
            print("Identified " + str(len(speckle_volumes) - numvol) + " volumes from " + nd2)
    # Create a histogram of speckle volumes
    print("\nDone. Creating Distribution Histogram.")
    plt.hist(speckle_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400))
    plt.xlabel('Speckle Volume (pixels)')
    plt.ylabel('Frequency (number)')
    plt.title(f"Distribution of Speckle Volumes (N={len(speckle_volumes)})")
    plt.grid(True)


    # Show the plot
    plt.show()

images = ["Hrp38GFP_CLAMP(red)_con001.nd2", "Hrp38GFP_CLAMP(red)_con002.nd2", "Hrp38GFP_CLAMP(red)_con003.nd2", "Hrp38GFP_CLAMP(red)_con004.nd2", "Hrp38GFP_CLAMP(red)_con005.nd2", "Hrp38GFP_CLAMP(red)_con006.nd2", "Hrp38GFP_CLAMP(red)_con007.nd2"]
loopThroughAllImages("/volumes/Research/BM_LarschanLab/mukulika/5_13_21_GFPlines+CLAMP(red)/", images,  0)