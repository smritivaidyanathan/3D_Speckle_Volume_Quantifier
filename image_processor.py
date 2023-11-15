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
    if (z == nd2.sizes['z'] and nd2.sizes['c'] == 3):
        print(f"Success! Tiff with {z} z-stacks of {w}x{h} size images was created")
    else:
        print(f"Uh oh! Something went wrong. Tiff with {z} z-stacks of {w}x{h} size images was created. Is this what you expected?")
        return -1
    io.imsave(f'tiff_images/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff', arr)
    return (f'tiff_images/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff')


def imageProcessor(path, speckle_volumes):
    # Load your 3D stack image
    image = np.stack(io.imread(path), axis = 0)

    # Apply Gaussian smoothing to reduce noise.
    sigma = 1.0
    smoothed_image = gaussian_filter(image, sigma=sigma)


    # Thresholding
    print(np.mean(smoothed_image))
    print(np.std(smoothed_image))
    threshold_new = np.mean(smoothed_image) + (8*np.std(smoothed_image))
    print(threshold_new)
    # Thresholding
    binary_image3 = smoothed_image > threshold_new



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

    print(np.mean(smoothed_image))
    print(np.std(smoothed_image))
    threshold_new = np.mean(smoothed_image) + (8*np.std(smoothed_image))
    print(threshold_new)
    # Thresholding
    binary_image3 = smoothed_image > threshold_new



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
            print("Tiff file " +tiff_path + " already exists.")
            speckle_volumes = imageProcessor(tiff_path, speckle_volumes)
            print("Identified " + str(len(speckle_volumes) - numvol) + " volumes from " + nd2)
        else:
            numvol = len(speckle_volumes)
            print("Tiff file " + tiff_path + " does not exist yet. Converting from nd2.") 
            result = nd2converter(path_to_nd2 + nd2, channel_num)
            if (result == -1):
                print("Skipping ... Error found ...")
                continue
            speckle_volumes = imageProcessor(result, speckle_volumes)
            print("Identified " + str(len(speckle_volumes) - numvol) + " volumes from " + nd2)
        print("---------------------------------------------------------")
    return speckle_volumes

print("\nRunning control images...")
control_images = ["Hrp38GFP_CLAMP(red)_con001.nd2", "Hrp38GFP_CLAMP(red)_con002.nd2", "Hrp38GFP_CLAMP(red)_con003.nd2", "Hrp38GFP_CLAMP(red)_con004.nd2", "Hrp38GFP_CLAMP(red)_con005.nd2", "Hrp38GFP_CLAMP(red)_con006.nd2", "Hrp38GFP_CLAMP(red)_con007.nd2"]
control_volumes = loopThroughAllImages("/volumes/Research/BM_LarschanLab/mukulika/5_13_21_GFPlines+CLAMP(red)/", control_images,  0)
print("\n ============================================================================ \n")
print("\nRunning clamp null images...")
clamp2null_images = ["Hrp38GFPGFP_CLAMP(red)_Clamp2(null).nd2", "Hrp38GFPGFP_CLAMP(red)_Clamp2(null)001.nd2"]
clamp2null_volumes = loopThroughAllImages("/volumes/Research/BM_LarschanLab/mukulika/5_13_21_GFPlines+CLAMP(red)/", clamp2null_images,  0)
print("\n ============================================================================ \n")
print("\nRunning clamp(h) images...")
clamph_images = ["Hrp38GFPGFP_CLAMP(red)_ClampH.nd2", "Hrp38GFPGFP_CLAMP(red)_ClampH001.nd2", "Hrp38GFPGFP_CLAMP(red)_ClampH002.nd2", "Hrp38GFPGFP_CLAMP(red)_ClampH003.nd2"]
clamph_volumes = loopThroughAllImages("/volumes/Research/BM_LarschanLab/mukulika/5_13_21_GFPlines+CLAMP(red)/", clamph_images,  0)
print("\n ============================================================================ \n")
print("\nRunning clamp(i) images...")
clampi_images = ["Hrp38GFPGFP_CLAMP(red)_ClampI.nd2", "Hrp38GFPGFP_CLAMP(red)_ClampI001.nd2", "Hrp38GFPGFP_CLAMP(red)_ClampI002.nd2", "Hrp38GFPGFP_CLAMP(red)_ClampI003.nd2"]
clampi_volumes = loopThroughAllImages("/volumes/Research/BM_LarschanLab/mukulika/5_13_21_GFPlines+CLAMP(red)/", clampi_images,  0)
print("\n ============================================================================ \n")
print("\nAll done!!! :)")

files = os.listdir("tiff_images")
files = [f for f in files if os.path.isfile(os.path.join("tiff_images", f))]
print(f"{len(files)}/{len(control_images) + len(clamp2null_images) +  len(clamph_images) + len(clampi_images)} tiff created/used.")

print("Making histograms now...")
plt.hist(control_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400), density = True, label = "Control")
plt.hist(clamp2null_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400), density = True, label = "Clamp2 (null)")
plt.hist(clamph_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400), density = True, label = "Clamp(H)")
plt.hist(clampi_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400), density = True, label = "Clamp(I)")
plt.xlabel('Speckle Volume (pixels)')
plt.ylabel('Probability (number)')
plt.title(f"Distribution of Probabilities of Speckle Volumes")
#plt.title(f"Distribution of Speckle Volumes (N={len(speckle_volumes)})")
plt.grid(True)
plt.legend()
plt.show()


plt.hist(control_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400), density = True, label = "Control")
plt.xlabel('Speckle Volume (pixels)')
plt.ylabel('Probability (number)')
plt.title(f"Distribution of Speckle Volumes (N={len(control_volumes)}) (Control)")
plt.grid(True)
plt.legend()
plt.show()

plt.hist(clamp2null_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400), density = True, label = "Clamp2 (null)")
plt.xlabel('Speckle Volume (pixels)')
plt.ylabel('Probability (number)')
plt.title(f"Distribution of Speckle Volumes (N={len(clamp2null_volumes)}) (Clamp2 (Null))")
plt.grid(True)
plt.legend()
plt.show()

plt.hist(clamph_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400), density = True, label = "Clamp(H)")
plt.xlabel('Speckle Volume (pixels)')
plt.ylabel('Probability (number)')
plt.title(f"Distribution of Speckle Volumes (N={len(clamph_volumes)}) (Clamp (H))")
plt.grid(True)
plt.legend()
plt.show()

plt.hist(clampi_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400), density = True, label = "Clamp(I)")
plt.xlabel('Speckle Volume (pixels)')
plt.ylabel('Probability (number)')
plt.title(f"Distribution of Speckle Volumes (N={len(clampi_volumes)}) (Clamp (I))")
plt.grid(True)
plt.legend()
plt.show()
