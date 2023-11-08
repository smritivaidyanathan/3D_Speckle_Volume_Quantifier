import numpy as np
import skimage.io as io
import skimage.feature
from skimage import filters, segmentation, measure, exposure
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter
import napari
import matplotlib.pyplot as plt


# Load your 3D stack image
image = np.stack(io.imread('Hrp38GFP_CLAMP(red)_con001.nd2 - C=2.tif'), axis = 0)

# Apply Gaussian smoothing to reduce noise.
sigma = 1.0
smoothed_image = gaussian_filter(image, sigma=sigma)


# Thresholding
binary_image3 = smoothed_image > 632



# 3D Region Segmentation
labeled_image = measure.label(binary_image3, connectivity=None)  # Adjust the connectivity as needed
print(labeled_image)

# Volume Calculation
regions = measure.regionprops(labeled_image)
speckle_volumes = []
for region in regions:
    volume = region.area  # In voxels
    speckle_volumes.append(volume)

#Convert voxel volume to physical units if your image is properly calibrated
viewer = napari.Viewer()

# Add the original 3D image

viewer.add_image(binary_image3)
viewer.add_labels(labeled_image)

# for region in regions:
#     viewer.add_labels(labeled_image, name=f"Region {region.label}")

viewer.show()

print(speckle_volumes)

# Create a histogram of speckle volumes
plt.hist(speckle_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400))
plt.xlabel('Speckle Volume (pixels)')
plt.ylabel('Frequency (number)')
plt.title('Distribution of Speckle Volumes')
plt.grid(True)

# Show the plot
plt.show()

napari.run()
