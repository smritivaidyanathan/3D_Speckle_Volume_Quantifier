import numpy as np
import skimage.io as io
import skimage.feature
from skimage import filters, segmentation, measure, exposure
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter
import napari


# Load your 3D stack image
image = np.stack(io.imread('Hrp38GFP_CLAMP(red)_con001.nd2 - C=1.tif'), axis = 0)

# Preprocessing
# Background Subtraction
# You can use methods like rolling ball background subtraction or simple mean filtering.
# Adjust the 'ball_radius' as needed.
# ball_radius = 5
# background_subtracted_image = image - filters.rank.mean(image, selem=np.ones((ball_radius, ball_radius, ball_radius)))

# Smoothing
# Apply Gaussian smoothing to reduce noise.
sigma = 1.0
smoothed_image = gaussian_filter(image, sigma=sigma)

# Contrast Adjustment
# Enhance contrast using linear stretching (0.5% saturation)
# p1, p99 = np.percentile(smoothed_image, (0.5, 99.5))
# contrast_adjusted_image = exposure.rescale_intensity(smoothed_image, in_range=(p1, p99))

# Thresholding

threshold_value3 = filters.threshold_li(smoothed_image, tolerance = 0)



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
napari.run()
