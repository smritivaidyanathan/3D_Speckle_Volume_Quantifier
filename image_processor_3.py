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
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from scipy import stats
import csv





def nd2converter(file_path, nd2_file, channel_num):
    nd2_file_path = file_path + "nd2/" + nd2_file
    nd2 = nd2reader.ND2Reader(nd2_file_path)
    print(f"{nd2.sizes['z']} z-stacks in {nd2_file_path}/nd2/ found.")
    arr = []
    print("Adding z-stacks...")
    for z in range (nd2.sizes['z']):
        arr.append(nd2.get_frame_2D(channel_num, 0, z, 0, 0, 0))
        progress = round(float(z/nd2.sizes['z']) * 100)
        if (progress % 10 == 0):
            print(f"{progress}% done.")
    arr = np.array(arr, dtype='uint16')
    print("Done adding z-stacks.")

    z, h, w = np.shape(arr)
    if (z == nd2.sizes['z'] and nd2.sizes['c'] == 2):
        print(f"Success! Tiff with {z} z-stacks of {w}x{h} size images was created")
    else:
        print(f"Uh oh! Something went wrong. Tiff with {z} z-stacks of {w}x{h} size images was created. Is this what you expected?")
        return -1
    tiff_file_path = file_path + "tiff"
    io.imsave(f'{tiff_file_path}/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff', arr)
    return (f'{tiff_file_path}/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff')


def imageProcessor(path):
    # Load your 3D stack image
    image = np.stack(io.imread(path), axis = 0)
    speckle_volumes = []
    # Apply Gaussian smoothing to reduce noise.
    sigma = 1.0
    smoothed_image = gaussian_filter(image, sigma=sigma)


    # Thresholding
    threshold_new = np.mean(smoothed_image) + (9*np.std(smoothed_image))
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
    threshold_new = np.mean(smoothed_image) + (9*np.std(smoothed_image))
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

def getFileNames(directory, backgrounds):
    count = 0
    files_names = [[] for background in backgrounds]
    for filename in os.listdir(directory):
        for i in range (len(backgrounds)):
            if backgrounds[i] in filename:
                files_names[i].append(filename)
    return files_names

def writeVolumesToCSV(volumes, background, csv_path):
    print("Writing to CSV")
    # Specify the CSV file path
    csv_file_path = csv_path + background + ".csv"


    with open(csv_file_path, mode='w', newline='') as file:
        csvwriter = csv.writer(file)
        for movie in volumes:
            csvwriter.writerow(movie) 


def loopThroughAllImages(path, backgrounds, channel_num):
    speckle_volumes = []
    path_to_nd2 = path + "nd2/"
    path_to_tiff =  path + "tiff/"
    path_to_csv =  path + "csv/"
    file_names = getFileNames(path_to_nd2, backgrounds)
    for i in range(len(backgrounds)):
        print(f"\nRunning {backgrounds[i]} images...")
        file_name_background = file_names[i]
        file_volumes = []
        num_vols_in_background = 0
        nd2num = 1
        for nd2 in file_name_background:
            tiff_path = f"{path_to_tiff}{os.path.basename(path_to_nd2 + nd2)}_C-{channel_num}.tiff"
            if not os.path.exists(tiff_path):
                print("Tiff file " + tiff_path + " does not exist yet. Converting from nd2.") 
                result = nd2converter(path, nd2, channel_num)
                if (result == -1):
                    print("Skipping ... Error found ...")
                    continue
            else:
                print("Tiff file " + tiff_path + " already exists. Proceeding to quantification.") 
            volumes_from_file = imageProcessor(tiff_path)    
            file_volumes.append(volumes_from_file)
            print(f"{nd2num}/{len(file_name_background)} files quantified")
            print(f"Identified {len(volumes_from_file)} volumes from {tiff_path}") 
        num_vols_in_background += len(file_volumes)
        speckle_volumes.append(file_volumes)
        print(file_volumes)
        writeVolumesToCSV(file_volumes, backgrounds[i], path_to_csv)
        print(f"Identified {num_vols_in_background} volumes from {backgrounds[i]}") 
        print("---------------------------------------------------------")
        
    return speckle_volumes






path_to_exp = "/volumes/Research/BM_LarschanLab/Mukulika/Feb2024/"

speckle_volumes = loopThroughAllImages(path_to_exp, ["Female_PrLD mutant"], 1)

viewSegmentation("/volumes/Research/BM_LarschanLab/Mukulika/Feb2024/tiff/Female_PrLD mutant_Hrp38GFP_SG.nd2_C-1.tiff")

# files = os.listdir("tiff_images_2")
# files = [f for f in files if os.path.isfile(os.path.join(f"{path_to_exp}/tiff", f))]
# print(f"{len(files)}/{len(control_images) + len(prld_images)} tiff created/used.")
# print(len(control_volumes))
# print(len(prld_volumes))


# print("Writing to CSVs")
# # Specify the CSV file path
# wt_file_path = path_to_data + "wt_volumes.csv"

# # Open the CSV file in write mode
# with open(wt_file_path, mode='w', newline='') as file:
#     # Create a CSV writer object
#     writer = csv.writer(file)

#     # Write each value as a separate row
#     for value in control_volumes:
#         writer.writerow([value])


# # Specify the CSV file path
# mutant_file_path = path_to_data + "mutant_volumes.csv"

# # Open the CSV file in write mode
# with open(mutant_file_path, mode='w', newline='') as file:
#     # Create a mutant_file_path writer object
#     writer = csv.writer(file)

#     # Write each value as a separate row
#     for value in prld_volumes:
#         writer.writerow([value])

# print("Making histograms now...")
# plt.hist(control_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 500), density = True, label = f"WT (N={len(control_volumes)})")
# plt.hist(prld_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 500), density = True, label = f"CLAMP PRLD Mutant (N={len(prld_volumes)})")
# plt.xlabel('Speckle Volume (pixels)')
# plt.ylabel('Count/Total Number')
# plt.title(f"In-Vitro Distribution of Speckle Volumes")
# #plt.title(f"Distribution of Speckle Volumes (N={len(speckle_volumes)})")
# plt.grid(True)
# plt.legend()
# plt.savefig(path_to_figs + 'In-Vitro Distribution of Speckle Volumes')
# plt.show()


# plt.hist(control_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 500), density = True, label = "WT")
# plt.xlabel('Speckle Volume (pixels)')
# plt.ylabel('Count/Total Number')
# plt.title(f"In-Vitro Distribution of Speckle Volumes (N={len(control_volumes)}) (WT)")
# plt.grid(True)
# plt.ylim(top=0.040) 
# plt.legend()
# plt.savefig(path_to_figs + 'In-Vitro Distribution of Speckle Volumes (WT)')
# plt.show()

# plt.hist(prld_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 500), density = True, label = "CLAMP PRLD Mutant")
# plt.xlabel('Speckle Volume (pixels)')
# plt.ylabel('Count/Total Number')
# plt.title(f"In-Vitro Distribution of Speckle Volumes (N={len(prld_volumes)}) (CLAMP PRLD Mutant)")
# plt.grid(True)
# plt.ylim(top=0.040) 
# plt.legend()
# plt.savefig(path_to_figs + 'In-Vitro Distribution of Speckle Volumes (M)')
# plt.show()


# conditions = [f'WT (N={len(control_volumes)})', f'CLAMP PRLD Mutant (N={len(prld_volumes)})']
# data = {
#     f'WT (N={len(control_volumes)})': control_volumes,
#     f'CLAMP PRLD Mutant (N={len(prld_volumes)})': prld_volumes
# }

# means = [np.mean(data[condition]) for condition in conditions]
# confidence_intervals = [stats.t.interval(0.95, len(data[condition])-1, loc=np.mean(data[condition])) for condition in conditions]
# print(means)
# print(confidence_intervals)

# yerr1=[[abs(means[0] - confidence_intervals[0][0]), abs(means[1] - confidence_intervals[1][0])], [abs(means[0] - confidence_intervals[0][1]), abs(means[1] - confidence_intervals[1][1])]]


# plt.bar(conditions, means, yerr=yerr1, capsize=5, alpha=0.7, label='Mean with 95% CI')

# plt.xlabel('Experimental Conditions')
# plt.ylabel('Mean Speckle Size (pixels)')
# plt.title('In-Vitro Mean Speckle Size with 95% Confidence Intervals')
# plt.legend()
# plt.savefig(path_to_figs + 'In-Vitro Mean Speckle Size with 95% Confidence Intervals')
# plt.show()




# medians = [np.median(data[condition]) for condition in conditions]

# def medianbootstrapcli(data):
#     num_bootstraps = 1000
#     bootstrap_medians = []
#     for _ in range(num_bootstraps):
#         bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
#         median = np.median(bootstrap_sample)
#         bootstrap_medians.append(median)
#     confidence_interval1 = stats.t.interval(0.95, len(bootstrap_medians)-1, loc=np.mean(bootstrap_medians))
#     return confidence_interval1



 
# confidence_interval = [medianbootstrapcli(data[condition]) for condition in conditions]
# print(confidence_interval)
# print(medians)

# yerr2 = [[abs(medians[0] - confidence_interval[0][0]), abs(medians[1] - confidence_interval[1][0])], [abs(medians[0] - confidence_interval[0][1]), abs(medians[1] - confidence_interval[1][1])]]
# print(yerr2)

# plt.bar(conditions, medians, yerr=yerr2, capsize=5, alpha=0.7, label='Median with 95% CI')
# plt.xlabel('Experimental Conditions')
# plt.ylabel('Median Speckle Size (pixels)')
# plt.title('In-Vitro Median Speckle Size with 95% Confidence Intervals')
# plt.legend()
# plt.savefig(path_to_figs + 'In-Vitro Median Speckle Size with 95% Confidence Intervals')
# plt.show()


# variances = [np.var(data[condition]) for condition in conditions]

# def variancebootstrapcli(data):
#     num_bootstraps = 1000
#     bootstrap_variances = []
#     for _ in range(num_bootstraps):
#         bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
#         variance = np.var(bootstrap_sample)
#         bootstrap_variances.append(variance)
#     confidence_interval1 = stats.t.interval(0.95, len(bootstrap_variances)-1, loc=np.mean(bootstrap_variances))
#     return confidence_interval1



 
# confidence_interval_v = [variancebootstrapcli(data[condition]) for condition in conditions]
# yerr3 = [[abs(variances[0] - confidence_interval_v[0][0]), abs(variances[1] - confidence_interval_v[1][0])], [abs(variances[0] - confidence_interval_v[0][1]), abs(variances[1] - confidence_interval_v[1][1])]]

# plt.bar(conditions, variances, yerr=yerr3, capsize=5, alpha=0.7, label='Variance with 95% CI')

# plt.xlabel('Experimental Conditions')
# plt.ylabel('Variance')
# plt.title('In-Vitro Variance of Speckle Size')
# plt.legend()
# plt.savefig(path_to_figs + 'In-Vitro Variance with 95% CI of Speckle Size.png')
# plt.show()



# csv_file_path = "/Users/smriti/Desktop/3D_Speckle_Volume_Calculator/statistics.csv"

# # Open the CSV file in write mode
# with open(csv_file_path, mode='w', newline='') as file:
#     # Create a CSV writer object
#     writer = csv.writer(file)

#     # Write each value as a separate row
#     writer.writerow("Mean WT, Mean Mutant, Mean WT Left Error Bar, Mean WT Right Error Bar, Mean Mutant Left Error Bar, Mean Mutant Right Error Bar, Median WT, Median Mutant, Median WT Left Error Bar, Median WT Right Error Bar, Median Mutant Left Error Bar, Median Mutant Right Error Bar, Variance WT, Variance Mutant")
#     writer.writerow([means[0], means[1], confidence_intervals[0][0], confidence_intervals[0][1], confidence_intervals[1][0], confidence_intervals[1][1],  medians[0], medians[1], confidence_interval[0][0], confidence_interval[0][1], confidence_interval[1][0], confidence_interval[1][1], variances[0], variances[1]])


