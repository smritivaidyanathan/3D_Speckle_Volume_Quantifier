import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi
from skimage import data, filters, measure, segmentation
from image_processor_volume_segmenter import nd2converter, getFileNames
import os
import csv


'''
Runs a test to see if all the packages are working properly. To use, just call the method!
'''
def colocalization_test():
    rng = np.random.default_rng()

    # segment nucleus
    nucleus = data.protein_transport()[0, 0, :, :180]
    smooth = filters.gaussian(nucleus, sigma=1.5)
    thresh = smooth > filters.threshold_otsu(smooth)
    fill = ndi.binary_fill_holes(thresh)
    nucleus_seg = segmentation.clear_border(fill)

    # protein blobs of varying intensity
    proteinA = np.zeros_like(nucleus, dtype="float64")
    proteinA_seg = np.zeros_like(nucleus, dtype="float64")

    for blob_seed in range(10):
        blobs = data.binary_blobs(
            180, blob_size_fraction=0.5, volume_fraction=(50 / (180**2)), rng=blob_seed
        )
        blobs_image = filters.gaussian(blobs, sigma=1.5) * rng.integers(50, 256)
        proteinA += blobs_image
        proteinA_seg += blobs

    proteinB = proteinA + rng.normal(loc=100, scale=10, size=proteinA.shape)

    # plot images
    fig, ax = plt.subplots(1, 2, figsize=(8, 8), sharey=True)

    black_magenta = LinearSegmentedColormap.from_list("", ["black", "magenta"])
    ax[0].imshow(proteinA, cmap=black_magenta)
    ax[0].set_title('Protein A')

    black_cyan = LinearSegmentedColormap.from_list("", ["black", "cyan"])
    ax[1].imshow(proteinB, cmap=black_cyan)
    ax[1].set_title('Protein B')

    for a in ax.ravel():
        a.set_axis_off()

    # plot pixel intensity scatter
    fig, ax = plt.subplots()
    ax.scatter(proteinA, proteinB)
    ax.set_title('Pixel intensity')
    ax.set_xlabel('Protein A intensity')
    ax.set_ylabel('Protein B intensity')
    plt.show()

    pcc, pval = measure.pearson_corr_coeff(proteinA, proteinB)
    print(f"PCC: {pcc:0.3g}, p-val: {pval:0.3g}")

'''
Colocalizes two 3D channels from an nd2 image. 

Both channels must be of type ndarray with 3 dimensions, and the 
dimensions of each channel must match. 

Set display_plot flag to indicate whether you would like to see the scatter plot for a particular
call of the method. 
'''
def colocalize_file(channel1, channel2, display_plot):

    flattened_channel1 = np.ravel(channel1)
    flattened_channel2 = np.ravel(channel2)

    if (display_plot):
        fig, ax = plt.subplots()
        ax.scatter(flattened_channel1, flattened_channel2)
        ax.set_title('Pixel intensity')
        ax.set_xlabel('Protein A intensity')
        ax.set_ylabel('Protein B intensity')
        plt.show()

    pcc, pval = measure.pearson_corr_coeff(channel1, channel2)
    print(f"PCC: {pcc:0.3g}, p-val: {pval:0.3g}")
    return pcc, pval

'''
Colocalizes all files of a specific background and returns a list of PCCs and pvals
from each nd2 file in that background. 
'''
def colocalize_background(path, list_of_files_in_background, channels, display_plot, path_to_csv):
    pccs = []
    pvals = []
    path_to_csv_pcc = path_to_csv + "pcc.csv"
    path_to_csv_pvalue = path_to_csv + "pvals.csv"
    with open(path_to_csv_pcc, 'a', newline='') as pccs_csv:
        with open(path_to_csv_pvalue, 'a', newline='') as pvalues_csv:
            for nd2 in list_of_files_in_background:
                channel1 = nd2converter(path, nd2, channels[0])
                channel2 = nd2converter(path, nd2, channels[1])
                pcc, pval = colocalize_file(channel1, channel2, display_plot)
                pccs.append(pcc)
                pvals.append(pval)
            writer1 = csv.writer(pccs_csv)
            writer1.writerow(pccs)
            writer2 = csv.writer(pvalues_csv)
            writer2.writerow(pvals)
    return pccs, pvals


'''
Colocalizes across all backgrounds listed in backgrounds for each 
corresponding nd2 which each background in path, for the two channels listed
in channels. Returns a list of list of pcc values and pvalues. 
'''
def colocalize_all_backgrounds(path, backgrounds, channels, display_plot = False):
    pccs_backgrounds = []
    pvals_backgrounds = []
    path_to_nd2 = path + "nd2/"
    path_to_csv = path + "csv/"
    list_of_files_in_backgrounds = getFileNames(path_to_nd2, backgrounds)
    for list_background in list_of_files_in_backgrounds:
        pccs, pvals = colocalize_background(path_to_nd2, list_background, channels, display_plot, path_to_csv)
        pccs_backgrounds.append(pccs)
        pvals_backgrounds.append(pvals)
    return pccs_backgrounds, pvals_backgrounds
    
#path to nd2 folder. can be the same nd2 folder as the 3D quantification script!
path = "/volumes/Research/BM_LarschanLab/Smriti/colocalization/nd2"
pccs_backgrounds, pvals_backgrounds = colocalize_all_backgrounds (path, [], [])




