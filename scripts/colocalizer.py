import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage as ndi
from skimage import data, filters, measure, segmentation
import image_processor_volume_segmenter as image_p
import os
import nd2reader
import random
import math
from scipy import stats
import csv


'''
Method to convert z-stack nd2 files into 3D arrays containing pixel data. Returns
array, method is called within loopThroughAllImages.
'''
def nd2converter(file_path, nd2_file, channel_nums):
    nd2_file_path = file_path + "nd2/" + nd2_file
    nd2 = nd2reader.ND2Reader(nd2_file_path)
    #Uncomment the below line if you would like the program to output the metadata of your nd2 file. 
    #print(nd2.metadata)
    print(f"{nd2.sizes['z']} z-stacks in {nd2_file_path} found.")
    arr_channel1 = []
    arr_channel2 = []
    print("Adding z-stacks...")
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for z in range (nd2.sizes['z']):
        arr_channel1.append(nd2.get_frame_2D(channel_nums[0], 0, z, 0, 0, 0))
        arr_channel2.append(nd2.get_frame_2D(channel_nums[1], 0, z, 0, 0, 0))
        progress = math.floor(float(z/nd2.sizes['z']) * 100)
        if (progress >= percentiles[0]):
            print(f"{percentiles[0]}% done.")
            percentiles.remove(percentiles[0])
    arr1 = np.array(arr_channel1, dtype='uint16')
    arr2 = np.array(arr_channel2, dtype='uint16')
    print(f"100% done.")
    print("Done adding z-stacks.")
    return arr1, arr2

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

def prog_checker(path):
    already_visited = set([])
    with open(path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                already_visited.add(row[0])
    return already_visited
'''
Colocalizes all files of a specific background and returns a list of PCCs and pvals
from each nd2 file in that background. 
'''
def colocalize_background(path, list_of_files_in_background, background, channels, display_plot, path_to_csv):
    pccs = []
    pvals = []
    path_to_csv_pcc = path_to_csv + f"{background}_pcc.csv"
    path_to_csv_pvalue = path_to_csv + f"{background}_pvals.csv"
    path_to_csv_PROG = path_to_csv + "progress.csv"

    already_visited = prog_checker(path_to_csv_PROG)
    random.shuffle(list_of_files_in_background)
    for nd2 in list_of_files_in_background:
        if ("male" in background and "female" not in background and not "female" in nd2) or ("female" in background):
            if nd2 not in already_visited:
                channel1, channel2 = nd2converter(path, nd2, channels)
                pcc, pval = colocalize_file(channel1, channel2, display_plot)
                pccs.append(pcc)
                pvals.append(pval)
                with open(path_to_csv_PROG, 'a', newline='') as prog_csv:
                    with open(path_to_csv_pcc, 'a', newline='') as pccs_csv:
                        with open(path_to_csv_pvalue, 'a', newline='') as pvalues_csv:
                            writer3 = csv.writer(prog_csv)
                            writer3.writerow([nd2])
                            writer1 = csv.writer(pccs_csv)
                            writer1.writerow([pcc])
                            writer2 = csv.writer(pvalues_csv)
                            writer2.writerow([pval])
            else:
                print(f'{nd2} is already (apparently) processed and colocalized, so skipping. Check progress.csv if unsure!')
        else :
            print("Male/Female Error, ignoring file." + nd2)
    return pccs, pvals

"""
Checks if a CSV file exists at the specified path.
If not, creates a blank CSV file with that name.

Args:
file_path (str): The path to the CSV file.


"""

def create_blank_csv_if_not_exists(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([])

# Check and create the CSV file if necessary
csv_file_path = 'progress.csv'
create_blank_csv_if_not_exists(csv_file_path)

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
    create_blank_csv_if_not_exists(path_to_csv + "progress.csv")
    list_of_files_in_backgrounds = image_p.getFileNames(path_to_nd2, backgrounds)
    background_index = 0
    for list_background in list_of_files_in_backgrounds:
        pccs, pvals = colocalize_background(path, list_background, backgrounds[background_index], channels, display_plot, path_to_csv)
        pccs_backgrounds.append(pccs)
        pvals_backgrounds.append(pvals)
        background_index+=1
    return pccs_backgrounds, pvals_backgrounds
    
#path to nd2 folder. can be the same nd2 folder as the 3D quantification script!
path = "/volumes/Research/BM_LarschanLab/Smriti/Jan2024/"
#pccs_backgrounds, pvals_backgrounds = colocalize_all_backgrounds (path, ["female_con", "male_PrLD"], [0, 1])

"""
Read a CSV file and return a list of values from the file.
"""
def read_csv(file_path):
    values = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            values.append(float(row[0]))
    return values

def read_csvs(file_path, csvs):
    pccs = []
    for csv in csvs:
        row = read_csv(file_path + csv)
        pccs.append(row)
    return pccs

def make_mean_scatter(movie_wise_data_backgrounds, legend, title, save_path, show = False,  colors = ['blue', 'deepskyblue', 'red', 'magenta']):
    means = [np.mean(sublist) for sublist in movie_wise_data_backgrounds]
    legend_with_N = [f"{legend[i]} (N={len(movie_wise_data_backgrounds[i])})" for i in range(len(legend))]
    error = [1.96*(np.std(data)/(math.sqrt(len(data)))) for data in movie_wise_data_backgrounds]
    plt.bar(range(len(means)), means, yerr= error,color='gray', alpha=0.5, capsize=10)
    for i, column_data in enumerate(movie_wise_data_backgrounds):
        plt.scatter([i] * len(column_data), column_data, color=colors[i], label=f'{legend[i]}', alpha = 0.5)
    plt.xticks(range(len(legend_with_N)), legend_with_N)
    plt.ylabel('Pearson Colocalization Coefficient')
    plt.title(f'{title}')
    #plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()

'''
Calculates pairwise p and t values of all data in data and outputs them into a csv file in /csv/. 

data should be of type list of list of volumes, where outer list is per background, inner list is 
pooled values from a specific background
'''
def calculate_p_and_t_values(data, legend, path_to_csv):
    p_values_table = np.zeros((len(data),len(data)))
    t_values_table = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            t_value, p_value = stats.ttest_ind(data[i], data[j])
            p_values_table[i, j] = p_value
            t_values_table[i, j] = t_value
            p_values_table[j, i] = p_value
            t_values_table[j, i] = t_value
    with open("p_and_t_values.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([''] + legend)
        for i, row in enumerate(p_values_table):
            writer.writerow([legend[i]] + list(row))
        writer.writerow('\n\n\n')
        writer.writerow([''] + legend)
        for i, row in enumerate(t_values_table):
            writer.writerow([legend[i]] + list(row))

def make_plot():
    csvs = ["male_con_pcc.csv", "male_PrLD_pcc.csv", "female_con_pcc.csv", "female_PrLD_pcc.csv"]
    pccs = read_csvs((path + "csv/"), csvs)
    make_mean_scatter(pccs, ["Male WT", "Male delPrLD",  "Female WT", "Female delPrLD"], "Hrp38 and CLAMP Pearson Colocalization Coefficients", path, show = True)
    calculate_p_and_t_values(pccs, ["Male WT", "Male delPrLD",  "Female WT", "Female delPrLD"], "pvalues.csv")
    


make_plot()