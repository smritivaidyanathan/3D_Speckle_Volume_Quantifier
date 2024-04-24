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
import math




'''
Method to convert z-stack nd2 files into 3D arrays containing pixel data. Returns
array, method is called within loopThroughAllImages.
'''
def nd2converter(file_path, nd2_file, channel_num):
    nd2_file_path = file_path + "nd2/" + nd2_file
    nd2 = nd2reader.ND2Reader(nd2_file_path)
    #Uncomment the below line if you would like the program to output the metadata of your nd2 file. 
    #print(nd2.metadata)
    print(f"{nd2.sizes['z']} z-stacks in {nd2_file_path}/nd2/ found.")
    arr = []
    print("Adding z-stacks...")
    percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for z in range (nd2.sizes['z']):
        arr.append(nd2.get_frame_2D(channel_num, 0, z, 0, 0, 0))
        progress = math.floor(float(z/nd2.sizes['z']) * 100)
        if (progress >= percentiles[0]):
            print(f"{percentiles[0]}% done.")
            percentiles.remove(percentiles[0])
    arr = np.array(arr, dtype='uint16')
    print(f"100% done.")
    print("Done adding z-stacks.")

    z, h, w = np.shape(arr)
    if (z == nd2.sizes['z']):
        print(f"Success! Array with {z} z-stacks of {w}x{h} size images was created")
    else:
        print(f"Uh oh! Something went wrong. Array with {z} z-stacks of {w}x{h} size images was created. Is this what you expected?")
        return []
    tiff_file_path = file_path + "tiff"
    #IMPORTANT - uncomment the below io.imsave line if you desire to see the segmentation results. 
    #Tiffs will be saved into the tiff folder. Necessary to call viewSegmentation!
    #io.imsave(f'{tiff_file_path}/{os.path.basename(nd2_file_path)}_C-{channel_num}.tiff', arr, check_contrast=False)
    return arr


'''
Method to convert a single 3D array of pixel values (arr) from one z-stack into binary image, and then segment 
into volumes using measure.label. Resulting volumes are returned.

Threshold value can be modified by changing the value of n when calling this method. 

Called within loopThroughAllImages
'''
def imageProcessor(arr, n):
    image = arr
    speckle_volumes = []
    sigma = 1.0
    smoothed_image = gaussian_filter(image, sigma=sigma)
    threshold_new = np.mean(smoothed_image) + (n*np.std(smoothed_image))
    binary_image3 = smoothed_image > threshold_new
    
    #segmentation
    labeled_image = measure.label(binary_image3, connectivity=None)  
    print("Image thresholded. Segmenting volumes.")
    regions = measure.regionprops(labeled_image)
    for region in regions:
        volume = region.area
        speckle_volumes.append(volume)
    return speckle_volumes

'''
Method similar to imageProcessor, except also launches Napari viewer to view the 
resulting segmentation. 

Threshold value can be modified by changing the value of n when calling this method. 

IMPORTANT: path must be a path of a single tiff image. If a tiff file needs to be generated, uncomment io.imsave 
line in nd2converter. 
'''
def viewSegmentation (path, n):
    image = np.stack(io.imread(path), axis = 0)
    print("a")
    #threshold
    sigma = 1.0
    smoothed_image = gaussian_filter(image, sigma=sigma)
    threshold_new = np.mean(smoothed_image) + (n*np.std(smoothed_image))
    binary_image3 = smoothed_image > threshold_new
    print("b")
    #label our volumes
    labeled_image = measure.label(binary_image3, connectivity=None)  
    speckle_volumes = []
    regions = measure.regionprops(labeled_image)
    for region in regions:
        volume = region.area
        speckle_volumes.append(volume)
    print("c")
    #launch 3D viewer
    viewer = napari.Viewer()
    viewer.add_image(image)
    viewer.add_image(binary_image3)
    viewer.add_labels(labeled_image)
    viewer.show()

    # creates a histogram of speckle volumes <400 pixels
    plt.hist(speckle_volumes, bins = 50, edgecolor='k', alpha=0.7, range=(0, 400))
    plt.xlabel('Speckle Volume (pixels)')
    plt.ylabel('Frequency (number)')
    plt.title(f"Distribution of Speckle Volumes N = {len(speckle_volumes)}")
    plt.grid(True)
    plt.show()

    napari.run()

'''
Helper method to get the nd2 file names within a single folder of a certain background. 
'''
def getFileNames(directory, backgrounds):
    count = 0
    files_names = [[] for background in backgrounds]
    for filename in os.listdir(directory):
        for i in range (len(backgrounds)):
            if backgrounds[i] in filename:
                files_names[i].append(filename)
    return files_names

'''
Helper method to write the volumes from a single background into a csv. Csv is saved in csv 
folder within experiment folder.
'''
def writeVolumesToCSV(volumes, background, csv_path):
    print("Writing to CSV")
    csv_file_path = csv_path + background + ".csv"

    with open(csv_file_path, mode='w', newline='') as file:
        csvwriter = csv.writer(file)
        for movie in volumes:
            csvwriter.writerow(movie) 

'''
!!!MAIN METHOD!!! 
Loops through all backgrounds, all files within background, 
processes each z-stack image, segments into volumes, and saves out results
to csv files for each background. 

This is the method you can make a call to, and customize the parameters based on your experiment. 

Example call: loopThroughAllImages("/Smriti/Larschan_Lab/Condensate_Volumes/in_vitro/", ["CLAMP_WT", "CLAMP_delPRLD", "CLAMP_RNAi"], 0, 2, 10)
--------------------------------------------------------------------------------
Parameters:

path = a string path to your experiment folder. Should include a / at the end!
ex: "/Smriti/Larschan_Lab/Condensate_Volumes/in_vitro/"
Guidelines:
1. This path must be accessible at the time of the call of the method. 
2. This path should include the following folders, spelt the same way:
- figs (optional, but required if data_analystics.py is invoked!)
- tiff (optional, but required if viewSegmentation is called!)
- nd2 (containing all of your nd2 files). note the naming convention mentioned at the end
- csv (can be empty to begin). Where all of the results from this method will be saved.
----------------------------------------
backgrounds = a list of strings containing the names of your experimental backgrounds. 

This is what the program uses to find appropriate nd2 files in your nd2 folder. 
Note the naming convention mentioned at the end.
Ex: ["CLAMP_WT", "CLAMP_delPRLD", "CLAMP_RNAi"]
----------------------------------------
channel_num = an int (0 indexed) representing the channel number in the nd2 that you want 
to analyze. 
Ex: 0

Note that this channel number will be the same for all backgrounds you input into
a single call of this method. If you want different channels for different backgrounds, 
you can seperate them into different calls. 
----------------------------------------
num_channels = an int representing the total number of channels in your image. 
Ex: 2

Very important to get this right, as this is used as a checkpoint to make sure that the nd2reader
interpretted your image correctly! Note that the number of channels will be assumed to be 
the same for all backgrounds you input into a single call of this method. If you want different
channel numbers for different backgrounds, you can seperate them into different calls. 
----------------------------------------
n = an int representing the threshold parameter n, as described in the thesis. 
Ex: 10. Technically you can also put float values in here, such as 5.5. 

A higher n value means more standard deviations will be added to the threshold 
to make the threshold higher. 
--------------------------------------------------------------------------------
~~~~Naming convention note! IMPORTANT!!~~~~~
nd2 files belonging to a specific background must include the name of that background
in the name of the file. 

For example: If one of my backgrounds is "CLAMP_WT", then all of my nd2 files that belong to the 
CLAMP_WT background must be named with CLAMP_WT in the name, spelled correctly. So in the nd2 folder, 
we may see something like. 
CLAMP_WT_00.nd2
CLAMP_WT_01.nd2
CLAMP_WT_02.nd2

This means you can (and should) put all your nd2 files of all your backgrounds into the nd2 folder, 
and you don't need to seperate them by internal folders. However, they must follow this naming convention.
'''
def loopThroughAllImages(path, backgrounds, channel_num, num_channels, n):
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
        nd2num = 1 #change
        for nd2 in file_name_background:
            tiff_path = f"{path_to_tiff}{os.path.basename(path_to_nd2 + nd2)}_C-{channel_num}.tiff"
            if not os.path.exists(tiff_path):
                print("Tiff file " + tiff_path + " does not exist. Creating array from nd2.") 
                result = nd2converter(path, nd2, channel_num)
                if (len(result) == 0):
                    print("Skipping ... Error found ...")
                    continue
            else:
                print("Tiff file " + tiff_path + " already exists. Proceeding to quantification.") 
            volumes_from_file = imageProcessor(result, n)    
            file_volumes.append(volumes_from_file)
            print(f"{nd2num}/{len(file_name_background)} files quantified")
            nd2num+=1
            print(f"Identified {len(volumes_from_file)} volumes from {tiff_path}") 
        num_vols_in_background += len(file_volumes)
        speckle_volumes.append(file_volumes)
        print(file_volumes)
        writeVolumesToCSV(file_volumes, backgrounds[i], path_to_csv)
        print(f"Identified {num_vols_in_background} volumes from {backgrounds[i]}") 
        print("---------------------------------------------------------")
        
    return speckle_volumes


'''
Make all your calls below! Note. I ran this pipeline in Python 3.11.6. I find that the nd2 converter does 
not work in other versions. If you want to use a different version, you will need to change the nd2 converter
method. 

Usage instructions:
1. Create experiment folder with the following folders inside: tiff, csv, figs, nd2
2. In nd2, place all your (appropriately named, see note in loopThroughAllImages documentation) 
nd2 files from each of your backgrounds
3. Make a call to loopThroughAllImages (make sure to read the documentation!)
4. If you want, make a call to viewSegmentation to test out different n values. You must uncomment the io.imsave
line in nd2converter if you want to do this!
5. Your results will be stored in csvs, with a csv file for each background you inputted. The results from 
each z-stack will be on a seperate row. 
'''

viewSegmentation("/Users/smriti/Desktop/3D_Speckle_Volume_Calculator/tiff_images_2/Number_2-Hrp38red and CLAMP1-300WTgreenwithTEV.nd2_C-0.tiff", 7)
#Example usage
path_to_exp = "/volumes/Research/BM_LarschanLab/Smriti/in_vitro/"
speckle_volumes = loopThroughAllImages(path_to_exp, ["CLAMP_WT"], 0, 2, 7)


