import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from scipy import stats
import csv

import random
import math

'''
Makes distribution histogram with different backgrounds overlayed on top of each other

Parameters:
data_backgrounds - List of arrays of speckle data, where each outer list represents a background and each inner list represents 
the pooled values from all movies/z-stacks. Can be created from calling get_data_for_backgrounds on csv_data_dict output from csv_files_to_dict

legend - list of strings representing the names of each experimental background 
ex: ["Control (CLAMP WT)", "CLAMP delPRLD"]

title - string representing title of graph
ex: "Distribution of Speckle Volumes"

save_path - string path represnting where we want the figures to be saved. note that this will probably be in the figs folder 
if you follow the format of my template method at the bottom. Should end in a /

ylim (optional) - float/int - in case we want a ylim, optional parameter, is default = -1 meaning no ylim. 

ranges (optional) - None or range type, if we want to constrict the x limits, then set to = the range value, default is none. 

show (optional) - boolean represnting whether we want to display the plots. default is false (do not show)

bins (optional)- int representing how many bins we want in our histogram (default is 50)

colors (optional)- list of strings representing the colors we want to represent each background. colors are default ['blue', 'deepskyblue', 'red', 'magenta'].
See https://matplotlib.org/stable/gallery/color/named_colors.html for options

'''
def make_dist_histogram(data_backgrounds, legend, title, save_path, ylim = -1, ranges = None, show = False, bins=50, colors = ['blue', 'deepskyblue', 'red', 'magenta']):
    for i in range(len(data_backgrounds)):
        data = data_backgrounds[i]
        legend_label = legend[i]
        plt.hist(data, bins = bins, edgecolor='k', range=ranges, density = True, alpha = 0.75, color = colors[i], label = f"{legend_label} (N={len(data)})")
        plt.xlabel('Speckle Volume (pixels)')
        plt.ylabel('Count/Total Number')
    plt.title(f"{title}")
    plt.grid(True)
    if (ylim != -1):
        plt.ylim(top = ylim)
    plt.legend()
    if (show):
        plt.show()
    plt.savefig(f'{save_path}{title}')

'''
Makes basic mean bars representing mean from each background with 95% confidence bars. 

Parameters:
data_backgrounds - List of arrays of speckle data, where each outer list represents a background and each inner list represents 
the pooled values from all movies/z-stacks. Can be created from calling get_data_for_backgrounds on csv_data_dict output from csv_files_to_dict

legend - list of strings representing the names of each experimental background 
ex: ["Control (CLAMP WT)", "CLAMP delPRLD"]

title - string representing title of graph
ex: "Distribution of Speckle Volumes"

save_path - string path represnting where we want the figures to be saved. note that this will probably be in the figs folder 
if you follow the format of my template method at the bottom. Should end in a /

show (optional) - boolean represnting whether we want to display the plots. default is false (do not show)

'''
def make_mean_bar(data_backgrounds, legend, title, save_path,  show = False):
    means = [np.mean(data) for data in data_backgrounds]
    legend_with_N = [f"{legend[i]} (N={len(data_backgrounds[i])})" for i in range(len(legend))]
    
    error = [1.96*(np.std(data)/(math.sqrt(len(data)))) for data in data_backgrounds]
    plt.bar(legend_with_N, means, yerr=error,align = 'center', capsize=5, alpha=0.7)
    plt.xlabel('Experimental Conditions')
    plt.ylabel('Mean Speckle Size (pixels)')
    plt.title(f'{title}')
    plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()
    return means, error

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

'''
Makes mean bar chart with 95% confidence intervals for each background and also plots movie-wise means, movie wise meaning per z-stack. 

Parameters:
movie_wise_data_backgrounds - List of arrays of arrays of speckle data, where each outer list represents a background and middle lists represent each 
movie/z-stack, and each inner-most ist represents the values from a movie/z-stack. Can be created from calling get_data_for_background on csv_movie_wise_dict output from csv_files_to_dict

data_backgrounds - List of arrays of speckle data, where each outer list represents a background and each inner list represents 
the pooled values from all movies/z-stacks. Can be created from csv_data_dict output from calling get_data_for_backgrounds on csv_files_to_dict

legend - list of strings representing the names of each experimental background 
ex: ["Control (CLAMP WT)", "CLAMP delPRLD"]

title - string representing title of graph
ex: "Distribution of Speckle Volumes"

save_path - string path represnting where we want the figures to be saved. note that this will probably be in the figs folder 
if you follow the format of my template method at the bottom. Should end in a /

show (optional) - boolean represnting whether we want to display the plots. default is false (do not show)

colors (optional) - list of strings representing the colors we want to represent each background. colors are default ['blue', 'deepskyblue', 'red', 'magenta'].
See https://matplotlib.org/stable/gallery/color/named_colors.html for options

'''
def make_mean_scatter(movie_wise_data_backgrounds, data_backgrounds, legend, title, save_path, show = False,  colors = ['blue', 'deepskyblue', 'red', 'magenta']):
    means = [[np.mean(inner_list) for inner_list in sublist] for sublist in movie_wise_data_backgrounds]

    outer_means =[np.mean(data) for data in data_backgrounds]
    legend_with_N = [f"{legend[i]} (N={len(data_backgrounds[i])})" for i in range(len(legend))]
    error = [1.96*(np.std(data)/(math.sqrt(len(data)))) for data in data_backgrounds]
    plt.bar(range(len(outer_means)), outer_means, yerr= error,color='gray', alpha=0.5, capsize=10)
    for i, column_data in enumerate(means):
        plt.scatter([i] * len(column_data), column_data, color=colors[i], label=f'{legend[i]}', alpha = 0.5)
    plt.xticks(range(len(legend_with_N)), legend_with_N)
    plt.ylabel('Speckle Volume (pixels)')
    plt.title(f'{title}')
    plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()


'''
Makes mean or variance bars from each background with 95% confidence bars calculated
by bootstrapping. 

Parameters:
data_backgrounds - List of arrays of speckle data, where each outer list represents a background and each inner list represents
the pooled values from all movies/z-stacks. Can be created from calling get_data_for_backgrounds on csv_data_dict output from csv_files_to_dict

legend - list of strings representing the names of each experimental background 
ex: ["Control (CLAMP WT)", "CLAMP delPRLD"]

title - string representing title of graph
ex: "Distribution of Speckle Volumes"

save_path - string path represnting where we want the figures to be saved. note that this will probably be in the figs folder 
if you follow the format of my template method at the bottom. Should end in a /

show (optional) - boolean represnting whether we want to display the plots. default is false (do not show)

var (optional) - boolean representing whether we want to calculate variance (as opposed to the median). 
default to calculate the median, ie var = False. 

colors (optional) - list of strings representing the colors we want to represent each background. colors are default ['blue', 'deepskyblue', 'red', 'magenta'].
See https://matplotlib.org/stable/gallery/color/named_colors.html for options

'''
def make_med_or_var_bar(data_backgrounds, legend, title, save_path, var = False, show = False, colors = ['blue', 'deepskyblue', 'red', 'magenta']):
    outer_quals = [np.median(data) for data in data_backgrounds]
    ylabel = "Median"
    error = [bootstrapcli(data, var = False) for data in data_backgrounds]
    if var:
        outer_quals = [np.var(data) for data in data_backgrounds]
        print(outer_quals)
        ylabel = "Variance"
        error = [bootstrapcli(data, var = True) for data in data_backgrounds]
    legend_with_N = [f"{legend[i]} (N={len(data_backgrounds[i])})" for i in range(len(legend))]

    plt.bar(legend_with_N, height = outer_quals, yerr=error, align = 'center', color = colors, capsize=10, alpha=0.7)
    plt.xlabel('Experimental Conditions')
    plt.ylabel(f'{ylabel} of Speckle Size (pixels)')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()
    return outer_quals

'''
Helper method to construct confidence intervals using bootstrapping. 
'''
def bootstrapcli(data , var = False) :
    num_bootstraps = 1000
    bootstrap_quals = []
    for _ in range(num_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        qual = np.median(bootstrap_sample)
        if var:
            qual = np.var(bootstrap_sample)
        bootstrap_quals.append(qual)
    error =1.96*(np.std(bootstrap_quals)/(math.sqrt(len(bootstrap_quals))))

    return error


path_to_exp = "/volumes/Research/BM_LarschanLab/Mukulika/Feb2024/"


'''
Helper method to create necessary dictionary data structures for data analytics.

Creates csv_data_dict, csv_movie_wise_dict, dictionaries of backgrounds mapped to lists of information.
csv_data_dict - dictionary mapping background to list of pooled data from all z-stacks
csv_movie_wise_dict - dictionary mapping background to list of lists, where outer lists represent each movie/z-stack
and each inner list represents the volumes from that movie-z-stack

path_to_csv is a path to the csv folder. should end in /. 
ex: "/Smriti/Experiments/in-vivo/csv/"
'''
def csv_files_to_dict(path_to_csv):
    csv_data_dict = {}
    csv_movie_wise_dict = {}
    for filename in os.listdir(path_to_csv):
        if filename.endswith(".csv"):
            background = filename.split('.')[0]
            file_path = os.path.join(path_to_csv, filename)
            with open(file_path, 'r', newline='') as csv_file:
                csv_reader = csv.reader(csv_file)
                data = []
                data_movie = []
                for row in csv_reader:
                    row = [float(item) for item in row]
                    data.extend(row)
                    data_movie.append(row)
                csv_data_dict[background] = data
                csv_movie_wise_dict[background] = data_movie
    return csv_data_dict, csv_movie_wise_dict

'''
Helper method to create data structure appending data from each selected background from data_backgrounds_dict into
one list. 

Creates csv_data_dict, csv_movie_wise_dict, dictionaries of backgrounds mapped to lists of information.
csv_data_dict - dictionary mapping background to list of pooled data from all z-stacks
csv_movie_wise_dict - dictionary mapping background to list of lists, where outer lists represent each movie/z-stack
and each inner list represents the volumes from that movie-z-stack

parameters:

data_backgrounds_dict - dictionary mapping background to list of data (can be moviewise or pooled), created by  csv_files_to_dict
selected_backgrounds - backgrounds from data_backgrounds_dict that we want to include in selected_data

ex: "/Smriti/Experiments/in-vivo/csv/"
'''
def get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds):
    selected_data = []
    for background in selected_backgrounds:
        if background in data_backgrounds_dict:
            selected_data.append(data_backgrounds_dict[background])
        else:
            print(f"Warning: Background '{background}' not found in the data dictionary.")
            selected_data.append([])
    return selected_data


'''
Template method.
path is path to experiment folder with figs, csv, nd2...

'''
def sample_data_analysis(path):
    path_to_csv= path + "csv/"
    legend = ["Background 1", "Background 2",  "Background 3"]
    data_backgrounds_dict, csv_movie_wise_dict = csv_files_to_dict(path_to_csv)

    selected_backgrounds = ["back_1", "back_2", "back_3"]
    data_backgrounds = get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds)
    data_backgrounds_movie_wise = get_data_for_backgrounds(csv_movie_wise_dict, selected_backgrounds)

    calculate_p_and_t_values(data_backgrounds, legend, path_to_csv)
    make_mean_scatter(data_backgrounds_movie_wise, data_backgrounds, legend, "Mean Sample Graph", f'{path}/figs/', show = True)
    make_dist_histogram(data_backgrounds, legend, "Histogram Sample Graph", f'{path}/figs/', ranges = (0,500), show = True)
    make_med_or_var_bar(data_backgrounds, legend, "Median Sample Graph",  f'{path}/figs/', show = True)
    make_med_or_var_bar(data_backgrounds, legend, "Variance Sample Graph",  f'{path}/figs/', var = True, show = True)

'''
An example method! What I used to create my in vivo figures. 
'''
def in_vivo_display_figs_from_exp(path):
    path_to_csv= path + "csv/"
    legend = ["Male WT", "Male delPrLD",  "Female WT", "Female delPrLD"]
    data_backgrounds_dict, csv_movie_wise_dict = csv_files_to_dict(path_to_csv)
    selected_backgrounds = ["Male_Con", "Male_PrLD", "female_con", "Female_PrLD"]
    data_backgrounds = get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds)
    data_backgrounds_movie_wise = get_data_for_backgrounds(csv_movie_wise_dict, selected_backgrounds)
    calculate_p_and_t_values(data_backgrounds, legend, path_to_csv)
    make_mean_scatter(data_backgrounds_movie_wise, data_backgrounds, legend, "Mean of In Vivo Speckle Volumes with Movie-Wise Means", f'{path}/figs/', show = True)
    make_dist_histogram(data_backgrounds, legend, " Distribution of In Vivo Speckle Volumes", f'{path}/figs/', ranges = (0,500), show = True)
    # make_mean_bar(data_backgrounds, selected_backgrounds, "In Vitro Mean of  PrLD speckle Volumes",  f'{path}/figs/', show = True)
    make_med_or_var_bar( data_backgrounds, legend, "Median of In Vivo Speckle Volumes",  f'{path}/figs/', show = True) 
    make_med_or_var_bar( data_backgrounds, legend, "Variance of In Vivo Speckle Volumes",  f'{path}/figs/',  var = True, show = True) 

'''
an example method! What I used to create my in vitro figures. 
'''

def in_vitro_display_figs_from_exp(path, colors = ["darkgreen", "lightgreen"]):
    path_to_csv= path + "csv/"
    legend = ["Control (CLAMP WT)", "CLAMP delPRLD"]
    data_backgrounds_dict, csv_movie_wise_dict = csv_files_to_dict(path_to_csv)
    selected_backgrounds = ["CLAMP_WT", "CLAMP_delPRLD"]
    data_backgrounds = get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds)
    data_backgrounds_movie_wise = get_data_for_backgrounds(csv_movie_wise_dict, selected_backgrounds)
    calculate_p_and_t_values(data_backgrounds, legend, path_to_csv)
    make_mean_scatter(data_backgrounds_movie_wise, data_backgrounds, legend, "Mean of In Vitro Speckle Volumes with Movie-Wise Means", f'{path}/figs/', show = True, colors = colors)
    make_dist_histogram(data_backgrounds, legend, " Distribution of In Vitro Speckle Volumes", f'{path}/figs/', ranges = (0,500), show = True, colors = colors)
    # make_mean_bar(data_backgrounds, selected_backgrounds, "In Vitro Mean of  PrLD speckle Volumes",  f'{path}/figs/', show = True)
    make_med_or_var_bar( data_backgrounds, legend, "Median of In Vitro Speckle Volumes",  f'{path}/figs/', show = True, colors = colors) 
    make_med_or_var_bar( data_backgrounds, legend, "Variance of In Vitro Speckle Volumes",  f'{path}/figs/',  var = True, show = True, colors = colors) 

'''
CALL YOUR METHODS BELOW! Make sure to specify the path to your experiment. 
'''


#Example usage
# path_to_exp =  "/volumes/Research/BM_LarschanLab/Smriti/in_vitro/"
# in_vitro_display_figs_from_exp(path_to_exp)
# path_to_exp =  "/volumes/Research/BM_LarschanLab/Mukulika/Feb2024/"
# #in_vivo_display_figs_from_exp(path_to_exp)




