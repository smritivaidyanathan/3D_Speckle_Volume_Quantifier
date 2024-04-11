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

def calculate_p_values(data, legend):
    p_values_table = np.zeros((len(data),len(data)))
    t_values_table = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(i+1, len(data)):  # Avoid redundant comparisons and the diagonal
            # Perform the statistical test (e.g., t-test)
            t_value, p_value = stats.ttest_ind(data[i], data[j])
            # Store the p-value in the table
            p_values_table[i, j] = p_value
            t_values_table[i, j] = t_value
            # print(f'{legend[i]} with a mean volume of {np.mean(data[i])} pixels and {legend[j]} pixels with a mean volume of {np.mean(data[j])} (p value={p_value}, t value={t_value})')
            # Since p-values are symmetric, store the same value in the symmetric position
            p_values_table[j, i] = p_value
            t_values_table[j, i] = t_value
    with open("p_and_t_values.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header row
        writer.writerow([''] + legend)
        
        # Write data rows
        for i, row in enumerate(p_values_table):
            writer.writerow([legend[i]] + list(row))

        writer.writerow('\n\n\n')

        writer.writerow([''] + legend)
        
        # Write data rows
        for i, row in enumerate(t_values_table):
            writer.writerow([legend[i]] + list(row))



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
    #plt.legend()
    plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()

    

# Show plot
plt.show()

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

def get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds):
    selected_data = []
    for background in selected_backgrounds:
        if background in data_backgrounds_dict:
            selected_data.append(data_backgrounds_dict[background])
        else:
            print(f"Warning: Background '{background}' not found in the data dictionary.")
            selected_data.append([])
    return selected_data


def in_vivo_display_figs_from_exp(path):
    path_to_csv= path + "csv/"
    legend = ["Male WT", "Male delPrLD",  "Female WT", "Female delPrLD"]
    data_backgrounds_dict, csv_movie_wise_dict = csv_files_to_dict(path_to_csv)
    selected_backgrounds = ["Male_Con", "Male_PrLD", "female_con", "Female_PrLD"]
    data_backgrounds = get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds)
    data_backgrounds_movie_wise = get_data_for_backgrounds(csv_movie_wise_dict, selected_backgrounds)
    calculate_p_values(data_backgrounds, legend)
    make_mean_scatter(data_backgrounds_movie_wise, data_backgrounds, legend, "Mean of In Vivo Speckle Volumes with Movie-Wise Means", f'{path}/figs/', show = True)
    make_dist_histogram(data_backgrounds, legend, " Distribution of In Vivo Speckle Volumes", f'{path}/figs/', ranges = (0,500), show = True)
    # make_mean_bar(data_backgrounds, selected_backgrounds, "In Vitro Mean of  PrLD speckle Volumes",  f'{path}/figs/', show = True)
    make_med_or_var_bar( data_backgrounds, legend, "Median of In Vivo Speckle Volumes",  f'{path}/figs/', show = True) 
    make_med_or_var_bar( data_backgrounds, legend, "Variance of In Vivo Speckle Volumes",  f'{path}/figs/',  var = True, show = True) 

def in_vitro_display_figs_from_exp(path, colors = ["darkgreen", "lightgreen"]):
    path_to_csv= path + "csv/"
    legend = ["Control (CLAMP WT)", "CLAMP delPRLD"]
    data_backgrounds_dict, csv_movie_wise_dict = csv_files_to_dict(path_to_csv)
    selected_backgrounds = ["CLAMP_WT", "CLAMP_delPRLD"]
    data_backgrounds = get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds)
    data_backgrounds_movie_wise = get_data_for_backgrounds(csv_movie_wise_dict, selected_backgrounds)
    calculate_p_values(data_backgrounds, legend)
    make_mean_scatter(data_backgrounds_movie_wise, data_backgrounds, legend, "Mean of In Vitro Speckle Volumes with Movie-Wise Means", f'{path}/figs/', show = True, colors = colors)
    make_dist_histogram(data_backgrounds, legend, " Distribution of In Vitro Speckle Volumes", f'{path}/figs/', ranges = (0,500), show = True, colors = colors)
    # make_mean_bar(data_backgrounds, selected_backgrounds, "In Vitro Mean of  PrLD speckle Volumes",  f'{path}/figs/', show = True)
    make_med_or_var_bar( data_backgrounds, legend, "Median of In Vitro Speckle Volumes",  f'{path}/figs/', show = True, colors = colors) 
    make_med_or_var_bar( data_backgrounds, legend, "Variance of In Vitro Speckle Volumes",  f'{path}/figs/',  var = True, show = True, colors = colors) 

path_to_exp =  "/volumes/Research/BM_LarschanLab/Smriti/in_vitro/"
in_vitro_display_figs_from_exp(path_to_exp)
path_to_exp =  "/volumes/Research/BM_LarschanLab/Mukulika/Feb2024/"
#in_vivo_display_figs_from_exp(path_to_exp)

