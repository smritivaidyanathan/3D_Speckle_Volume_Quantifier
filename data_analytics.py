import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from scipy import stats
import csv
import tkinter as tk
from tkinter import filedialog
import random
import math

def make_dist_histogram(data_backgrounds, legend, title, save_path, ylim = -1, ranges = None, show = False, bins=50):
    for i in range(len(data_backgrounds)):
        data = data_backgrounds[i]
        legend_label = legend[i]
        plt.hist(data, bins = bins, edgecolor='k', alpha=0.7, range=ranges, density = True, label = f"{legend_label} (N={len(data)})")
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

def make_mean_scatter(movie_wise_data_backgrounds, data_backgrounds, legend, title, save_path, show = False):
    means = [[np.mean(inner_list) for inner_list in sublist] for sublist in movie_wise_data_backgrounds]
    outer_means =[np.mean(data) for data in data_backgrounds]
    colors = ['blue', 'red', 'deepskyblue', 'magenta']
    legend_with_N = [f"{legend[i]} (N={len(data_backgrounds[i])})" for i in range(len(legend))]
    error = [1.96*(np.std(data)/(math.sqrt(len(data)))) for data in data_backgrounds]
    plt.bar(range(len(outer_means)), outer_means, yerr= error,color='gray', alpha=0.5, capsize=10)
    #confidence_intervals= [stats.t.interval(0.95, len(data)-1, loc=np.mean(data)) for data in data_backgrounds]
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

def make_med_or_var_bar(data_backgrounds, legend, title, save_path, var = False, show = False):
    quals = [np.median(data) for data in data_backgrounds]
    ylabel = "Median"
    if var:
        quals = [np.var(data) for data in data_backgrounds]
        ylabel = "Variance"

    legend_with_N = [f"{legend[i]} (N={len(data_backgrounds[i])})" for i in range(len(legend))]
    
    error = [bootstrapcli(data) for data in data_backgrounds]
    plt.bar(legend_with_N, height = quals, yerr=error, align = 'center', capsize=5, alpha=0.7)
    plt.xlabel('Experimental Conditions')
    plt.ylabel(f'{ylabel} of Speckle Size (pixels)')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()
    return quals

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

#ideally turn this into a GUI application
def display_figs_from_exp(path):
    path_to_csv= path + "/csv/"
    data_backgrounds_dict, csv_movie_wise_dict = csv_files_to_dict(path_to_csv)
    selected_backgrounds = ["Male_Con", "female_con", "Male_PrLD", "Female_PrLD"]
    data_backgrounds = get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds)
    data_backgrounds_movie_wise = get_data_for_backgrounds(csv_movie_wise_dict, selected_backgrounds)
    make_mean_scatter(data_backgrounds_movie_wise, data_backgrounds, ["Male WT", "Female WT", "Male delPrLD", "Female delPrLD"], "Mean of In Vitro Speckle Volumes with Movie-Wise Means", f'{path}/figs/', show = True)
    # make_dist_histogram(data_backgrounds, selected_backgrounds, "In Vitro Distribution of  PrLD speckle Volumes", f'{path}/figs/', ranges = (0,500), show = True)
    # make_mean_bar(data_backgrounds, selected_backgrounds, "In Vitro Mean of  PrLD speckle Volumes",  f'{path}/figs/', show = True)
    # make_med_or_var_bar(data_backgrounds, selected_backgrounds, "In Vitro Median of  PrLD speckle Volumes",  f'{path}/figs/', show = True) 
    # make_med_or_var_bar(data_backgrounds, selected_backgrounds, "In Vitro Variance of  PrLD speckle Volumes",  f'{path}/figs/',  var = False, show = True) 


path_to_exp = "/volumes/Research/BM_LarschanLab/Mukulika/Feb2024"

# root = tk.Tk()
# root.title("Figure Generator")
# browse_button = tk.Button(root, text="Browse for path to experiment", command=browse_folder )
# browse_button.pack()
# root.mainloop()

display_figs_from_exp(path_to_exp)

