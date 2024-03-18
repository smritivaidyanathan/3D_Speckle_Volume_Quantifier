import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
from scipy import stats
import csv

def make_dist_histogram(data_backgrounds, legend, title, save_path, ylim = -1, range = None, show = False, bins=50):
    for i in range (len(data_backgrounds)):
        data = data_backgrounds[i]
        legend_label = legend[i]
        plt.hist(data, bins = bins, edgecolor='k', alpha=0.7, range=None, density = True, label = f"{legend_label} (N={len(data)})")
        plt.xlabel('Speckle Volume (pixels)')
        plt.ylabel('Count/Total Number')
    plt.title(f"{title}")
    plt.grid(True)
    if (ylim != -1):
        plt.ylim(top=ylim) 
    plt.legend()
    if (show):
        plt.show()
    plt.savefig(f'{save_path}{title}')

def make_mean_bar(data_backgrounds, legend, title, save_path,  show = False):
    means = [np.mean(data) for data in data_backgrounds]
    confidence_interval= [stats.t.interval(0.95, len(data)-1, loc=np.mean(data)) for data in data_backgrounds]
    yerr=[[abs(means[i] - confidence_interval[0][i])  for i in range(len(data_backgrounds))], [abs(means[i] - confidence_interval[1][i])  for i in range(len(data_backgrounds))]]
    plt.bar(legend, means, yerr=yerr, capsize=5, alpha=0.7)
    plt.xlabel('Experimental Conditions')
    plt.ylabel('Mean Speckle Size (pixels)')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()
    return means, confidence_interval

def make_med_or_var_bar(data_backgrounds, legend, title, save_path, var = False, show = False):
    quals = [np.median(data) for data in data_backgrounds]
    ylabel = "Median"
    if var:
        quals = [np.var(data) for data in data_backgrounds]
        ylabel = "Variance"

    confidence_interval = [bootstrapcli(data) for data in data_backgrounds]
    yerr = [[abs(quals[i] - confidence_interval[0][i])  for i in range(len(data_backgrounds))], [abs(quals[i] - confidence_interval[1][i])  for i in range(len(data_backgrounds))]]
    plt.bar(legend, quals, yerr=yerr, capsize=5, alpha=0.7)
    plt.xlabel('Experimental Conditions')
    plt.ylabel(f'{ylabel} of Speckle Size (pixels)')
    plt.title(f'{title}')
    plt.legend()
    plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()
    return quals, confidence_interval

def bootstrapcli(data , var = False) :
    num_bootstraps = 1000
    bootstrap_quals = []
    for _ in range(num_bootstraps):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        qual = np.median(bootstrap_sample)
        if var:
            qual = np.var(bootstrap_sample)
        bootstrap_quals.append(qual)
    confidence_interval = stats.t.interval(0.95, len(bootstrap_quals)-1, loc=np.mean(bootstrap_quals))
    return confidence_interval


path_to_exp = "/volumes/Research/BM_LarschanLab/Mukulika/Feb2024/"

def csv_files_to_dict(path_to_csv):
    csv_data_dict = {}
    for filename in os.listdir(path_to_csv):
        if filename.endswith(".csv"):
            background = filename.split('.')[0]
            file_path = os.path.join(path_to_csv, filename)
            with open(file_path, 'r', newline='') as csv_file:
                csv_reader = csv.reader(csv_file)
                data = []
                for row in csv_reader:
                    data.extend(row)
                csv_data_dict[background] = data
    return csv_data_dict

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
    path_to_csv= path_to_exp + "/csv/"
    data_backgrounds_dict = csv_files_to_dict(path_to_csv)
    selected_backgrounds = ["Female PrLD"]
    data_backgrounds = get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds)

    
