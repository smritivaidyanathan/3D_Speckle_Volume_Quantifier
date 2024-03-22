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
    confidence_interval= [stats.t.interval(0.95, len(data)-1, loc=np.mean(data)) for data in data_backgrounds]
    print(confidence_interval)
    print(means)
    yerr=[[abs(means[i] - confidence_interval[i][0])  for i in range(len(data_backgrounds))], [abs(means[i] - confidence_interval[i][1])  for i in range(len(data_backgrounds))]]
    print(yerr)
    plt.bar(legend_with_N, means, yerr=yerr,align = 'center', capsize=5, alpha=0.7)
    plt.xlabel('Experimental Conditions')
    plt.ylabel('Mean Speckle Size (pixels)')
    plt.title(f'{title}')
    plt.savefig(f'{save_path}{title}')
    if (show):
        plt.show()
    return means, confidence_interval

def make_mean_scatter(data_backgrounds, legend, title, save_path, show = False):
    means = data_backgrounds.mean(axis=1)
    colors = ['red', 'green', 'blue', 'green']
    #confidence_intervals= [stats.t.interval(0.95, len(data)-1, loc=np.mean(data)) for data in data_backgrounds]
    for i, column_data in enumerate(means):
        plt.scatter([i] * len(column_data), column_data, color=colors[i], label=f'{legend[i]}')
    plt.xlabel('Column')
    plt.ylabel('Value')
    plt.title(f'{title}')
    plt.legend()
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
    confidence_interval = [bootstrapcli(data) for data in data_backgrounds]
    yerr = [[abs(quals[i] - confidence_interval[i][0])  for i in range(len(data_backgrounds))], [abs(quals[i] - confidence_interval[i][1])  for i in range(len(data_backgrounds))]]
    plt.bar(legend_with_N, height = quals, yerr=yerr, align = 'center', capsize=5, alpha=0.7)
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
                    row = [float(item) for item in row]
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
    path_to_csv= path + "/csv/"
    data_backgrounds_dict = csv_files_to_dict(path_to_csv)
    selected_backgrounds = ["Male_Con", "female_con"]
    data_backgrounds = get_data_for_backgrounds(data_backgrounds_dict, selected_backgrounds)
    make_dist_histogram(data_backgrounds, selected_backgrounds, "In Vitro Distribution of  PrLD speckle Volumes", f'{path}/figs/', ranges = (0,500), show = True)
    make_mean_bar(data_backgrounds, selected_backgrounds, "In Vitro Mean of  PrLD speckle Volumes",  f'{path}/figs/', show = True)
    make_med_or_var_bar(data_backgrounds, selected_backgrounds, "In Vitro Median of  PrLD speckle Volumes",  f'{path}/figs/', show = True) 
    make_med_or_var_bar(data_backgrounds, selected_backgrounds, "In Vitro Variance of  PrLD speckle Volumes",  f'{path}/figs/',  var = False, show = True) 


path_to_exp = "/volumes/Research/BM_LarschanLab/Mukulika/Feb2024/"

def browse_folder():
    print("hi")
    folder_path = filedialog.askdirectory()
    if folder_path:
        root.update()
        root.destroy()
        display_figs_from_exp(folder_path)


# root = tk.Tk()
# root.title("Figure Generator")
# browse_button = tk.Button(root, text="Browse for path to experiment", command=browse_folder )
# browse_button.pack()
# root.mainloop()

display_figs_from_exp(path_to_exp)

