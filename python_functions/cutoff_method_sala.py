
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

def cutoff_analysis(data, data_name, venn_save_path, LH_column, distances=[75, 100, 150, 175, 200], plot_kde=False, lineplot_data=True, 
                    plot_scatter=True, plot_model=True):
    num_plots = sum([plot_kde, lineplot_data, plot_scatter, plot_model])
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 6 * num_plots))
    if num_plots == 1:
        axes = [axes]
    plot_idx = 0
    slope, intercept = None, None

    if plot_kde:
        LH_data = data[LH_column]
        sns.kdeplot(LH_data, color='black', label='Full L/H data', linewidth=2, ax=axes[plot_idx])
        axes[plot_idx].set_title('KDE of Full Protein L/H log2 Ratios')
        axes[plot_idx].set_ylabel('Density')
        axes[plot_idx].set_xlabel('log2(L/H)')
        axes[plot_idx].grid(True)
        axes[plot_idx].legend()
        plot_idx += 1

    if lineplot_data:
        LH_data = data[LH_column]
        sns.lineplot(data=LH_data, color='black', label='Full L/H data', linewidth=2, ax=axes[plot_idx])
        axes[plot_idx].set_title('Lineplot of Full Protein L/H log2 Ratios')
        axes[plot_idx].set_xlabel('Protein Rank Number')
        axes[plot_idx].set_ylabel('log2(L/H)')
        axes[plot_idx].grid(True)
        axes[plot_idx].legend()
        plot_idx += 1

    if plot_scatter:
        best_r_squared = -np.inf
        sorted_df = data.sort_values(by=LH_column).reset_index(drop=True)
        median_val = sorted_df[LH_column].median()
        ref_idx = (sorted_df[LH_column] - median_val).abs().idxmin()

        for distance in distances:
            start_idx = max(0, ref_idx - distance)
            end_idx = min(len(sorted_df), ref_idx + distance + 1)
            subset_data = sorted_df.iloc[start_idx:end_idx]
            x, y = subset_data.index, subset_data[LH_column]
            s, i = np.polyfit(x, y, 1)
            r2 = 1 - np.sum((y - (s * x + i))**2) / np.sum((y - np.mean(y))**2)
            if r2 > best_r_squared:
                best_r_squared, slope, intercept = r2, s, i
                best_subset = subset_data

        x = best_subset.index
        y = best_subset[LH_column]
        axes[plot_idx].scatter(x, y, color='blue', label='Subset around median')
        axes[plot_idx].plot(x, slope * x + intercept, color='red', label=f'Linear fit')
        axes[plot_idx].set_title(f'Scatterplot with best R² = {best_r_squared:.4f}')
        axes[plot_idx].set_xlabel('Protein Index')
        axes[plot_idx].set_ylabel('log2(L/H)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        plot_idx += 1

    if plot_model:
        LH_data_full = data[LH_column]
        ranks = np.arange(1, len(LH_data_full) + 1)
        model_line = slope * ranks + intercept
        stdev = np.std(model_line)
        upper_sd = model_line + stdev
        lower_sd = model_line - stdev

        axes[plot_idx].scatter(ranks, LH_data_full, color='blue', label='Experimental Data')
        axes[plot_idx].plot(ranks, model_line, color='red', label='Model')
        axes[plot_idx].plot(ranks, upper_sd, color='green', linestyle='--', label='Model + SD')
        axes[plot_idx].plot(ranks, lower_sd, color='green', linestyle='--', label='Model - SD')

        tol = 0.05 * np.std(LH_data_full)
        upper_idx = np.where(np.isclose(upper_sd, LH_data_full, atol=tol))[0]
        lower_idx = np.where(np.isclose(lower_sd, LH_data_full, atol=tol))[0]

        if len(upper_idx) > 0 and len(lower_idx) > 0:
            upper_val = upper_sd[upper_idx[0]]
            lower_val = lower_sd[lower_idx[0]]

            subset_aspecific = data[data[LH_column] >= upper_val]
            subset_specific = data[data[LH_column] <= lower_val]
            subset_undecided = data[(data[LH_column] > lower_val) & (data[LH_column] < upper_val)]

            plot_venn_cutoff(data_name, subset_specific, subset_undecided, subset_aspecific, venn_save_path = venn_save_path)

        axes[plot_idx].set_title('Distribution of L/H Ratios with Model and Cut-off')
        axes[plot_idx].set_xlabel('Protein Rank Number')
        axes[plot_idx].set_ylabel('log2(L/H)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)

    plt.tight_layout()
    plt.show()

    return subset_aspecific, subset_specific, subset_undecided

# Visualize the resulting S/A/U datasets using a Venn diagram
def plot_venn_cutoff(name, specific, undecided, aspecific, venn_save_path):
    specific_ids = set(specific['accession'])
    aspecific_ids = set(aspecific['accession'])
    undecided_ids = set(undecided['accession'])

    plt.figure(figsize=(8, 8))
    venn = venn3([specific_ids, aspecific_ids, undecided_ids], set_labels=('Specific', 'Aspecific', 'Undecided'))

    colors = {
        '100': 'pink', '010': 'skyblue', '001': 'lightgreen',
        '110': 'purple', '011': 'teal', '101': 'orange', '111': 'grey'
    }

    for patch_id, color in colors.items():
        patch = venn.get_patch_by_id(patch_id)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

    plt.title(f"Venn Diagram for {name}")
    os.makedirs(venn_save_path, exist_ok=True)
    venn_path = os.path.join(venn_save_path, f"{name}_venn_cutoff.png")
    plt.savefig(venn_path, bbox_inches='tight')
    print(f"Venn diagram saved at {venn_path}")
    plt.show()

# Analysis as used by Sala et al. : fixed distance of 75 (on each side of the median)
def cutoff_sala(base_path, LH_column, venn_save_path, distances=[75], sheet_name="proteins"):
    if venn_save_path is None:
        venn_save_path = os.path.join(base_path, "Venns")

    os.makedirs(venn_save_path, exist_ok=True)

    filenames = [f for f in os.listdir(base_path) if f.endswith('.xlsx')]
    results_dict = {}

    for fname in filenames:
        name = os.path.splitext(fname)[0]
        try:
            df = pd.read_excel(os.path.join(base_path, fname), sheet_name=sheet_name, engine="openpyxl")
            print(f"Processing: {name}")
            aspecific, specific, undecided = cutoff_analysis(
                df, LH_column = LH_column, data_name=name, distances=distances, 
                venn_save_path=venn_save_path
            )
            results_dict[name] = {
                "specific": specific,
                "aspecific": aspecific,
                "undecided": undecided
            }
        except Exception as e:
            print(f"Skipped {name} due to error: {e}")

    return results_dict

# Function to save the data dictionary from the cutoff_sala function into Excel files
def save_processed_results(processed_results, output_dir):
    import os
    import pandas as pd

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, datasets in processed_results.items():
        output_file = os.path.join(output_dir, f"{key}_Sala_method_results.xlsx")
        with pd.ExcelWriter(output_file) as writer:
            for dataset_name, df in datasets.items():
                sheet_name = dataset_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
