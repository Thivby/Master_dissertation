import os
import time
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from matplotlib.patches import Patch


def plot_venn_cutoff(name, below, undecided, above, 
                     save_path="G:/My Drive/Uni/Thesis/Data/datasets_cutoff/venn_cutoffs"):
    """
    Creates a customized Venn diagram to show overlaps between three sets (specific, aspecific, undecided) with a legend.

    Parameters:
    - name: Name of the dataset for the Venn diagram title and the file name.
    - below: DataFrame containing the 'Specific' entries.
    - undecided: DataFrame containing the 'Undecided' entries.
    - above: DataFrame containing the 'Aspecific' entries.
    - save_path: Path to save the Venn diagram.
    """
    below_ids = set(below['accession'])
    above_ids = set(above['accession'])
    undecided_ids = set(undecided['accession'])

    plt.figure(figsize=(8, 8))
    venn = venn3([below_ids, above_ids, undecided_ids], set_labels=('', '', ''))  # No text labels

    # Regions with both color and label (to show in legend)
    labeled_regions = {
        '100': ('pink', 'Specific'),
        '010': ('skyblue', 'Aspecific'),
        '001': ('lightgreen', 'Undecided'),
    }

    # Regions with color only (no label shown in legend)
    unlabeled_regions = {
        '110': 'plum',
        '011': 'lightcyan',
        '101': 'peachpuff',
        '111': 'grey'
    }

    legend_elements = []

    # Apply styling and collect legend entries for labeled regions
    for region, (color, label) in labeled_regions.items():
        patch = venn.get_patch_by_id(region)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

    # Apply styling only (no legend entry) for unlabeled regions
    for region, color in unlabeled_regions.items():
        patch = venn.get_patch_by_id(region)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

    # Show legend
    plt.legend(handles=legend_elements, title='Legend', loc='upper right', bbox_to_anchor=(1.3, 1))

    # Title and save
    plt.title("Customized Overlap between Specific, Aspecific and Undecided Entries of " + name)
    venn_save_path = os.path.join(save_path, name + "_venn_cutoff.png")
    plt.savefig(venn_save_path, bbox_inches='tight')
    print(f"Venn diagram saved at {venn_save_path}")
    plt.show()


# Combine the above cutoff, below cutoff and undecided fucntions into a single function together with the visualisation in Venn diagrams.
def cutoff_analysis(data, data_name, median_idx=None, distances=[75, 100, 150, 175, 200], plot_kde=False, lineplot_data=True, 
                    plot_scatter=True, plot_model=True, save_path="G:/My Drive/Uni/Thesis/Data/datasets_cutoff"):
    """
    Generates multiple plots (KDE, lineplot, scatterplot, and model plot with SD) in a single figure,
    and saves subsets of the data above, below, and between the cut-off values.

    Parameters:
    - data: DataFrame containing the protein data with a 'Median(log2(q.ratio))' column.
    - median_idx: Optional index to specify the median or a reference point.
    - distance: Number of proteins to include above and below the median (default 75 on each side).
    - plot_kde: If True, generate the KDE plot of the full data.
    - lineplot_data: If True, generate the lineplot of the full data.
    - plot_scatter: If True, generate the scatterplot of the subset around the median with a regression line.
    - plot_model: If True, generate the model plot with extrapolated data, standard deviation lines, and cut-off points.
    - data_name: String to represent the name of the dataset when saving the subset.
    - save_path: Directory path where the output CSV file should be saved.
    """
    # Ensure save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Determine how many plots to include
    num_plots = sum([plot_kde, lineplot_data, plot_scatter, plot_model])

    # Create a figure and subplots based on the number of plots
    fig, axes = plt.subplots(num_plots, 1, figsize=(8, 6 * num_plots))
    
    # Adjust axes handling for single/multiple plots
    if num_plots == 1:
        axes = [axes]  # Ensure it's iterable for a single plot
    
    plot_idx = 0
    slope, intercept = None, None  # Initialize slope and intercept to be used in both parts

    if plot_kde:
        # KDE Plot of the Full Data
        LH_data = data['Median(log2(q.ratio))']
        sns.kdeplot(LH_data, color='black', label='Full L/H data', linewidth=2, ax=axes[plot_idx])
        axes[plot_idx].set_title('KDE of Full Protein L/H log2 Ratios')
        axes[plot_idx].set_ylabel('Density')
        axes[plot_idx].set_xlabel('log2(L/H)')
        axes[plot_idx].grid(True)
        axes[plot_idx].legend()
        plot_idx += 1
    
    if lineplot_data:
        # Lineplot of the Full Data
        LH_data = data['Median(log2(q.ratio))']
        sns.lineplot(data=LH_data, color='black', label='Full L/H data', linewidth=2, ax=axes[plot_idx])
        axes[plot_idx].set_title('Lineplot of Full Protein L/H log2 Ratios')
        axes[plot_idx].set_xlabel('Protein Rank Number')
        axes[plot_idx].set_ylabel('log2(L/H)')
        axes[plot_idx].grid(True)
        axes[plot_idx].legend()
        plot_idx += 1

    if plot_scatter:
        # Scatterplot of the Subset
        best_distance = None
        best_r_squared = -np.inf
        best_slope, best_intercept = None, None
        ref_idx = None

        # Sort the DataFrame by 'Median(log2(q.ratio))'
        sorted_df = data.sort_values(by='Median(log2(q.ratio))').reset_index(drop=True)

        if median_idx is None:
            median_protein_value = sorted_df['Median(log2(q.ratio))'].median()
            ref_idx = (sorted_df['Median(log2(q.ratio))'] - median_protein_value).abs().idxmin()

        # Iterate over the possible distances
        for distance in distances:
            start_idx = max(0, ref_idx - distance)
            end_idx = min(len(sorted_df), ref_idx + distance + 1)
            
            subset_data = sorted_df.iloc[start_idx:end_idx]
            x = subset_data.index
            y = subset_data['Median(log2(q.ratio))']

            # Perform linear regression
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept

            # Calculate R² (goodness of fit)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Update best fit based on R² value
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_distance = distance
                best_slope, best_intercept = slope, intercept

        # Update the chosen distance and slope/intercept for plotting
        distance = best_distance
        slope, intercept = best_slope, best_intercept
        print(f"Best distance selected: {distance} with R² = {best_r_squared:.4f}")
        
        # Select subset of proteins
        start_idx = max(0, ref_idx - distance)
        end_idx = min(len(sorted_df), ref_idx + distance + 1)
        subset_data = sorted_df.iloc[start_idx:end_idx]
        
        x = subset_data.index
        y = subset_data['Median(log2(q.ratio))']
        
        # Scatterplot and linear fit
        axes[plot_idx].scatter(x, y, color='blue', label='Subset around median')
        axes[plot_idx].plot(x, slope * x + intercept, color='red', label=f'Linear fit: y={slope:.5f}x + {intercept:.5f}')
        
        axes[plot_idx].set_title(f'Scatterplot of Subset Around Median: Linear Fit with distance {best_distance} and R² = {best_r_squared:.4f}')
        axes[plot_idx].set_xlabel('Protein Index')
        axes[plot_idx].set_ylabel('log2(L/H)')
        axes[plot_idx].grid(True)
        axes[plot_idx].legend()
        plot_idx += 1

    if plot_model:
        # Ensure slope and intercept are available
        if slope is None or intercept is None:
            protein_rank_full = np.arange(1, len(data) + 1)
            LH_data_full = data['Median(log2(q.ratio))']
            slope, intercept = np.polyfit(protein_rank_full, LH_data_full, 1)

        LH_data_full = data['Median(log2(q.ratio))']
        protein_rank_full = np.arange(1, len(LH_data_full) + 1)
        dynamic_tol = 0.05 * np.std(LH_data_full)
        
        extrapolated_model_data = slope * protein_rank_full + intercept
        st_dev = np.std(extrapolated_model_data)
        
        model_line = slope * protein_rank_full + intercept
        upper_sd_line = model_line + st_dev
        lower_sd_line = model_line - st_dev

        axes[plot_idx].scatter(protein_rank_full, LH_data_full, color='blue', label='Experimental Data')
        axes[plot_idx].plot(protein_rank_full, model_line, color='red', label='Model Data')
        axes[plot_idx].plot(protein_rank_full, upper_sd_line, color='green', linestyle='--', label='Model + SD')
        axes[plot_idx].plot(protein_rank_full, lower_sd_line, color='green', linestyle='--', label='Model - SD')

        upper_cut_off_index = np.where(np.isclose(upper_sd_line, LH_data_full, atol=dynamic_tol))[0]
        lower_cut_off_index = np.where(np.isclose(lower_sd_line, LH_data_full, atol=dynamic_tol))[0]

        if lower_cut_off_index.size > 0 and upper_cut_off_index.size > 0:
            lower_cut_off_x = lower_cut_off_index[0]
            upper_cut_off_x = upper_cut_off_index[0]

            upper_cut_off_value = upper_sd_line[upper_cut_off_x]
            axes[plot_idx].scatter(upper_cut_off_x, upper_cut_off_value, color='yellow', zorder=5, label='Above Cut-off Point')
            subset_above_cutoff = data[data['Median(log2(q.ratio))'] >= upper_cut_off_value]

            lower_cut_off_value = lower_sd_line[lower_cut_off_x]
            axes[plot_idx].scatter(lower_cut_off_x, lower_cut_off_value, color='cyan', zorder=5, label='Lower Cut-off Point')
            subset_below_cutoff = data[data['Median(log2(q.ratio))'] <= lower_cut_off_value]
            
            subset_undecided = data[(data['Median(log2(q.ratio))'] > lower_cut_off_value) &
                                    (data['Median(log2(q.ratio))'] < upper_cut_off_value)]

        axes[plot_idx].set_title('Distribution of L/H Ratios with Model and Cut-off')
        axes[plot_idx].set_xlabel('Protein Rank Number')
        axes[plot_idx].set_ylabel('log2(L/H)')
        axes[plot_idx].axhline(0, color='black', lw=0.5, ls='--')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
    
    plt.tight_layout()
    plt.show()

    # Plot and save Venn diagram for the subsets
    plot_venn_cutoff(data_name, subset_below_cutoff, subset_undecided, subset_above_cutoff, save_path="G:/My Drive/Uni/Thesis/Data/datasets_cutoff/venn_cutoffs")

    return subset_above_cutoff, subset_below_cutoff, subset_undecided


# Function 1: fetch the interactions for the above or below dataset.

def fetch_biogrid_interactions(input_dataset, acces_key="2c48a1ace1f16d534f59f559323a8eb0", tax_id=9606):
    """
    Fetch interactions from BioGRID for genes in the input dataset.
    
    Parameters:
    input_dataset (DataFrame): Input dataset containing a column 'accession' with gene names. Typically this is the dataset with genes above or below the cutoff.
    acces_key (str): BioGRID API access key. This can be freely requested from BioGRID (default is my own key).
    tax_id (int): Taxonomy ID for species (default is 9606 for human).
    
    Returns:
    DataFrame: Dataframe containing BioGRID interactions of all the proteins in the input dataset.
    """
    base_url = "https://webservice.thebiogrid.org"
    request_url = base_url + "/interactions"

    # List of genes to search for
    gene_list = input_dataset["accession"].tolist()
    evidence_list = ["POSITIVE GENETIC", "PHENOTYPIC ENHANCEMENT", "Reconstituted Complex"]

    all_results = pd.DataFrame()
    start = 0
    max_results = 10000

    while True:
        # Parameters for BioGRID API
        params = {
            "accesskey": acces_key,
            "format": "json",
            "geneList": "|".join(gene_list),
            "searchNames": "true",
            "includeInteractors": "true",
            "taxId": tax_id,
            "evidenceList": "|".join(evidence_list),
            "includeEvidence": "false",
            "includeHeader": "true",
            "interSpeciesExcluded": "false",
            "additionalIdentifierTypes": "UNIPROT|UNIPROTKB",
            "start": start,
            "max": max_results
        }

        # Send request to BioGRID API
        response = requests.get(request_url, params=params)
        response.raise_for_status()
        interactions = response.json()

        # Create a dictionary of results by interaction identifier
        data = {}
        for interaction_id, interaction in interactions.items():
            data[interaction_id] = interaction
            data[interaction_id]["INTERACTION_ID"] = interaction_id

        # Load the data into a pandas dataframe
        dataset = pd.DataFrame.from_dict(data, orient="index")
        all_results = pd.concat([all_results, dataset], ignore_index=True)

        # Check if we received fewer results than the maximum, indicating the end
        if len(dataset) < max_results:
            break

        # Update the start parameter for the next batch
        start += max_results

    # Select only the desired columns
    columns = [
        "INTERACTION_ID",
        "ENTREZ_GENE_A",
        "ENTREZ_GENE_B",
        "OFFICIAL_SYMBOL_A",
        "OFFICIAL_SYMBOL_B",
        "EXPERIMENTAL_SYSTEM",
        "PUBMED_ID",
        "PUBMED_AUTHOR",
        "THROUGHPUT",
        "QUALIFICATIONS",
    ]
    interactor_dataset = all_results[columns]

    return interactor_dataset

# Function 2: map UniProt accession codes to Entrez Gene IDs using UniProt API for the undecided dataset. (Such that it can be compared to the interactor datasets which use entrez as ID)

def map_uniprot_to_entrez(undecided_dataset, batch_size=10):
    """
    Map UniProt accession codes to Entrez Gene IDs using UniProt API.

    Parameters:
    undecided_dataset (DataFrame): Input dataset containing a column 'accession' with UniProt IDs.
    acces_key (str): UniProt API access key.
    batch_size (int): Size of batches for submitting ID mapping jobs.

    Returns:
    DataFrame: Updated dataset with an added column 'entrez_gene_id'.
    """
    def submit_id_mapping(uniprot_ids):
        url = "https://rest.uniprot.org/idmapping/run"
        payload = {
            'from': 'UniProtKB_AC-ID',
            'to': 'Gene_Name',
            'ids': ','.join(uniprot_ids)
        }
        response = requests.post(url, data=payload)
        response.raise_for_status()
        result = response.json()

        if "jobId" in result:
            return result["jobId"]
        elif "results" in result:
            return pd.DataFrame(result['results'])
        raise Exception(f"Unexpected response format: {result}")

    def get_mapping_results(job_id):
        results_url = f"https://rest.uniprot.org/idmapping/results/{job_id}"
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            response = requests.get(results_url)
            if response.status_code == 404:
                print(f"Results not ready yet, retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(5)
                continue
            response.raise_for_status()
            results = response.json()
            if "results" in results:
                return pd.DataFrame(results['results'])
            elif "jobStatus" in results and results["jobStatus"] == "FINISHED":
                retry_count += 1
                time.sleep(5)
            elif "jobStatus" in results and results["jobStatus"] == "RUNNING":
                print(f"Job is still running, retrying... ({retry_count + 1}/{max_retries})")
                retry_count += 1
                time.sleep(5)
        raise Exception(f"Job did not complete after {max_retries} retries")

    uniprot_ids = undecided_dataset['accession'].unique()
    total_batches = (len(uniprot_ids) + batch_size - 1) // batch_size
    print(f"Total batches to process: {total_batches}")

    all_results = pd.DataFrame()

    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1} of {total_batches}: {len(batch)} IDs")
        job_id = submit_id_mapping(batch)
        try:
            batch_results = get_mapping_results(job_id)
            all_results = pd.concat([all_results, batch_results], ignore_index=True)
        except Exception as e:
            print(f"Failed to process batch {i//batch_size + 1}: {e}")

    undecided = undecided_dataset.merge(all_results, left_on='accession', right_on='from', how='left')
    undecided.rename(columns={'to': 'entrez_gene_id'}, inplace=True)

    return undecided

## Venn function for overlap between interactors and uncertain fractions
def plot_venn_interactors_overlap(name, interactors, undecided_entrez,
                                   save_path="G:/My Drive/Uni/Thesis/Data/datasets_cutoff/venn_interactors"):
    """
    Creates a customized Venn diagram to show overlaps between the interactors of either above or below datasets 
    with the uncertain dataset, using a color-coded legend.

    Parameters:
    - name: Name of the dataset for the Venn diagram title and the file name.
    - interactors: DataFrame containing the interactors with 'OFFICIAL_SYMBOL_A' and 'OFFICIAL_SYMBOL_B' columns.
    - undecided_entrez: DataFrame containing undecided entries with 'entrez_gene_id' column.
    - save_path: Path to save the Venn diagram.
    """
    # Combine interactors from both columns into a single set
    interactors_ids_A = set(interactors['OFFICIAL_SYMBOL_A'])
    interactors_ids_B = set(interactors['OFFICIAL_SYMBOL_B'])
    interactors_ids = interactors_ids_A.union(interactors_ids_B)

    undecided_ids = set(undecided_entrez['entrez_gene_id'])

    plt.figure(figsize=(8, 8))
    venn = venn2([interactors_ids, undecided_ids], set_labels=('', ''))  # Remove default labels

    # Regions to include in the legend
    labeled_regions = {
        '10': ('lightcoral', 'Interactors'),
        '01': ('lightgreen', 'Undecided')
    }

    # Regions to style but not label in the legend
    unlabeled_regions = {
        '11': 'mistyrose'
    }

    legend_elements = []

    # Style labeled regions and collect legend entries
    for region_id, (color, label) in labeled_regions.items():
        patch = venn.get_patch_by_id(region_id)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

    # Style unlabeled regions only
    for region_id, color in unlabeled_regions.items():
        patch = venn.get_patch_by_id(region_id)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

    # Add legend
    plt.legend(handles=legend_elements, title='Overlap Regions', loc='upper right', bbox_to_anchor=(1.25, 1))

    # Add title
    plt.title(f"Customized Overlap between Interactors and Undecided Entries of {name}")

    # Save the plot
    venn_save_path = os.path.join(save_path, name + "_venn_interactors.png")
    plt.savefig(venn_save_path, bbox_inches='tight')
    print(f"Venn diagram saved at {venn_save_path}")
    plt.show()



# Funcion 3: find matching entries between the undecided dataset and the interactor dataset of above or below and add matches to these sets.

def find_matching_entries(data_name, data, undecided, interactor_dataset):
    """
    Find matching entries between undecided and interactor dataset of 'above' or 'below' dataset and add them to the appropriate dataset.

    Parameters:
    Data (DataFrame): Dataset containing initial protein data from either the 'above' or the 'below' subset.
    Undecided (DataFrame): Dataset with UniProt accessions mapped to Entrez Gene IDs.
    interactor_dataset (DataFrame): Dataset containing interaction data with columns 'OFFICIAL_SYMBOL_A' and 'OFFICIAL_SYMBOL_B'.

    Returns:
    DataFrame: Updated lmcd1_above dataset with matching entries appended.
    Venn diagram showing the overlap between the interactors and undecided entries.
    """
    plot_venn_interactors_overlap(data_name, interactor_dataset, undecided)

    # Extract unique Entrez gene IDs from undecided dataset
    undecided_ids = set(undecided['entrez_gene_id'])

    # Extract interactors as a set of unique IDs
    interactors_ids_A = set(interactor_dataset['OFFICIAL_SYMBOL_A'])
    interactors_ids_B = set(interactor_dataset['OFFICIAL_SYMBOL_B'])
    interactors_ids = interactors_ids_A.union(interactors_ids_B)

    # Find the intersection between undecided IDs and interactors
    matching_ids = undecided_ids.intersection(interactors_ids)

    # Filter undecided rows that have matching IDs in interactors
    matching_entries = undecided[undecided['entrez_gene_id'].isin(matching_ids)]

    # Concatenate matching entries to the updated data
    updated_data = pd.concat([data, matching_entries], ignore_index=True)

    return updated_data

# Function 3.5: find the remaining undecided data.
def find_remaining_undecided(full_data, above_updated, below_updated):
    """
    Finds the remaining undecided entries in the full dataset that are not present in either the
    above_updated or below_updated datasets.

    Parameters:
    - full_data: DataFrame containing the full dataset.
    - above_updated: DataFrame containing the 'Above' entries.
    - below_updated: DataFrame containing the 'Below' entries.

    Returns:
    - DataFrame containing the remaining undecided entries.
    """
    # Combine 'above_updated' and 'below_updated' to create a set of entries to exclude
    combined_above_below = pd.concat([above_updated, below_updated])
    
    # Find the remaining undecided entries in 'full_data' that are not in 'combined_above_below'
    remaining_undecided = full_data[~full_data['accession'].isin(combined_above_below['accession'])]
    
    return remaining_undecided


## Venn function for overlap between updated fractions of above, below and undecided datasets. (same as plot_venn_cutoff but with updated datasets)
def plot_updated_venn_cutoff(name, below_updated, undecided_updated, above_updated,
                              save_path="G:/My Drive/Uni/Thesis/Data/datasets_cutoff/updated_venn_cutoffs"):
    """
    Creates a customized Venn diagram to show overlaps between three sets with a legend for main categories only.

    Parameters:
    - name: Name of the dataset for the Venn diagram title and the file name.
    - below_updated: DataFrame containing the updated 'Below' entries.
    - undecided_updated: DataFrame containing the updated 'Undecided' entries.
    - above_updated: DataFrame containing the updated 'Above' entries.
    - save_path: Path to save the Venn diagram.
    """
    # Convert 'accession' column to sets
    below_ids = set(below_updated['accession'])
    above_ids = set(above_updated['accession'])
    undecided_ids = set(undecided_updated['accession'])

    plt.figure(figsize=(8, 8))
    venn = venn3([below_ids, above_ids, undecided_ids], set_labels=('', '', ''))  # No set labels

    # Regions to include in the legend
    labeled_patches = {
        '100': ('pink', 'Specific Updated'),
        '010': ('skyblue', 'Aspecific Updated'),
        '001': ('lightgreen', 'Undecided Updated')
    }

    # Regions to color but NOT include in the legend
    unlabeled_patches = {
        '110': 'plum',
        '011': 'lightcyan',
        '101': 'peachpuff',
        '111': 'grey'
    }

    legend_elements = []

    # Style labeled patches and collect legend entries
    for patch_id, (color, label) in labeled_patches.items():
        patch = venn.get_patch_by_id(patch_id)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label))

    # Style unlabeled patches (no legend entry)
    for patch_id, color in unlabeled_patches.items():
        patch = venn.get_patch_by_id(patch_id)
        if patch:
            patch.set_color(color)
            patch.set_alpha(0.5)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

    # Add the legend
    plt.legend(handles=legend_elements, title='Overlap Regions', loc='upper right', bbox_to_anchor=(1.3, 1))

    # Add title
    plt.title("Overlap between the updated Specific, Aspecific, and Undecided Entries of " + name)

    # Save the plot
    venn_save_path = os.path.join(save_path, name + "_updated_venn_cutoff.png")
    plt.savefig(venn_save_path, bbox_inches='tight')
    print(f"Venn diagram saved at {venn_save_path}")
    plt.show()



def find_unique_entries(name, below_updated, above_updated, undecided_updated):
    """
    Find unique entries between below_updated, above_updated, and undecided_updated,
    and display a Venn diagram showing the overlap between the three datasets before and after the unique entries are found.

    Parameters:
    - below_updated (DataFrame): Dataset containing entries classified as below.
    - above_updated (DataFrame): Dataset containing entries classified as above.
    - undecided_updated (DataFrame): Dataset containing undecided entries with Entrez Gene IDs.

    Returns:
    - Tuple[DataFrame, DataFrame, DataFrame, DataFrame]: Four DataFrames containing unique entries:
        - Rows in below_updated but not in above_updated.
        - Rows in above_updated but not in below_updated.
        - Rows in undecided_updated that are not in either below_updated or above_updated.
    """
    # Plot the overlap between the three datasets using a Venn diagram before finding unique entries
    plot_updated_venn_cutoff(name+'_Initial_Overlap', below_updated, undecided_updated, above_updated)
    
    # Extract unique accessions from below and above datasets
    below_accessions = set(below_updated['accession'])
    above_accessions = set(above_updated['accession'])

    # Find unique entries
    below_not_in_above = below_updated[~below_updated['accession'].isin(above_accessions)]
    above_not_in_below = above_updated[~above_updated['accession'].isin(below_accessions)]

    # Extract unique accessions from the updated below and above datasets
    unique_accessions = below_accessions.union(above_accessions)

    # Find undecided entries that are not in either below or above
    undecided_not_in_below_or_above = undecided_updated[~undecided_updated['accession'].isin(unique_accessions)]

    # Find entries that are in both 'below_updated' and 'above_updated'
    common_entries = below_updated[below_updated['accession'].isin(above_accessions)]

    # Add these entries to the undecided dataset, ensuring no duplicates
    undecided_updated_2 = pd.concat([undecided_not_in_below_or_above, common_entries], ignore_index=True).drop_duplicates(subset='accession')

    # Plot the overlap between the three datasets using a Venn diagram after finding unique entries
    plot_updated_venn_cutoff(name+'_Final_Overlap', below_not_in_above, undecided_updated_2, above_not_in_below)
    
    return below_not_in_above, above_not_in_below, undecided_updated_2

# Show the difference between the kde plots made with below_updated above updated and undecided_updated datasets and the kde plot of the full dataset.

def plot_kde_datasets(name, full_data, below_unique, above_unique, undecided_unique, save_path="G:/My Drive/Uni/Thesis/Data/datasets_cutoff/kde_plots"):
    """
    Creates and saves KDE plots for the provided datasets to visualize the distributions of 'Median(log2(q.ratio))'.

    Parameters:
    - full_data (DataFrame): Full dataset containing 'Median(log2(q.ratio))'.
    - below_unique (DataFrame): DataFrame containing the unique 'Below' entries.
    - above_unique (DataFrame): DataFrame containing the unique 'Above' entries.
    - undecided_unique (DataFrame): DataFrame containing the undecided entries.
    - name (str): Name of the dataset to use in plot titles and filenames.
    - save_path (str): Path to save the KDE plots.
    """
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Create KDE plot for the full dataset
    plt.figure(figsize=(12, 6))
    sns.kdeplot(full_data['Median(log2(q.ratio))'], label=f'{name} Full', fill=True)
    plt.title(f'KDE Plot of Median(log2(q.ratio)) for {name} Full Dataset')
    plt.xlabel('Median(log2(q.ratio))')
    plt.ylabel('Density')
    plt.legend()
    
    # Save the full dataset KDE plot
    full_save_path = os.path.join(save_path, f"{name}_full_kde_plot.png")
    plt.savefig(full_save_path)
    plt.show()

    # Create KDE plots for the below, above, and undecided unique datasets
    plt.figure(figsize=(12, 6))
    sns.kdeplot(below_unique['Median(log2(q.ratio))'], label='Specific', fill=True, color='#FFC0CB')
    sns.kdeplot(above_unique['Median(log2(q.ratio))'], label='Aspecific', fill=True, color='#87CEEB')
    sns.kdeplot(undecided_unique['Median(log2(q.ratio))'], label='Undecided', fill=True, color='#90EE90')
    plt.title(f'KDE Plot of Median(log2(q.ratio)) for Specific, Aspecific, and Undecided Unique Entries in {name}')
    plt.xlabel('Median(log2(q.ratio))')
    plt.ylabel('Density')
    plt.legend()

    # Save the KDE plot for below, above, and undecided
    unique_save_path = os.path.join(save_path, f"{name}_separated_kde_plot.png")
    plt.savefig(unique_save_path, bbox_inches='tight')
    plt.show()

    
#Function 5: combination of the previous 4 functions to fetch the interactions for the above or below dataset and find unique entries between the above, below, and undecided datasets.
def combined_function(name, full_data, subset_below_cutoff, subset_above_cutoff, subset_undecided):
    """
    Combined function to process protein data subsets and return unique entries.
    
    Parameters:
    subset_below_cutoff (DataFrame): Dataset containing entries classified as below the cutoff.
    subset_above_cutoff (DataFrame): Dataset containing entries classified as above the cutoff.
    subset_undecided (DataFrame): Dataset containing undecided entries with UniProt accession codes.
    
    Returns:
    Tuple[DataFrame, DataFrame, DataFrame]: Three DataFrames containing unique entries:
        - below_not_in_above: Rows in below but not in above.
        - above_not_in_below: Rows in above but not in below.
        - undecided_not_in_below_or_above: Rows in undecided that are not in either below or above.
    """
    # Step 1: Fetch BioGRID interactions for below and above datasets
    below_interactions = fetch_biogrid_interactions(subset_below_cutoff)
    above_interactions = fetch_biogrid_interactions(subset_above_cutoff)

    # Step 2: Map UniProt accession codes to Entrez Gene IDs using UniProt API for the undecided dataset
    undecided_entrez = map_uniprot_to_entrez(subset_undecided)

    # Step 3: Find matching entries and update below, above and undecided datasets
    name1 = name +'_below'
    below_updated = find_matching_entries(name1, subset_below_cutoff, undecided_entrez, below_interactions).drop_duplicates(subset='accession')
    name2 = name +'_above'
    above_updated = find_matching_entries(name2, subset_above_cutoff, undecided_entrez, above_interactions).drop_duplicates(subset='accession')
    name3 = name +'_undecided'
    undecided_updated = find_remaining_undecided(full_data, above_updated, below_updated).drop_duplicates(subset='accession')

    # Step 4: Find unique entries across updated datasets
    below_updated, above_updated, undecided_updated = find_unique_entries(name, below_updated, above_updated, undecided_updated)

    # Step 5: Plot KDE plots for the full dataset and unique entries
    plot_kde_datasets(name, full_data, below_updated, above_updated, undecided_updated)

    return below_updated, above_updated, undecided_updated



# Creating a function to apply `combined_function` to each entry in `results_dict` and store the result in a similar dictionary structure
def process_results_dict(results_dict, save_path="G:/My Drive/Uni/Thesis/Data/datasets_cutoff"):
    """
    Apply combined_function to each entry in results_dict and return a new dictionary with the results.

    Parameters:
    results_dict (dict): A dictionary where each key is associated with a dictionary containing
                         "subset_below_cutoff", "subset_above_cutoff", and "subset_undecided" DataFrames.
    save_path (str): Path where plots and results should be saved.

    Returns:
    dict: A new dictionary where each key is associated with a dictionary containing the updated DataFrames
          "below_not_in_above", "above_not_in_below", and "undecided_not_in_below_or_above".
    """
    processed_results = {}

    for name, subsets in results_dict.items():
        # Extract subsets
        full_data = subsets["full_dataset"]
        subset_below_cutoff = subsets["subset_below_cutoff"]
        subset_above_cutoff = subsets["subset_above_cutoff"]
        subset_undecided = subsets["subset_undecided"]

        # Apply combined_function to get the updated datasets
        below_not_in_above, above_not_in_below, undecided_not_in_below_or_above = combined_function(
            name, full_data, subset_below_cutoff, subset_above_cutoff, subset_undecided
        )

        # Store the result in the new dictionary
        processed_results[name] = {
            "full_data": full_data,
            "below_not_in_above": below_not_in_above,
            "above_not_in_below": above_not_in_below,
            "undecided_not_in_below_or_above": undecided_not_in_below_or_above
        }

    return processed_results
