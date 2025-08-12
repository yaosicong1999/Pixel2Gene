import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from utils import load_pickle

def save_box_style_legend(output, start_idx, end_idx):
    """Saves a standalone legend with box-style elements"""
    # Create dummy legend elements with box patches
    legend_elements = [
        # Green box (Original-Observed)
        plt.Rectangle((0,0), 1, 1, 
                     facecolor='#BED4B7', edgecolor='black', linewidth=1,
                     label='Observed'),
        # Orange box (Downsampled-Observed)
        plt.Rectangle((0,0), 1, 1, 
                     facecolor='#F6C6AE', edgecolor='black', linewidth=1,
                     label='Downsampled'),
        # Blue box (Downsampled-Imputation)
        plt.Rectangle((0,0), 1, 1, 
                     facecolor='#A0DBFB', edgecolor='black', linewidth=1,
                     label='Downsampled-Imputed')
    ]
    
    # Create standalone legend figure
    fig_leg = plt.figure(figsize=(5, 0.8))
    fig_leg.legend(handles=legend_elements,
                   loc='center',
                   ncol=1,
                   frameon=False,
                   handlelength=1.5,
                   handleheight=1.5)
    
    # Save legend
    fig_leg.savefig(f'{output}No.{start_idx}-No.{end_idx}_box_moran_legend.png', 
                    dpi=300,
                    bbox_inches='tight',
                    transparent=True)
    plt.close(fig_leg)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=20000)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--all_spots', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    start_idx = args.start_idx
    end_idx = args.end_idx
    all_spots = args.all_spots

    output = args.output + 'gene_metrics_comparison_all_spots/' if all_spots else args.output + 'gene_metrics_comparison/'
    output = output + args.mask.split('/')[-1].split('.')[0] + '/' if args.mask is not None else output + 'no_mask/'
    x = load_pickle(f'{output}No.{start_idx}-No.{end_idx}_metrics_data.pickle')
    data = []

    # 1. Moran's I
    for condition in ['Original-Observed', 'Downsampled-Observed', 'Downsampled-Imputation']:
        for scale in ['16', '32', '64']:
            key = f"Morans I-{scale}"
            if key in x:
                values = x[key][condition]
                for v in values:
                    data.append({'Metric': f"Moran's I-{scale}", 'Condition': condition, 'Value': v})


    # Convert to DataFrame
    df = pd.DataFrame(data)

    # --- Set Arial Font ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'  # Fallback options
    plt.rcParams['ytick.labelsize'] = 15  # Default font size

    # --- Data Preparation ---
    # [Your existing data preparation code]
    df['Group'] = df['Metric'] + '|' + df['Condition']

    # --- Define Colors ---
    # Box colors (light)
    box_colors = {
        'Downsampled-Observed': '#F6C6AE',  # Light orange
        'Downsampled-Imputation': '#A0DBFB', # Light blue
        'Original-Observed': '#BED4B7'       # Light green
    }

    # Point colors (your exact RGB values)
    point_colors = {
        'Downsampled-Observed': (233/255, 133/255, 50/255),  # Orange (233,133,50)
        'Downsampled-Imputation': (9/255, 118/255, 160/255), # Blue (9,118,160)
        'Original-Observed': (59/255, 126/255, 35/255)       # Green (59,126,35)
    }

    # Group order and settings
    all_groups = [
        "Moran's I-16|Original-Observed",
        "Moran's I-16|Downsampled-Observed", 
        "Moran's I-16|Downsampled-Imputation",
        'SPACER',
        "Moran's I-32|Original-Observed",
        "Moran's I-32|Downsampled-Observed", 
        "Moran's I-32|Downsampled-Imputation",
        'SPACER',
        "Moran's I-64|Original-Observed",
        "Moran's I-64|Downsampled-Observed", 
        "Moran's I-64|Downsampled-Imputation"
    ]
    # --- Plot Setup ---
    plt.figure(figsize=(8, 6))
    sns.set_style("white", {
        "grid.color": "#EBEBEB",
        "grid.linewidth": 0.8
    })
    
    positions = np.arange(len(all_groups))
    box_width = 0.6

    # --- Plot Boxes ---
    for pos, group in zip(positions, all_groups):
        if group != 'SPACER':
            metric, condition = group.split('|')
            subset = df[df['Group'] == group]
            
            if not subset.empty:
                bp = plt.boxplot(
                    subset['Value'],
                    positions=[pos],
                    widths=box_width,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=box_colors[condition], 
                                edgecolor='black', linewidth=1),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1),
                    medianprops=dict(color='black', linewidth=1.5, linestyle='--')
                )
        else:
            bp = plt.boxplot(
                [[np.nan]],
                positions=[pos],
                widths=box_width*0.8,
                patch_artist=True,
                boxprops=dict(facecolor='white', edgecolor='white', linewidth=0),
                whiskerprops=dict(color='white', linewidth=0),
                capprops=dict(color='white', linewidth=0),
                medianprops=dict(color='white', linewidth=0)
            )

    # --- Add Points with Correct Colors ---
    for pos, group in zip(positions, all_groups):
        if group != 'SPACER':
            metric, condition = group.split('|')
            subset = df[df['Group'] == group]
            
            if not subset.empty:
                x_jitter = np.random.normal(0, 0.06, size=len(subset))
                plt.scatter(
                    x=pos + x_jitter,
                    y=subset['Value'],
                    color=point_colors[condition],  # Using your exact RGB values
                    alpha=0.8,
                    s=5,
                    edgecolor='gray',
                    linewidths=0.5,  # Adjust this value to control edge width
                    zorder=10
                )

    # --- Final Formatting ---
    ax = plt.gca()
    ax.set(
        xlim=(positions[0]-0.5, positions[-1]+0.5),
        ylim=(0, np.max(df['Value'])*1.2),
        xticks=[],
        xticklabels=[],
        ylabel=None
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
    ax.grid(True, axis='y', color='#EBEBEB', linewidth=0.8)
    ax.grid(True, axis='x', color='#EBEBEB', linewidth=0.5, alpha=0.3)
    ax.tick_params(left=True, bottom=False)
    save_box_style_legend(output, start_idx, end_idx)
    plt.tight_layout()
    plt.savefig(f'{output}No.{start_idx}-No.{end_idx}_final_violin_moran.png', dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()