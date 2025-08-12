import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from utils import load_pickle

def save_box_style_legend(output, start_idx, end_idx):
    """Saves a standalone legend with box-style elements"""
    # Create dummy legend elements with box patches
    legend_elements = [
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
    fig_leg.savefig(f'{output}No.{start_idx}-No.{end_idx}_box_legend.png', 
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

    # 3. RMSE
    for condition in ['Downsampled-Observed', 'Downsampled-Imputation']:
        values = x['RMSE'][condition]
        for v in values:
            data.append({'Metric': 'RMSE', 'Condition': condition, 'Value': v})
    
    # 4. MAE
    for condition in ['Downsampled-Observed', 'Downsampled-Imputation']:
        values = x['MAE'][condition]
        for v in values:
            data.append({'Metric': 'MAE', 'Condition': condition, 'Value': v})

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
    }

    # Point colors (your exact RGB values)
    point_colors = {
        'Downsampled-Observed': (233/255, 133/255, 50/255),  # Orange (233,133,50)
        'Downsampled-Imputation': (9/255, 118/255, 160/255), # Blue (9,118,160)
    }

    # Group order and settings
    all_groups = [
        'RMSE|Downsampled-Observed',
        'RMSE|Downsampled-Imputation',
        'SPACER',  # Spacer for visual separation
        'MAE|Downsampled-Observed',
        'MAE|Downsampled-Imputation' 
    ]

    # --- Plot Setup ---
    plt.figure(figsize=(7, 6))
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

    # Compute separate max for RMSE and mae
    ax = plt.gca()
    max_value = np.percentile(df['Value'], 99)
    
    from matplotlib.ticker import ScalarFormatter, MaxNLocator

    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((-5,5))
    yticks = ax.get_yticks()
    if max_value * 1.1 > yticks[-1]:
        yticks = np.append(yticks, max_value * 1.1)
    ax.set_yticks(yticks)
    ax.set(
        xlim=(positions[0]-0.5, positions[-1]+0.5),
        ylim=(0, max_value*1.2),
        xticks=[],
        xticklabels=[],
        ylabel=None
    )
    # Common grid and ticks
    ax.grid(True, axis='y', color='#EBEBEB', linewidth=0.8)
    ax.grid(True, axis='x', color='#EBEBEB', linewidth=0.5, alpha=0.3)
    ax.grid(True, axis='y', color='#EBEBEB', linewidth=0.8)
    ax.grid(True, axis='x', color='#EBEBEB', linewidth=0.5, alpha=0.3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    ax.tick_params(left=True, bottom=False)
    save_box_style_legend(output, start_idx, end_idx)
    plt.tight_layout()
    plt.savefig(f'{output}No.{start_idx}-No.{end_idx}_final_violin_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()