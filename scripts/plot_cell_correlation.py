import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from shapely.wkt import loads
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
from utils import load_pickle, read_lines, load_mask, load_image


def write_list_to_wkt(polygon_list, file_path):
    with open(file_path, 'w') as f:
        for polygon in polygon_list:
            f.write(polygon.wkt + '\n')


def read_wkt_to_list(file_path):
    polygons_list = []
    with open(file_path, 'r') as f:
        for line in f:
            # Remove leading/trailing whitespace and ignore empty lines
            wkt_string = line.strip()
            if wkt_string:
                try:
                    polygon = loads(wkt_string)
                    polygons_list.append(polygon)
                except Exception as e:
                    print(f"Error parsing WKT: {wkt_string}, Error: {e}")
    return polygons_list


def get_clipped(x, min_level=5, percentile=95):
    y = x.copy()
    y = y[~np.isnan(y)]
    n_levels = np.unique(y).__len__()
    if n_levels <= min_level:
        return x / np.nanmax(x)
    else:
        cutoff_l = np.sort(np.unique(y))[min_level - 1]
        cutoff_p = np.percentile(y, percentile)
        y = x.copy()
        y = np.clip(y, a_min=0, a_max=np.max([cutoff_l, cutoff_p]))
        y = y / np.nanmax(y)
        return y


def polygon_area_within_grid0(cell_id, polygon, grid_df):
    min_x, min_y, max_x, max_y = polygon.bounds
    area_df = pd.DataFrame(None, columns=['id', 'x', 'y', 'area'])
    for _, row in grid_df[(grid_df['x'] + 1 >= min_x) & (grid_df['x'] <= max_x) &
                            (grid_df['y'] + 1 >= min_y) & (grid_df['y'] <= max_y)].iterrows():
        cell = box(row['x'], row['y'], row['x'] + 1, row['y'] + 1)
        if polygon.intersects(cell):
            intersection = polygon.intersection(cell)
            new_row = {'id': cell_id, 'x': row['x'], 'y': row['y'], 'area': np.round(intersection.area, 5)}
            area_df = pd.concat([area_df, pd.DataFrame(new_row)], ignore_index=True)
            
    return area_df

def polygon_area_within_grid(cell_id, polygon, grid_df):
    min_x, min_y, max_x, max_y = polygon.bounds
    rows = []
    for _, row in grid_df[(grid_df['x'] + 1 >= min_x) & (grid_df['x'] <= max_x) &
                            (grid_df['y'] + 1 >= min_y) & (grid_df['y'] <= max_y)].iterrows():
        cell = box(row['x'], row['y'], row['x'] + 1, row['y'] + 1)
        if polygon.intersects(cell):
            intersection = polygon.intersection(cell)
            rows.append({
                'id': cell_id,
                'x': row['x'],
                'y': row['y'],
                'area': np.round(intersection.area, 5)
            })
            
    return pd.DataFrame(rows, columns=['id', 'x', 'y', 'area'])


def get_polygons_with_area(boundary_df, grid_df, mask_df=None):
    cell_ids = boundary_df['id'].unique()
    polygon_list = []
    areas_df = pd.DataFrame(None, columns=['id', 'x', 'y', 'area'])
    print("getting all polygons...")
    for cell_idx in tqdm(range(cell_ids.__len__())):
        cell_id = cell_ids[cell_idx]
        subset = boundary_df[boundary_df['id'] == cell_id]
        if subset.shape[0] >= 4:
            polygon = Polygon(subset[['x', 'y']].values)
            polygon = polygon.buffer(0)
        if mask_df is not None:
            mask_areas = polygon_area_within_grid(cell_id, polygon, mask_df)
            if mask_areas.shape[0] == 0:
                polygon_areas = polygon_area_within_grid(cell_id, polygon, grid_df)
                areas_df = pd.concat([areas_df, polygon_areas], ignore_index=True)
                polygon_list.append(polygon)
        else:
            polygon_areas = polygon_area_within_grid(cell_id, polygon, grid_df)
            areas_df = pd.concat([areas_df, polygon_areas], ignore_index=True)
            polygon_list.append(polygon)
    areas_df.id = areas_df.id.astype(str)
    areas_df.x = areas_df.x.astype(int)
    areas_df.y = areas_df.y.astype(int)
    return polygon_list, areas_df

def get_polygons_with_area(boundary_df, grid_df, mask_df=None):
    cell_ids = boundary_df['id'].unique()
    polygon_list = []
    areas_rows = []
    for cell_id in tqdm(cell_ids):
        subset = boundary_df[boundary_df['id'] == cell_id]
        if subset.shape[0] < 4:
            continue
        polygon = Polygon(subset[['x', 'y']].values).buffer(0)
        if mask_df is not None:
            mask_areas = polygon_area_within_grid(cell_id, polygon, mask_df)
            if mask_areas.empty:
                polygon_areas = polygon_area_within_grid(cell_id, polygon, grid_df)
                areas_rows.extend(polygon_areas.to_dict('records'))
                polygon_list.append(polygon)
        else:
            polygon_areas = polygon_area_within_grid(cell_id, polygon, grid_df)
            areas_rows.extend(polygon_areas.to_dict('records'))
            polygon_list.append(polygon)
    areas_df = pd.DataFrame(areas_rows, columns=['id', 'x', 'y', 'area'])
    areas_df = areas_df.astype({'id': str, 'x': int, 'y': int})
    return polygon_list, areas_df


def calculate_weighted_ge(areas_df, grid_df):
    merged_df = pd.merge(areas_df, grid_df, on=['x', 'y'], how='left')
    merged_df['weighted_ge'] = merged_df['ge'] * merged_df['area']
    result = merged_df.groupby('id').apply(
        lambda x: pd.Series({
            'total_area': x['area'].sum(),
            'avg_ge': x['weighted_ge'].sum() / x['area'].sum()
        })
    ).reset_index()
    return result


def plot_polygons_with_ge(polygon_list, ge, width, height, xlim, ylim, save_fig=True, out_dir=None,
                            fig_name='cells.jpg', title='Gene expression of cells', dpi=500):
    assert polygon_list.__len__() == ge.__len__()
    ge_arr = np.array(ge)
    ge_arr = ge_arr / np.max(ge_arr)
    width = width
    height = height
    plt.figure(figsize=(width, height))
    for idx in range(polygon_list.__len__()):
        polygon = polygon_list[idx]
        if polygon.geom_type == 'Polygon':
            x, y = polygon.exterior.xy
            plt.plot(x, y, linewidth=0, label=f'Polygon {idx}')
            plt.fill(x, y, color=plt.cm.viridis(ge_arr[idx]), alpha=0.5)
        elif polygon.geom_type == 'MultiPolygon':
            for subpolygon in polygon.geoms:
                x, y = subpolygon.exterior.xy
                plt.plot(x, y, linewidth=0, label=f'Polygon {idx}')
                plt.fill(x, y, color=plt.cm.viridis(ge_arr[idx]), alpha=0.5)
        else:
            print("wrong polygon gemo type...")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.colorbar(plt.cm.ScalarMappable(cmap='turbo'), label='Relative Expression')
    plt.gca().invert_yaxis()
    if save_fig:
        for dir in out_dir:
            plt.savefig(dir + fig_name, dpi=dpi)
    else:
        plt.show()


def plot_polygons_comparison(polygon_list, pred, truth, gn, title, width, height, xlim, ylim, save_fig=True,
                                out_dir=None, fig_name='cells_comparison.jpg', dpi=100):
    assert polygon_list.__len__() == pred.__len__()
    pred_arr = np.array(pred)
    pred_arr = pred_arr / np.max(pred_arr)
    assert polygon_list.__len__() == truth.__len__()
    truth_arr = np.array(truth)
    truth_arr = truth_arr / np.max(truth_arr)
    width = width
    height = height
    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    for idx in range(len(polygon_list)):
        polygon = polygon_list[idx]
        if polygon.geom_type == 'Polygon':
            x, y = polygon.exterior.xy
            axes[0].fill(x, y, color=plt.cm.turbo(pred_arr[idx]), alpha=0.5)
            axes[1].fill(x, y, color=plt.cm.turbo(truth_arr[idx]), alpha=0.5)
        elif polygon.geom_type == 'MultiPolygon':
            for subpolygon in polygon.geoms:
                x, y = subpolygon.exterior.xy
                axes[0].fill(x, y, color=plt.cm.turbo(pred_arr[idx]), alpha=0.5)
                axes[1].fill(x, y, color=plt.cm.turbo(truth_arr[idx]), alpha=0.5)
        else:
            print("wrong polygon gemo type...")
    axes[0].set_xlim(xlim)
    axes[0].set_ylim(ylim)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Predicted Gene Expression', fontsize=width * 0.9)
    axes[0].invert_yaxis()
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_title('True Gene Expression', fontsize=width * 0.9)
    axes[1].invert_yaxis()
    fig.suptitle('Gene: ' + gn + '\n', fontsize=width * 1.3, y=0.99)
    fig.text(0.5, 0.925 + height * 0.001, title,
             ha='center', va='center', fontsize=width * 1, color='blue')
    plt.tight_layout()
    if save_fig:
        for d in out_dir:
            plt.savefig(d + fig_name, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', type=str)
    parser.add_argument('--clip0', type=float, default=95)
    parser.add_argument('--clip1', type=float, default=99)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pref = args.pref
    gene_names = read_lines(f'{pref}gene-names.txt')
    clip0 = args.clip0
    clip1 = args.clip1
    factor = 16

    mask = load_mask(f"{pref}mask-small-hs.png")
    print("mask shape is ", mask.shape)

    print("loading boundary information...")

    cell_boundary = pd.read_csv(pref + "boundary.tsv", sep='\t', index_col=0)
    boundary = cell_boundary.copy()

    boundary.columns = boundary.columns.astype(str)
    boundary.id = boundary.id.astype(str)
    boundary.x = boundary.x // factor
    boundary.y = boundary.y // factor

    grid_df = pd.DataFrame({
        'x': np.tile(np.arange(0, mask.shape[1]), mask.shape[0]),
        'y': np.repeat(np.arange(0, mask.shape[0]), mask.shape[1]),
    })
    mask_df = pd.DataFrame({
        'x': np.where(~mask)[1],
        'y': np.where(~mask)[0]
    })

    if os.path.exists(pref + 'cell-polygons.wkt') and os.path.exists(pref + 'cell-area.tsv'):
        print("polygons already exist, now loading polygons...")
        polygons_list = read_wkt_to_list(pref + 'cell-polygons.wkt')
        areas_df = pd.read_csv(pref + 'cell-area.tsv', sep='\t', index_col=0)
    else:
        print("polygons do NOT exist, now calculating polygons...")
        polygons_list, areas_df = get_polygons_with_area(boundary, grid_df, mask_df)
        write_list_to_wkt(polygons_list, pref + 'cell-polygons.wkt')
        areas_df.to_csv(pref + 'cell-area.tsv', sep='\t')
    
    cell_ids = areas_df.id.unique().astype(str)

    adata = sc.read_10x_h5(pref + "cell_feature_matrix.h5")

    os.makedirs(pref + 'cell_corr_full', exist_ok=True)
    os.makedirs(pref + 'cell_corr_top100', exist_ok=True)
    corr_pearson_list = []
    corr_spearman_list = []
    gene_name_list = []

    # height = mask.shape[1] // 25
    # width = (int(np.floor(boundary.x.max()))) / (int(np.floor(boundary.y.max()))) * height * 2 + 1
    # height = height * 1.3
    height = 8
    width = mask.shape[1] / mask.shape[0] * (height-1) * 2

    intersected_genes = list(set(gene_names).intersection(adata.var_names.to_list()))
    print("number of genes for plotting is", intersected_genes.__len__())

    if os.path.exists(pref + 'cell_corr_comparison.tsv'):
        df = pd.read_csv(pref + 'cell_corr_comparison.tsv', sep='\t', index_col=0)
        corr_pearson_list = df['Pearson Corr.'].tolist()
        corr_spearman_list = df['Spearman Corr.'].tolist()
    else:
        for gn in gene_names:
            if gn in intersected_genes:
                ge_mat = load_pickle(pref + 'cnts-super/' + gn + '.pickle')
                assert ge_mat.shape[0] == grid_df.y.max() + 1
                assert ge_mat.shape[1] == grid_df.x.max() + 1
                grid_df['ge'] = ge_mat.flatten()

                avg_ge_df = calculate_weighted_ge(areas_df, grid_df)
                avg_ge_df['id'] = pd.Categorical(avg_ge_df['id'], categories=cell_ids, ordered=True)
                avg_ge_df = avg_ge_df.sort_values('id')

                pred_ge = np.array(avg_ge_df['avg_ge'])
                pred_ge = pred_ge / np.max(pred_ge)
                truth_ge = np.array(adata[cell_ids, gn].X.todense())
                truth_ge = truth_ge.reshape((truth_ge.shape[0],))
                truth_ge = truth_ge / avg_ge_df['total_area']
                truth_ge = truth_ge / np.max(truth_ge)

                p_corr = pearsonr(truth_ge, pred_ge)[0]
                corr_pearson_list.append(p_corr)
                s_corr = spearmanr(truth_ge, pred_ge)[0]
                corr_spearman_list.append(s_corr)
                gene_name_list.append(gn)

                title = 'Pearson Correlation: ' + str(np.round(p_corr, 3)) + ', Spearman Correlation: ' + str(
                    np.round(s_corr, 3))

                pred_ge = get_clipped(pred_ge, percentile=clip1)
                truth_ge = get_clipped(truth_ge, percentile=clip0)
                print("predicted ge shape is", pred_ge.shape)
                print("ground truth ge shape is", truth_ge.shape)
                print("polygons number is", polygons_list.__len__())

                if corr_pearson_list.__len__() <= 100:
                    plot_polygons_comparison(polygons_list, pred=pred_ge, truth=truth_ge, gn=gn,
                                                width=width, height=height,
                                                xlim=(0, mask.shape[1]), ylim=(0, mask.shape[0]),
                                                save_fig=True,
                                                out_dir=[pref + 'cell_corr_full/', pref + 'cell_corr_top100/'],
                                                title=title, fig_name=gn + '.jpg')
                else:
                    plot_polygons_comparison(polygons_list, pred=pred_ge, truth=truth_ge, gn=gn,
                                                width=width, height=height,
                                                xlim=(0, mask.shape[1]), ylim=(0, mask.shape[0]),
                                                save_fig=True, out_dir=[pref + 'cell_corr_full/'],
                                                title=title, fig_name=gn + '.jpg')

        df = pd.DataFrame(
            {'Gene': gene_name_list, 'Pearson Corr.': corr_pearson_list, 'Spearman Corr.': corr_spearman_list})
        df.to_csv(pref + 'cell_corr_comparison.tsv', sep='\t')

    df = pd.DataFrame({'Pearson Corr.': corr_pearson_list, 'Spearman Corr.': corr_spearman_list})
    fig, ax = plt.subplots()
    ax = sns.violinplot(data=df, orient="v", color='cornflowerblue')
    ax = sns.stripplot(data=df, orient="v", color='lightskyblue', jitter=True)
    quantiles = [0.25, 0.5, 0.75]
    linestyles = [':', '-', '--']
    for i in range(3):
        q = quantiles[i]
        quantile_line = df.quantile(q)
        ax.axhline(quantile_line['Pearson Corr.'], linestyle=linestyles[i], color='red',
                   label=f'{int(q * 100)}% Quantile of Pearson Corr.')
        ax.axhline(quantile_line['Spearman Corr.'], linestyle=linestyles[i], color='blue',
                   label=f'{int(q * 100)}% Quantile of Spearman Corr.')
    plt.ylim(-0.3, 1.2)
    plt.legend()
    plt.title('Pearson and Spearman Correlations for All Genes', fontsize=16)
    plt.savefig(pref + 'violin_plot_cell_corr_full.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax = sns.violinplot(data=df.iloc[0:100, :], orient="v", color='cornflowerblue')
    ax = sns.stripplot(data=df.iloc[0:100, :], orient="v", color='lightskyblue', jitter=True)
    quantiles = [0.25, 0.5, 0.75]
    linestyles = [':', '-', '--']
    for i in range(3):
        q = quantiles[i]
        quantile_line = df.quantile(q)
        ax.axhline(quantile_line['Pearson Corr.'], linestyle=linestyles[i], color='red',
                   label=f'{int(q * 100)}% Quantile of Pearson Corr.')
        ax.axhline(quantile_line['Spearman Corr.'], linestyle=linestyles[i], color='blue',
                   label=f'{int(q * 100)}% Quantile of Spearman Corr.')
    plt.ylim(-0.3, 1.2)
    plt.legend()
    plt.title('Pearson and Spearman Correlations for Top 100 HVG', fontsize=16)
    plt.savefig(pref + 'violin_plot_cell_corr_top100.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
