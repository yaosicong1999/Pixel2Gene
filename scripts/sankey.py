import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
from scipy.optimize import linear_sum_assignment
from collections import Counter
import pandas as pd
import itertools
from utils import save_image
from visual import plot_labels
from sklearn.metrics import normalized_mutual_info_score

def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x

def cmap_tab20(x):
    cmap = plt.get_cmap('tab20')
    x = x % 20
    x = (x // 10) + (x % 10) * 2
    return cmap(x)

def get_aligned_labels(cl0, cl1):
    # Compute confusion matrix between cl0 and cl1
    matrix = pd.crosstab(cl0, cl1)
    cost_matrix = -matrix.values  # Hungarian algorithm minimizes cost
    # Find optimal one-to-one mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Create mapping from cl1 to cl0
    cl0_labels = matrix.index.values
    cl1_labels = matrix.columns.values
    mapping = {cl1_labels[col_ind[i]]: cl0_labels[row_ind[i]] for i in range(len(row_ind))}
    # Apply mapping
    new_cl1 = np.array([mapping[label] for label in cl1])
    return new_cl1


data_dir = "/Users/sicongy/Documents/GitHub/rotation_1/"
o_labels = pd.read_pickle(data_dir + "labels.pickle")

# d_labels = pd.read_pickle(data_dir + "d_labels.pickle")
dp_labels = pd.read_pickle(data_dir + "cosmx_o.pickle")

o_labels_vec = o_labels.flatten()
# d_labels_vec = d_labels.flatten()
dp_labels_vec = dp_labels.flatten()

#mask = (o_labels_vec == -1) | (d_labels_vec == -1) | (dp_labels_vec == -1)
mask = (o_labels_vec == -1) | (dp_labels_vec == -1)

o_labels_vec2 = o_labels_vec[~mask]
label_mapping = {old_label: new_label for new_label, old_label in enumerate(pd.Series(o_labels_vec2).value_counts().index)}
o_labels_vec2 = pd.Series(o_labels_vec2).map(label_mapping).to_numpy()

# d_labels_vec2 = d_labels_vec[~mask].astype(str)
dp_labels_vec2 = dp_labels_vec[~mask].astype(str)

## getting aligned vectors
d_labels_vec2 = get_aligned_labels(o_labels_vec2, d_labels_vec2)
dp_labels_vec2 = get_aligned_labels(o_labels_vec2, dp_labels_vec2)

## calculate NMI
nmi_d = normalized_mutual_info_score(o_labels_vec2, d_labels_vec2)
print("NMI-d:", nmi_d)
nmi_dp = normalized_mutual_info_score(o_labels_vec2, dp_labels_vec2)
print("NMI-dp:", nmi_dp)

## save matrix-like label images
d_labels_align = np.full_like(mask, -1, dtype=int)
d_labels_align[~mask] = d_labels_vec2
d_labels_align = d_labels_align.reshape(o_labels.shape[0], o_labels.shape[1])
plot_labels(d_labels_align, 'd_aligned.png')

dp_labels_align = np.full_like(mask, -1, dtype=int)
dp_labels_align[~mask] = dp_labels_vec2
dp_labels_align = dp_labels_align.reshape(o_labels.shape[0], o_labels.shape[1])
plot_labels(dp_labels_align, 'dp_aligned3.png')

o_labels_align = np.full_like(mask, -1, dtype=int)
o_labels_align[~mask] = o_labels_vec2
o_labels_align = o_labels_align.reshape(o_labels.shape[0], o_labels.shape[1])
plot_labels(o_labels_align, 'o_aligned3.png')


for i in range(20):
    mask2 = (o_labels == 6) | (o_labels == i)
    o_labels_align = np.where(mask2, o_labels, -1)
    plot_labels(o_labels_align, f'cosmx_{i}.png')

mask2 = (o_labels == 3) | (o_labels == 14)
o_labels_align = np.where(mask2, o_labels, -1)
plot_labels(o_labels_align, f'cosmx.png')



d_label_named = [f'Downsampled-observed-{i}' for i in d_labels_vec2]
o_label_named = [f'Observed-{i}' for i in o_labels_vec2]
dp_label_named = [f'Downsampled-enhanced-{i}' for i in dp_labels_vec2]

n_cluster = 15
labels = (
    [f'Downsampled-observed-{i}' for i in range(n_cluster)] +
    [f'Observed-{i}' for i in range(n_cluster)] +
    [f'Downsampled-enhanced-{i}' for i in range(n_cluster)]
)
label_to_idx = {label: idx for idx, label in enumerate(labels)}

d_label_named  = [f'Downsampled-observed-{i}' for i in d_labels_vec2]
o_label_named  = [f'Observed-{i}' for i in o_labels_vec2]
dp_label_named = [f'Downsampled-enhanced-{i}' for i in dp_labels_vec2]

edges_left = list(zip(d_label_named, o_label_named))
edges_right = list(zip(o_label_named, dp_label_named))


edge_counts_left = Counter(edges_left)
df_edges_left = pd.DataFrame(
    [(source, destination, value) for (source, destination), value in edge_counts_left.items()],
    columns=['source', 'destination', 'value']
)
df_edges_left['source'] = df_edges_left['source'].astype('category')
df_edges_left['destination'] = df_edges_left['destination'].astype('category')
sources = df_edges_left['source'].cat.categories
destinations = df_edges_left['destination'].cat.categories
all_pairs = pd.DataFrame(
    list(itertools.product(sources, destinations)),
    columns=['source', 'destination']
)
df_full_left = all_pairs.merge(df_edges_left, on=['source', 'destination'], how='left')
df_full_left['value'] = df_full_left['value'].fillna(0).astype(int)
df_full_left['source'] = pd.Categorical(df_full_left['source'], [f'Downsampled-observed-{i}' for i in range(n_cluster)])
df_full_left['destination'] = pd.Categorical(df_full_left['destination'], [f'Observed-{i}' for i in range(n_cluster)])
df_full_left.sort_values(['source', 'destination'], inplace = True)
df_full_left = df_full_left.reset_index(drop=True)
df_full_left['color'] = [i for i in range(n_cluster)]*n_cluster


edge_counts_right = Counter(edges_right)
df_edges_right = pd.DataFrame(
    [(source, destination, value) for (source, destination), value in edge_counts_right.items()],
    columns=['source', 'destination', 'value']
)
df_edges_right['source'] = df_edges_right['source'].astype('category')
df_edges_right['destination'] = df_edges_right['destination'].astype('category')
sources = df_edges_right['source'].cat.categories
destinations = df_edges_right['destination'].cat.categories
all_pairs = pd.DataFrame(
    list(itertools.product(sources, destinations)),
    columns=['source', 'destination']
)
df_full_right = all_pairs.merge(df_edges_right, on=['source', 'destination'], how='left')
df_full_right['value'] = df_full_right['value'].fillna(0).astype(int)
df_full_right['source'] = pd.Categorical(df_full_right['source'], [f'Observed-{i}' for i in range(n_cluster)])
df_full_right['destination'] = pd.Categorical(df_full_right['destination'], [f'Downsampled-enhanced-{i}' for i in range(n_cluster)])
df_full_right.sort_values(['source', 'destination'], inplace = True)
df_full_right = df_full_right.reset_index(drop=True)
df_full_right['color'] = [i for i in range(n_cluster) for _ in range(n_cluster)]

def rgb_to_hex(rgb_tuple):
    return "#{:02x}{:02x}{:02x}".format(*rgb_tuple)

def darken_rgb(rgb_tuple, factor=0.8):
    # factor between 0 (black) and 1 (original color)
    r, g, b = rgb_tuple
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return (r, g, b)

rgb_list = [tuple((np.array(cmap_tab20(i)[:3]) * 255).astype(int)) for i in range(n_cluster)]
color_map = {i: color for i, color in enumerate(rgb_list)}

hex_list = [rgb_to_hex(rgb) for rgb in rgb_list]

hex_dark_list = [rgb_to_hex(darken_rgb(rgb)) for rgb in rgb_list]


mynode = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      # label = [f'Downsamped-observed-{i}' for i in range(20)] + [f'Observed-{i}' for i in range(20)] + [f'Downsamped-enhanced-{i}' for i in range(20)],
      label=[''] * n_cluster * 3,
      x = [0.001]*n_cluster + [0.5]*n_cluster + [0.999]*n_cluster,
      y = [0.025, 0.085] + [(i + 0.5) / n_cluster for i in range(2, n_cluster)] + [(i + 0.5) / n_cluster for i in range(n_cluster)] + [0.025, 0.085] + [(i + 0.5) / n_cluster for i in range(2, n_cluster)],
      color = hex_dark_list*3)

## if right one 2nd (cl1) is bigger than 1st (cl0), then you need to add y of 2nd one a little bit

mylink = dict(
    source = [i for i in range(n_cluster) for _ in range(n_cluster)] + [i for i in range(n_cluster, n_cluster*2) for _ in range(n_cluster)],
    target = list(range(n_cluster, n_cluster*2))*n_cluster + list(range(n_cluster*2, n_cluster*3))*n_cluster,
    value = df_full_left.value.to_list() + df_full_right.value.to_list(),
    color = [hex_list[i] for i in df_full_left.color.to_list()] + [hex_list[i] for i in df_full_right.color.to_list()] )

fig = go.Figure(data=[go.Sankey(
    arrangement='snap',
    node = mynode,
    link = mylink
)])

fig.update_layout(title_text="Basic Sankey Diagram", font_size=20)
fig.show()
