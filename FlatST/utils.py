import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from scipy.spatial import ConvexHull
import seaborn as sns
import scanpy as sc
from shapely.geometry import Polygon


def find_region_mapping0(adata, pred_column=None, original_region_column='Region'):
    if pred_column is None:
        raise ValueError("Please provide the names of the predicted regions.")

    # Obtain the unique category of the original area and the predicted area
    original_regions = adata.obs[original_region_column].unique()
    pred_regions = adata.obs[pred_column].unique()

    # Check whether the number of categories is the same
    if len(original_regions) != len(pred_regions):
        print(
            f"Warning: The original area quantity is{len(original_regions)}，The number of predicted areas is {len(pred_regions)}\n"
            f"The number of categories is different. Please clean the data and then call this function again.")
        return None

    all_mappings = []
    for pred in pred_regions:
        pred_mask = adata.obs[pred_column] == pred
        region_counts = adata.obs[original_region_column][pred_mask].value_counts()
        for region, count in region_counts.items():
            all_mappings.append((pred, region, count))

    all_mappings.sort(key=lambda x: x[2], reverse=True)

    mapping = {}
    used_original_regions = set()
    used_pred_regions = set()

    for pred, region, count in all_mappings:
        if pred not in used_pred_regions and region not in used_original_regions:
            mapping[pred] = region
            used_original_regions.add(region)
            used_pred_regions.add(pred)

    for pred in pred_regions:
        if pred not in mapping:
            mapping[pred] = ''

    return mapping


def find_region_mapping(adata, pred_column=None, original_region_column='Region'):
    """
    find_region_mapping is an upgraded version of find_region_mapping0, enabling one-to-one correspondence even when regions are not equal
    """
    if pred_column is None:
        raise ValueError("Please provide the names of the predicted regions.")

    original_regions = adata.obs[original_region_column].unique()
    pred_regions = adata.obs[pred_column].unique()

    all_mappings = []
    for pred in pred_regions:
        pred_mask = adata.obs[pred_column] == pred
        region_counts = adata.obs[original_region_column][pred_mask].value_counts()
        for region, count in region_counts.items():
            all_mappings.append((pred, region, count))

    all_mappings.sort(key=lambda x: x[2], reverse=True)

    mapping = {}
    used_original_regions = set()
    used_pred_regions = set()

    for pred, region, count in all_mappings:
        if pred not in used_pred_regions and region not in used_original_regions:
            mapping[pred] = region
            used_original_regions.add(region)
            used_pred_regions.add(pred)

    for pred in pred_regions:
        if pred not in mapping:
            mapping[pred] = ''

    if len(pred_regions) < len(original_regions):
        print(f"Warning: The number of predicted areas is{len(pred_regions)}，Less than the original area quantity{len(original_regions)}，Matching has been completed.")

    return mapping


def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data


def plot_single_region_boundary(adata, target_region, label_column='mclust', invert_y=False, num=10,
                                custom_color='#33FF57', s=20):
    regions = adata.obs['Region'].unique()
    if target_region not in regions:
        print(f"Error: The region '{target_region}' is not found in adata.obs['Region'].")
        return
    spatial_coords = adata.obsm['spatial']

    def plot_boundary(coords, color, line_style='--', alpha=0.8, line_width=1.5):
        if len(coords) >= num:
            hull = ConvexHull(coords)
            for simplex in hull.simplices:
                plt.plot(
                    coords[simplex, 0],
                    coords[simplex, 1],
                    color=color,
                    linestyle=line_style,
                    alpha=alpha,
                    linewidth=line_width,
                    zorder=2
                )
        plt.axis('off')

    region_mask = adata.obs['Region'] == target_region

    # If the user provides a custom color, use that color
    if custom_color is not None:
        palette = {target_region: custom_color}
    else:
        palette = None

    # Obtain the coordinates of the current area
    origin_indices = np.where(region_mask)[0]
    origin_coords = spatial_coords[origin_indices]

    # Process the predicted area
    pred_labels = adata.obs[label_column][region_mask]
    unique_pred_labels = np.unique(pred_labels)
    if len(unique_pred_labels) > 0:
        main_pred_label = max(unique_pred_labels, key=lambda x: sum(pred_labels == x))
        pred_indices = np.where(pred_labels == main_pred_label)[0]
        pred_coords = origin_coords[pred_indices]
    else:
        pred_coords = np.array([])

    def calculate_iou(origin_coords, pred_coords):
        if len(origin_coords) < num or len(pred_coords) < num:
            return 0
        origin_hull = ConvexHull(origin_coords)
        origin_polygon = Polygon(origin_coords[origin_hull.vertices])
        pred_hull = ConvexHull(pred_coords)
        pred_polygon = Polygon(pred_coords[pred_hull.vertices])
        intersection = origin_polygon.intersection(pred_polygon).area
        union = origin_polygon.union(pred_polygon).area
        if union == 0:
            return 0
        return intersection / union

    IoU = calculate_iou(origin_coords, pred_coords)

    # Draw the original image
    sc.pl.embedding(
        adata[region_mask],
        basis="spatial",
        color='Region',
        s=s,
        show=False,
        title='{}(IoU={:.2f})'.format(target_region, IoU),
        frameon=False,
        zorder=1,
        palette=palette
    )

    # Draw the boundaries of the original area
    plot_boundary(origin_coords, color='dodgerblue')

    if len(pred_coords) > 0:
        plot_boundary(pred_coords, color='tomato', line_style='-.')

    plt.tight_layout()
    if invert_y:
        plt.gca().invert_yaxis()
    plt.show()


def plot_all_regions_boundaries(adata, label_column='mclust', rows=3, invert_y=False, num=10):
    regions = adata.obs['Region'].unique()
    num_regions = len(regions)
    cols = int(np.ceil(num_regions / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    spatial_coords = adata.obsm['spatial']

    def plot_boundary(ax, coords, color, line_style='--', alpha=0.8, line_width=1.5):
        if len(coords) >= num:
            hull = ConvexHull(coords)
            for simplex in hull.simplices:
                ax.plot(
                    coords[simplex, 0],
                    coords[simplex, 1],
                    color=color,
                    linestyle=line_style,
                    alpha=alpha,
                    linewidth=line_width
                )
        ax.axis('off')

    def calculate_iou(origin_coords, pred_coords):
        if len(origin_coords) >= num:
            origin_hull = ConvexHull(origin_coords)
            origin_polygon = Polygon(origin_coords[origin_hull.vertices])
        else:
            origin_polygon = Polygon()

        if len(pred_coords) >= num:
            pred_hull = ConvexHull(pred_coords)
            pred_polygon = Polygon(pred_coords[pred_hull.vertices])
        else:
            pred_polygon = Polygon()

        # Calculate the areas of the intersection and union
        intersection_area = origin_polygon.intersection(pred_polygon).area
        union_area = origin_polygon.union(pred_polygon).area

        # Calculate IoU
        if union_area > 0:
            iou = intersection_area / union_area
        else:
            iou = 0

        return iou

    for i, region in enumerate(regions):
        row = i // cols
        col = i % cols
        region_mask = adata.obs['Region'] == region
        origin_indices = np.where(region_mask)[0]
        origin_coords = spatial_coords[origin_indices]
        pred_labels = adata.obs[label_column][region_mask]
        unique_pred_labels = np.unique(pred_labels)
        if len(unique_pred_labels) > 0:
            max_count = 0
            main_pred_label = None
            for pred_label in unique_pred_labels:
                pred_indices = np.where(pred_labels == pred_label)[0]
                count = len(pred_indices)
                if count > max_count:
                    max_count = count
                    main_pred_label = pred_label
            pred_indices = np.where(pred_labels == main_pred_label)[0]
            pred_coords = origin_coords[pred_indices]
        else:
            pred_coords = np.array([])
        iou = calculate_iou(origin_coords, pred_coords)

        plot_boundary(axes[row, col], origin_coords, color='dodgerblue', line_style='--', alpha=0.8, line_width=1.5)
        plot_boundary(axes[row, col], pred_coords, color='tomato', line_style='-.', alpha=0.8, line_width=1.5)
        axes[row, col].set_title(f"{region} (IoU={iou:.2f})")

    # Hide the redundant subgraphs
    for i in range(num_regions, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    if invert_y:
        for row in axes:
            for ax in row:
                ax.invert_yaxis()
    plt.show()


def plot_region_boundaries(adata, label_column='mclust', invert_y=False, num=10):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    adata.obsm['spatial'][:, 1] = -adata.obsm['spatial'][:, 1]
    spatial_coords = adata.obsm['spatial']

    def plot_regions(ax, labels, use_numeric_label=False):
        region_labels = labels.astype(str)
        unique_regions = np.unique(region_labels)
        for i, region in enumerate(unique_regions):
            region_indices = np.where(region_labels == region)[0]
            region_coords = spatial_coords[region_indices]
            if len(region_coords) >= num:
                hull = ConvexHull(region_coords)
                for simplex in hull.simplices:
                    ax.plot(
                        region_coords[simplex, 0],
                        region_coords[simplex, 1],
                        color='#808080',
                        linestyle='--',
                        alpha=1,
                        linewidth=0.8
                    )

                center_x = np.mean(region_coords[:, 0])
                center_y = np.mean(region_coords[:, 1])

                if use_numeric_label:
                    label = str(i)
                else:
                    label = region
                ax.text(
                    center_x,
                    center_y,
                    label,
                    fontsize=8,
                    ha='center',
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.1)
                )

        ax.axis('off')

    # The first subgraph: Draw adata.obs['Region']
    plot_regions(axes[0], adata.obs['Region'], use_numeric_label=False)
    axes[0].set_title('origin_boundary')
    IoU = calculate_iou(adata, label_column=label_column, num=num)
    # The second subgraph: Draw the prediction results
    plot_regions(axes[1], adata.obs[label_column], use_numeric_label=True)
    axes[1].set_title('pred_boundary(IoU={:.2f})'.format(IoU))

    plt.tight_layout()
    if invert_y:
        for ax in axes:
            ax.invert_yaxis()
    plt.show()


def calculate_iou(adata, label_column='mclust', num=10):
    regions = adata.obs['Region'].unique()
    num_regions = len(regions)
    spatial_coords = adata.obsm['spatial']
    total_cells = len(adata)
    weighted_iou_sum = 0

    for region in regions:
        region_mask = adata.obs['Region'] == region
        origin_indices = np.where(region_mask)[0]
        origin_coords = spatial_coords[origin_indices]

        pred_labels = adata.obs[label_column][region_mask]
        unique_pred_labels = np.unique(pred_labels)
        if len(unique_pred_labels) > 0:
            max_count = 0
            main_pred_label = None
            for pred_label in unique_pred_labels:
                pred_indices = np.where(pred_labels == pred_label)[0]
                count = len(pred_indices)
                if count > max_count:
                    max_count = count
                    main_pred_label = pred_label
            pred_indices = np.where(pred_labels == main_pred_label)[0]
            pred_coords = origin_coords[pred_indices]
        else:
            pred_coords = np.array([])

        if len(origin_coords) >= num:
            origin_hull = ConvexHull(origin_coords)
            origin_polygon = Polygon(origin_coords[origin_hull.vertices])
        else:
            origin_polygon = Polygon()

        if len(pred_coords) >= num:
            pred_hull = ConvexHull(pred_coords)
            pred_polygon = Polygon(pred_coords[pred_hull.vertices])
        else:
            pred_polygon = Polygon()

        intersection_area = origin_polygon.intersection(pred_polygon).area
        union_area = origin_polygon.union(pred_polygon).area

        if union_area > 0:
            iou = intersection_area / union_area
        else:
            iou = 0
        weight = len(origin_indices) / total_cells
        weighted_iou_sum += weight * iou

    return weighted_iou_sum


def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y'], plot_Stats=False):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1 / num_batch_x) * x * 100) for x in range(num_batch_x + 1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1 / num_batch_y) * x * 100) for x in range(num_batch_y + 1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x + 1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y + 1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list


def Cal_Spatial_Net(adata, rad_cutoff=None, verbose=True):
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    # Radiion-based method
    nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index)))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    # Obtain the cell correlation coefficient matrix
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
        print("'highly_variable' not in adata.var.columns!")
    dense_X = adata_Vars.X.toarray()
    corr_matrix = np.corrcoef(dense_X)

    # Adjust the weights of the edges according to the correlation
    edge_weights = []
    cell_index_dict = {cell: idx for idx, cell in enumerate(adata.obs.index)}
    for i, j in zip(Spatial_Net['Cell1'], Spatial_Net['Cell2']):
        i_idx = cell_index_dict[i]
        j_idx = cell_index_dict[j]
        weight = corr_matrix[i_idx, j_idx]
        edge_weights.append(weight)
    Spatial_Net['weight'] = edge_weights

    # Use edge weights when constructing graph data
    edge_index = torch.LongTensor(
        np.array([Spatial_Net['Cell1'].map(cell_index_dict), Spatial_Net['Cell2'].map(cell_index_dict)]))
    # Convert Pandas Series to a one-dimensional array
    edge_weight = torch.FloatTensor(Spatial_Net['weight'].values)
    # data = Data(edge_index=edge_index, x=torch.FloatTensor(adata.X.todense()), edge_weight=edge_weight)
    if sp.issparse(adata.X):
        data = Data(edge_index=edge_index, x=torch.FloatTensor(adata.X.todense()), edge_weight=edge_weight)
    else:
        data = Data(edge_index=edge_index, x=torch.FloatTensor(adata.X), edge_weight=edge_weight)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net
    return data


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,
                       key_section='Section_id', section_order=None, verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff_2D
        radius cutoff for 2D SNN construction.
    rad_cutoff_Zaxis
        radius cutoff for 2D SNN construction for consturcting SNNs between adjacent sections.
    key_section
        The columns names of section_ID in adata.obs.
    section_order
        The order of sections. The SNNs between adjacent sections are constructed according to this order.

    Returns
    -------
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
    adata.uns['Spatial_Net_2D'] = pd.DataFrame()
    adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
    num_section = np.unique(adata.obs[key_section]).shape[0]
    if verbose:
        print('Radius used for 2D SNN:', rad_cutoff_2D)
        print('Radius used for SNN between sections:', rad_cutoff_Zaxis)
    for temp_section in np.unique(adata.obs[key_section]):
        if verbose:
            print('------Calculating 2D SNN of section ', temp_section)
        temp_adata = adata[adata.obs[key_section] == temp_section,]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_2D, verbose=False)
        temp_adata.uns['Spatial_Net']['SNN'] = temp_section
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
        adata.uns['Spatial_Net_2D'] = pd.concat(
            [adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])
    for it in range(num_section - 1):
        section_1 = section_order[it]
        section_2 = section_order[it + 1]
        if verbose:
            print('------Calculating SNN between adjacent section %s and %s.' %
                  (section_1, section_2))
        Z_Net_ID = section_1 + '-' + section_2
        temp_adata = adata[adata.obs[key_section].isin(
            [section_1, section_2]),]
        Cal_Spatial_Net(
            temp_adata, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
        spot_section_trans = dict(
            zip(temp_adata.obs.index, temp_adata.obs[key_section]))
        temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(
            spot_section_trans)
        temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(
            spot_section_trans)
        used_edge = temp_adata.uns['Spatial_Net'].apply(
            lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge,]
        temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, [
                                                                                 'Cell1', 'Cell2', 'Distance']]
        temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
        if verbose:
            print('This graph contains %d edges, %d cells.' %
                  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
            print('%.4f neighbors per cell on average.' %
                  (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
        adata.uns['Spatial_Net_Zaxis'] = pd.concat(
            [adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])
    adata.uns['Spatial_Net'] = pd.concat(
        [adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
    if verbose:
        print('3D SNN contains %d edges, %d cells.' %
              (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %
              (adata.uns['Spatial_Net'].shape[0] / adata.n_obs))


def Stats_Spatial_Net(adata):
    sns.set_style("whitegrid")
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    palette = sns.color_palette("muted", len(plot_df.index))
    ax.bar(plot_df.index, plot_df, color=palette)
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.tick_params(axis='x')
    ax.tick_params(axis='y')
    ax.set_facecolor('#f9f9f9')
    for spine in ax.spines.values():
        spine.set_edgecolor('lightgray')
    plt.show()


# EEE
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='FlatST', random_seed=2025):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    # mclust_res = np.array(res[-2])

    try:
        mclust_res = np.array(res[-2])
        # raise RuntimeError("Mclust returned NULL")
    except Exception as e:
        print(f"\nWarning: There are a large number of duplicate vectors in the graph structure.\n" \
              f"Mclust clustering failed: {e}.\n" \
              f"We recommend that you can reduce the num_smooth_iterations or increase the initial_alpha then try again!!!\n")

        from sklearn.cluster import KMeans
        print("Using KMeans clustering instead of Mclust...")
        kmeans = KMeans(n_clusters=num_cluster, random_state=random_seed)
        mclust_res = kmeans.fit_predict(adata.obsm[used_obsm])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
