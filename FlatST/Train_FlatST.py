import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from .FlatST import FlatST
from .utils import Transfer_pytorch_Data
import torch
import torch.backends.cudnn as cudnn
from sklearn.mixture import GaussianMixture

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
import matplotlib.pyplot as plt

def train_FlatST(adata, hidden_dims=[600, 10], n_epochs=1300, lr=0.001, key_added='FlatST',
                 gradient_clipping=5., weight_decay=0.0001, verbose=True,
                 random_seed=2025, save_loss=False, save_reconstrction=False,
                 is_early_stopping=True, cuda_device=0,
                 patience=100, keep_percent=0.99, num_smooth_iterations=[1, 0], is_distribution=1, initial_alpha=1.0):
    """

    Parameters
    ----------
    is_early_stopping:Whether to activate the early stop mechanism;
    cuda_device:The cuda device used;
    patience:The minimum number of training sessions;
    keep_percent:The retention ratio of highly variable genes;
    num_smooth_iterations:For the set smoothing parameters, generally, we recommend the second parameter to be 0;
    is_distribution:Whether to enable distributed loss;
    initial_alpha:Initialization of the smoothing coefficient.
    -------

    """
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:

        adata_Vars = adata[:, adata.var['highly_variable']]
        dense_X = adata_Vars.X.toarray()
        # corr_matrix = np.corrcoef(dense_X)
        corr_matrix1 = np.corrcoef(dense_X.T)
        mean_corr_per_gene = np.mean(corr_matrix1, axis=1)
        mean_corr_per_gene = np.abs(mean_corr_per_gene)
        sorted_mean_corr = np.sort(mean_corr_per_gene)
        index_percent = int(len(sorted_mean_corr) * keep_percent)
        threshold = sorted_mean_corr[index_percent]
        keep_genes = np.where(mean_corr_per_gene < threshold)[0]
        adata_Vars = adata_Vars[:, keep_genes]
        low_corr_genes = np.where(mean_corr_per_gene >= threshold)[0]

        # draw the result
        plt.figure(figsize=(3, 2))
        n_keep, bins_keep, patches_keep = plt.hist(mean_corr_per_gene[keep_genes], bins=20, color='#8491B4',
                                                   edgecolor='black', alpha=0.7)
        n_remove, bins_remove, patches_remove = plt.hist(mean_corr_per_gene[low_corr_genes], bins=20, color='#8491B4',
                                                         edgecolor='black', alpha=0.3, linestyle='--')
        plt.xlabel('The mean correlation of genes')
        plt.ylabel('Frequency')
        plt.title('Histogram of Mean Correlation of Genes')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    else:
        adata_Vars = adata
        print("'highly_variable' not in adata.var.columns!")

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

    data = Transfer_pytorch_Data(adata_Vars)
    model = FlatST(hidden_dims=[data.x.shape[1]] + hidden_dims, num_smooth_iterations=num_smooth_iterations,
                   initial_alpha=initial_alpha).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float('inf')
    patience_counter = 0
    # entropy_weight = 0.0001
    # Recession factor
    annealing_factor = 0.95

    # loss_list = []
    if is_distribution == 0:
        for epoch in tqdm(range(1, n_epochs + 1)):
            model.train()
            optimizer.zero_grad()
            z, out = model(data.x, data.edge_index)
            loss = F.mse_loss(data.x, out) + 1e-3 * torch.norm(model.conv1.lin_src, p=2)  # Add L2 regularization
            # loss_list.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            # Decide whether to enable the early stop mechanism based on the is_early_stopping parameter
            if is_early_stopping == True:
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f'Early stopping at epoch {epoch}')
                        break
    else:
        for epoch in tqdm(range(1, n_epochs + 1)):
            model.train()
            optimizer.zero_grad()
            # z, out = model(data.x, data.edge_index)
            # loss = F.mse_loss(data.x, out) + 1e-3 * torch.norm(model.conv1.lin_src, p=2)  # Add L2 regularization
            # target_distribution = torch.randn(data.x.size(0), hidden_dims[1]).to(device)
            z, out = model(data.x, data.edge_index)
            target_distribution = torch.FloatTensor(z.to('cpu').detach().numpy()).to(device)
            # entropy = -torch.sum(F.softmax(target_distribution, dim=1) * F.log_softmax(target_distribution, dim=1), dim=1).mean()
            # kl_loss = F.kl_div(F.softmax(z, dim=1), F.log_softmax(target_distribution, dim=1), reduction='batchmean')
            kl_loss = F.kl_div(F.log_softmax(z, dim=1), F.softmax(target_distribution, dim=1), reduction='batchmean')
            # Combine the reconstruction error and KL divergence loss
            # print(F.mse_loss(data.x, out), kl_loss, 1e-3*torch.norm(model.conv1.lin_src, p=2))
            loss = F.mse_loss(data.x, out) + is_distribution * kl_loss + 1e-3 * torch.norm(model.conv1.lin_src,
                                                                                           p=2)  # + entropy_weight * entropy
            # Dynamic adjustment of weights
            if epoch % 100 == 0:  # Adjust every 100 epochs
                is_distribution *= annealing_factor

            # # # loss_list.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            if is_early_stopping == True:
                # Early stop mechanism
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f'Early stopping at epoch {epoch}')
                        break

    model.eval()
    z, out = model(data.x, data.edge_index)

    FlatST_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = FlatST_rep

    if save_loss:
        adata.uns['FlatST_loss'] = loss
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX < 0] = 0
        adata.layers['FlatST_ReX'] = ReX

    return adata