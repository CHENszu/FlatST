import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from .gat_conv import GATConv


class FlatST(torch.nn.Module):
    def __init__(self, hidden_dims, num_smooth_iterations, initial_alpha):
        super(FlatST, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

        # Initialize two learnable smoothing intensity parameters
        self.num_smooth_iterations = num_smooth_iterations
        self.smooth_alpha_1 = nn.Parameter(torch.tensor(initial_alpha))
        self.smooth_alpha_2 = nn.Parameter(torch.tensor(initial_alpha))
        # self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, features, edge_index):  # relu/leakyrelu/elu/selu/prelu/gelu/mish/sigmoid/tanh
        h1 = F.selu(self.conv1(features, edge_index, attention=True))
        h2 = self.conv2(h1, edge_index, attention=True)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.selu(self.conv3(h2, edge_index, attention=True))
        h4 = self.conv4(h3, edge_index, attention=False)

        if self.num_smooth_iterations[0] > 0 and self.num_smooth_iterations[1] > 0:
            smoothed_h2 = self.smooth_features_learnable(h2, edge_index)
            smoothed_h2 = self.smooth_features_dynamic(smoothed_h2, edge_index)
        elif self.num_smooth_iterations[0] > 0 and self.num_smooth_iterations[1] == 0:
            smoothed_h2 = self.smooth_features_learnable(h2, edge_index)
        elif self.num_smooth_iterations[0] == 0 and self.num_smooth_iterations[1] > 0:
            smoothed_h2 = self.smooth_features_dynamic(h2, edge_index)
        else:
            smoothed_h2 = h2
        return smoothed_h2, h4

    def smooth_features_learnable(self, features, edge_index):
        """
        Smooth the features to make the features of adjacent points as similar as possible
        """
        # Cache degree information
        if not hasattr(self, 'deg_inv_sqrt'):
            row, col = edge_index
            deg = torch.zeros(features.size(0), dtype=features.dtype, device=features.device)
            deg.scatter_add_(0, row, torch.ones_like(row, dtype=features.dtype, device=features.device))
            deg_inv_sqrt = deg.pow(-0.6)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            self.deg_inv_sqrt = deg_inv_sqrt

        # Smoothness of different scales
        smoothed_features_1 = features
        smoothed_features_2 = features

        for i in range(self.num_smooth_iterations[0]):
            row, col = edge_index
            # Construct a smoothing matrix using the first parameter
            norm_1 = (self.deg_inv_sqrt[row] * self.deg_inv_sqrt[col] * self.smooth_alpha_1)
            adj_1 = torch.sparse_coo_tensor(edge_index, norm_1, (features.size(0), features.size(0)),
                                            dtype=features.dtype, device=features.device)
            smoothed_features_1 = torch.sparse.mm(adj_1, smoothed_features_1)

            # Construct a smoothing matrix using the second parameter
            norm_2 = (self.deg_inv_sqrt[row] * self.deg_inv_sqrt[col] * self.smooth_alpha_2 * 2)
            adj_2 = torch.sparse_coo_tensor(edge_index, norm_2, (features.size(0), features.size(0)),
                                            dtype=features.dtype, device=features.device)
            smoothed_features_2 = torch.sparse.mm(adj_2, smoothed_features_2)

        # Learnable weighted parameters
        # print(self.smooth_alpha_1,self.smooth_alpha_2)
        # weight = self.weight
        weight = 0.5
        # print(weight)
        # Integrate the smoothing results of different scales
        smoothed_features = weight * smoothed_features_1 + (1 - weight) * smoothed_features_2

        return smoothed_features

    def smooth_features_dynamic(self, features, edge_index):
        """
        Smooth the features to make the features of adjacent points as similar as possible.
        Improvement idea: Use multiple iterations of smoothing operations, dynamically adjust the smoothing intensity based on the node degree,
        and enhance the smoothing effect
        """
        smoothed_features = features
        for _ in range(self.num_smooth_iterations[1]):
            row, col = edge_index
            deg = torch.zeros(features.size(0), dtype=features.dtype, device=features.device)
            deg.scatter_add_(0, row, torch.ones_like(row, dtype=features.dtype, device=features.device))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

            # Dynamically adjust the smoothing intensity according to the node degree
            alpha = 1.0 / (1.0 + torch.log(deg + 1))
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * alpha[row]

            adj = torch.sparse_coo_tensor(edge_index, norm, (features.size(0), features.size(0)), dtype=features.dtype,
                                          device=features.device)
            smoothed_features = torch.sparse.mm(adj, smoothed_features)
        return smoothed_features