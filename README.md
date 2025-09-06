# FlatST
Here we propose FlatST, a graph attention autoencoder framework with dual smoothing properties: a multi-scale learnable smoothing mechanism and a node-degree adaptive smoothing mechanism.
![Graph structure preprocessing](https://s21.ax1x.com/2025/09/02/pVgnprT.png)
# Overview
From the Graph Autoencoder (GAE) to the Graph Attention Encoder (GAT), on the basis of these two frameworks, many outstanding works have emerged in the research of spatial transcriptomics. Compared with other methods for establishing complex models to optimize low-dimensional latent matrices, we propose a very simple optimization method: FlatST. FlatST updates the low-dimensional latent matrix based on GAT through a dual smoothing mechanism. Specifically, we first solve the smoothing coefficients based on encoding and decoding, and then construct the smoothing matrix in combination with the number of spots neighbors. We have compared FlatST with 9 methods([Louvain](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0#citeas),) on 28 datasets to demonstrate its superiority in clustering and denoising.
# Software dependencies
FlatST is developed based on pytorch.
# Installation
cd xx/FlatST-main  
python setup.py build  
python setup.py install
# Tutorial
If you are interested in FlatST, you can click [here](https://flatst-tutorial.readthedocs.io/en/latest/) to go to the tutorial of FlatST.In our experiment, we found that the hyperparameters of FlatST play a decisive role in the experimental results. You can refer to the parameters we provide to run it.  
All data used in this article we have converted to .h5ad format and it has been uploaded to [Google](https://drive.google.com/drive/folders/1WmBwN9hPjBlyJsMhX62u0gO7vzeOTH1q?dmr=1&ec=wgc-drive-globalnav-goto).You can use them for free.
# Citation
The paper is being submitted...  
FlatST: A Smoothed Attention Graph Autoencoder for Multi-Task Spatial Transcriptomic Analysis—Denoising, Batch Effect Mitigation, Domain Identification, and 3D Extraction
# Special Acknowledgements
Here, we would like to express our special gratitude to the author of STAGATE for his contributions. If you are interested, please refer to [STAGATE](https://www.nature.com/articles/s41467-022-29439-6#citeas)
