# FlatST
Here we propose FlatST, a graph attention autoencoder framework with dual smoothing properties: a multi-scale learnable smoothing mechanism and a node-degree adaptive smoothing mechanism.
![Graph structure preprocessing](https://s21.ax1x.com/2025/09/02/pVgnprT.png)
# Overview
From the Graph Autoencoder (GAE) to the Graph Attention Encoder (GAT), on the basis of these two frameworks, many outstanding works have emerged in the research of spatial transcriptomics. Compared with other methods for establishing complex models to optimize low-dimensional latent matrices, we propose a very simple optimization method: FlatST. FlatST updates the low-dimensional latent matrix based on GAT through a dual smoothing mechanism. Specifically, we first solve the smoothing coefficients based on encoding and decoding, and then construct the smoothing matrix in combination with the number of spots neighbors. We have compared FlatST with 8 methods: [Louvain](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1382-0#citeas), [stLearn](https://www.nature.com/articles/s41467-023-43120-6#citeas), [SpaGCN](https://www.nature.com/articles/s41592-021-01255-8), [SpaceFlow](https://www.nature.com/articles/s41467-022-31739-w), [SEDR](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-024-01283-x), [STAGATE](https://www.nature.com/articles/s41467-022-29439-6), [GraphST](https://www.nature.com/articles/s41467-023-36796-3) and [STAIG](https://www.nature.com/articles/s41467-025-56276-0) on 29 datasets to demonstrate its superiority in clustering and denoising.
# Software dependencies
FlatST is developed based on pytorch, you can quickly install the relevant python modules by naming them as follows:  
```bash
pip install -r requirements.txt
pip install -e .  
```

# Installation
The configuration method of FlatST is very simple:  
step1:  
```bash
git clone https://github.com/CHENszu/FlatST.git
```
step2:  
```bash
unzip FlatST*.zip  
cd FlatST-main
```
step3:  
```bash
python setup.py build  
python setup.py install
```
# Tutorial
If you are interested in FlatST, you can click [here](https://flatst-tutorial.readthedocs.io/en/latest/) to go to the tutorial of FlatST.In our experiment, we found that the hyperparameters of FlatST play a decisive role in the experimental results. You can refer to the parameters we provide to run it.  
 
All datasets used in this article we have converted to .h5ad format and it has been uploaded to [Google Drive](https://drive.google.com/drive/folders/1WmBwN9hPjBlyJsMhX62u0gO7vzeOTH1q?usp=drive_link).You can use them for free.  

To reduce the runtime of FlatST on large-scale datasets, we optimized the construction of cellular spatial neighborhoods in FlatST using a k-d tree. You can refer to the file at https://github.com/CHENszu/FlatST/blob/main/FlatST/utils_kd.py.
# Idea
When running other methods, we found the following three main problems:  
+ The clustering results are scattered and discontinuous.  
+ The boundary recognition is uneven.  
+ The domain was unexpectedly divided.   
<img width="2135" height="591" alt="图片2" src="https://github.com/user-attachments/assets/4fa3790a-adae-4f03-b089-9a575cb72329" />  
  
To ensure that the clustering presents smooth and continuous features, we were inspired by the **SmoothQuant** technique of large models and achieved the desired results. The experimental results show that only one smoothing coefficient can achieve an improvement of more than **10%** in clustering accuracy on the DLPFC dataset. Please note: We have added distribution loss to improve the clustering accuracy of repeated experiments. However, if your hardware device has a small memory, please set the parameter **is_distribution=0.0**.
# Citation
The paper is being submitted...  
A multitask spatial transcriptomics analysis tool using smoothed attention graph autoencoder.
# Special Acknowledgements
Here, we would like to express our special gratitude to the author of STAGATE for his contributions. If you are interested, please refer to [STAGATE](https://www.nature.com/articles/s41467-022-29439-6#citeas)
