**This work is an implementation of the paper "Survival Modeling from Whole Slide Images via Patch-Level Graph Clustering and Mixture Density Experts"(https://arxiv.org/abs/2507.16476). It present a modular framework for predicting cancer-specific survival from whole-slide pathology images (WSIs). The approach integrates four main components: (i) Quantile-Based  Patch Filtering, which employs quantile-based thresholding to identify prognostically informative tissue regions; (ii) Graph Regularized Patch Clustering using $k$-NN graph to model phenotype-level heterogeneity through spatial–morphological coherence; (iii) Hierarchical Feature Aggregation for learning intra- and inter-cluster dependencies; and (iv) an Expert-Guided Mixture Density Modeling module to estimate complex survival distributions using Gaussian distributions. The proposed model achieves a concordance index of $0.653±0.037$ on TCGA-LUAD, $0.719±0.011$ on TCGA-KIRC, and $0.733±0.037$ on TCGA-BRCA, surpassing current state-of-the-art methods.**

## 🔄 Pipeline Steps for creating the Virtual Environment

1. Create and activate environment
   ```
   conda create -n sgcmde python=3.9 

   conda activate sgcmde
   ```

2. Install PyTorch (CUDA 11.3 build)
   ```
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

3. Install RAPIDS cuDF/cuML
   ```
   pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11==23.10.0 cuml-cu11==23.10.0
   ```

4. Install other Python dependencies
   ```
   pip install tqdm lifelines munch tensorboardX einops h5py seaborn
   ```
5. While running the train script, some errors may appear after creating the virtual environment. If errors appear then follow the steps in the text file 'changes_in_virtual_env_types_python_file.txt' right after the creation of the         virtual environment.
   
## 📂 Data Preparation

1. TCGA data:
Download diagnostic WSIs and corresponding clinical metadata from TCGA(https://portal.gdc.cancer.gov/).

2. Patch extraction:
Use the CLAM WSI processing tool(https://github.com/mahmoodlab/CLAM) to crop WSIs into 256×256 patches at 40× magnification.

3. Feature extraction:
Extract patch-level features with a ViT(https://github.com/lunit-io/benchmark-ssl-pathology#pre-trained-weights) model pretrained on large-scale WSI collections using self-supervised learning.

4. Annoation files and folder structure: 
Prepare the annotaion file as below. Prepare you own 'wsi_annos_vit-s-dino-p16.txt' file.

## 📂 Dataset Structure

```
data/
├── kirc/
│   ├── 5fold_wsi-rnaseq/
│   │   ├── fold1/
│   │   │   ├── train.txt
│   │   │   └── val.txt
│   │   ├── fold2/
│   │   ├── fold3/
│   │   ├── fold4/
│   │   └── fold5/
│   ├── clinical.csv
│   └── wsi_annos_vit-s-dino-p16.txt
└── luad/
    ├── 5fold_wsi-rnaseq/
    │   ├── fold1/
    │   │   ├── train.txt
    │   │   └── val.txt
    │   ├── fold2/
    │   ├── fold3/
    │   ├── fold4/
    │   └── fold5/
    ├── clinical.csv
    └── wsi_annos_vit-s-dino-p16.txt
```
    
## 🧪  Train the model
python train.py --config configs/luad_sgcmde.yaml

## 🌡️ Final Plot
python plot.py --config configs/luad_sgcmde.yaml
<p align="center">
  <img src="plots/luad.png" alt="Centered Image" width="500"/>
</p>

## 📚 Acknowledgements
Thanks Zekang Yang, Hong Liu, and Xiangdong Wang for the public access of the codebase https://github.com/yang-ze-kang/SCMIL
