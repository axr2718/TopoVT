# Topo-VT

## Paper
Implementation of "Topology Meets Deep Learning for Breast Cancer Detection" using PyTorch. Submitted to CVPR, 2025.

## Method
Uses the betti vectors $B_0$, which correspond to components of cubical filtration in images, and $B_1$, which correspond to loops of cubical filtration in images, in the vision transformer SwinV2 to increase generalization. This works by creating transformer encoders for the betti vectors, both of which are independent, then applying cross-attention between all the layers of SwinV2 and the transformer encoder of betti vector embeddings. 

Add image here, Brighton.

The datasets used were BUSI-3, BUSI-2 (2 labels instead of 3), BusBra, and Mendeley. All of these datasets are X-ray images of breast cancer. (Add links to them, Brighton)

## Results
Combining betti vector embeddings with SwinV2 cross-attention showed promise in increasing generalization, specifically in the BusBra dataset. It saw small gains in BUSI-3. However, the encoders were not tuned to be the best, so more experimentation will be needed.
