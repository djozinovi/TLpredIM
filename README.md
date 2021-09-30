# TLpredIM


A repository of the paper: "Transfer learning: Improving neural network based prediction of earthquake ground shaking for an area with insufficient training data" (https://arxiv.org/abs/2105.05075, under revision in Geophysical Journal international)

The CW dataset is available at https://doi.org/10.5281/zenodo.4756985. 

The code is available in the file train.py (please look at the comments in the file for details).

The following pre-trained models are available in the repository:  modelParr10s.h5 (multi-station IM prediction model trained on CI), 10secMagCNNSTEAD.h5 (single-station magnitude determination trained on STEAD data), centIT10sSTead.hdf5 (multi-station IM prediction model trained on CI using a pre-trained model 10secMagCNNSTEAD.h5).
