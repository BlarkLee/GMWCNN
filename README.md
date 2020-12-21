# GMWCNN
Gated Multi-Level Wavelet Convolutional Neural Networks in Semantic Segmentation, the three authors Logan Lawrence, Runfa Li, Zihan Li are contributed equally to this work.

## Dataset Preparation
We totally use Cityscapes Dataset for this work. Specifically we use gt_Fine for labelset and leftImg8bit for imageset, from Cityscapes official website. To implement our code, please first prepare the dataset, then go to the directory /configs/ to change the data path of all yaml files.

## Training
For training a specific model, use train_%%%.py, our code use multi-gpu for training, a directory /run/ will be built automatically once start training, and the checkpoint will be saved there

## Testing
To test a specific model, use validate_mwcnn.py, but make sure to change the configs file to the model you are testing at line 103, and modify the path to the checkpoint at line 111.

## Visulization
To visulize the segmentation map of the prediction, using visualize_mwcnn.ipynb, follow the instruction in the jupyter notebook file.


