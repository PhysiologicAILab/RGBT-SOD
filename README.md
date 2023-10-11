# RGB-T-segmentation


  
## Datasets for RGB-T SOD

* We follow the work 'LSNet: Lightweight Spatial Boosting Network for Detecting Salient Objects in RGB-Thermal Images', IEEE TIP, 2023.(https://github.com/zyrant/LSNet)

* Training and Testing Dataset: https://drive.google.com/file/d/1vjdD13DTh9mM69mRRRdFBbpWbmj6MSKj/view



## Requirement
* Python 3.7
* PyTorch 1.13.1
* torchvision
* numpy
* Pillow
* Cython
* mmcv



## Training Model on VT5000

### Fully supervised Training
* 1.Run train_sod.py


## Testing on VT5000,VT1000,VT821
* 1.Set the path of testing sets in test_sod.py    
* 2.Run test_sod.py (can generate the segmentation maps) : python test_sod.py --exp_name='(checkpoint folder)'
* 3.Run test_score.py (change root, method_name), record MAE, E, F ,S.


## Datasets for PST900
* We use PST900 (https://github.com/ShreyasSkandanS/pst900_thermal_rgb) as an example to build a pipeline for RGB-T segmentation

* Training and Testing Dataset: https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view

* Build a new folder "data" and unzip the dataset under it, the path of dataset should be './data/PST900_RGBT_Dataset/..'


## Training Model on PST900

### Fully supervised Training
* 1.Set the path of training sets in train_pst900.py 
* 2.Run train_pst900.py


### Training with self-supervised loss
* 1.Add two outputs (c5, t5) from the model FPN.py 
* 2.Add the loss_fn in train_pst900.py
* 3.Run train_pst900.py

## Testing on PST900
* 1.Set the path of testing sets in test_pst900.py    
* 2.Run test.py (can generate the segmentation maps and original RGB images)  
* 3.Compare the results of miou
