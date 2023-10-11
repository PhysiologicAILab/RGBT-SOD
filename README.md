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




