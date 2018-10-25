# LadderNet
* This is repository for my paper [LadderNet: Multi-path networks based on U-Net for medical image segmentation](https://arxiv.org/abs/1810.07810). <br>
* This implementation is based on [orobix implementation](https://github.com/orobix/retina-unet). Main difference is the structure of the model.

# Requirement
* Python3 
* PyTorch 0.4
* configparser

# How to run
* run <b>"python prepare_datasets_DRIVE.py"</b> to generate hdf5 file of training data
* run <b>"cd src"</b>
* run <b>"python retinaNN_training.py"</b> to train
* run <b>"python retinaNN_predict.py"</b> to test

# Parameter defination
* parameters (path, patch size, et al.) are defined in "configuration.txt"
* training parameters are defined in src/retinaNN_training.py line 49 t 84 with notes "=====Define parameters here ========="
