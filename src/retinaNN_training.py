###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
from six.moves import configparser
import torch.backends.cudnn as cudnn

import sys
sys.path.insert(0, '../lib/')
from lib.help_functions import *

#function to obtain data for training/testing (validation)
from lib.extract_patches import get_data_training

import os

from losses import *

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#=========  Load settings from Config file
config = configparser.RawConfigParser()
config.read('../configuration.txt')

#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#========== Define parameters here =============================
# log file
if not os.path.exists('./logs'):
    os.mkdir('logs')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 200

val_portion = 0.1

lr_epoch = np.array([50,150,total_epoch])
lr_value= np.array([0.01,0.001,0.0001])

layers = 4
filters = 10

from LadderNetv65 import LadderNetv6

net = LadderNetv6(num_classes=2,layers=layers,filters=filters,inplanes=1)
print("Toral number of parameters: "+str(count_parameters(net)))

check_path = 'LadderNetv65_layer_%d_filter_%d.pt7'% (layers,filters) #'UNet16.pt7'#'UNet_Resnet101.pt7'

resume = True

criterion = LossMulti(jaccard_weight=0)
#criterion = CrossEntropy2d()

#optimizer = optim.SGD(net.parameters(),
#                     lr=lr_schedule[0], momentum=0.9, weight_decay=5e-4, nesterov=True)

optimizer = optim.Adam(net.parameters(),lr=lr_value[0])

#============ Load the data and divided in patches
patches_imgs_train, patches_masks_train = get_data_training(
    DRIVE_train_imgs_original = path_data + config.get('data paths', 'train_imgs_original'),
    DRIVE_train_groudTruth = path_data + config.get('data paths', 'train_groundTruth'),  #masks
    patch_height = int(config.get('data attributes', 'patch_height')),
    patch_width = int(config.get('data attributes', 'patch_width')),
    N_subimgs = int(config.get('training settings', 'N_subimgs')),
    inside_FOV = config.getboolean('training settings', 'inside_FOV') #select the patches only inside the FOV  (default == True)
)

class TrainDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs,patches_masks_train):
        self.imgs = patches_imgs
        self.masks = patches_masks_train

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        tmp = self.masks[idx]
        tmp = np.squeeze(tmp,0)
        return torch.from_numpy(self.imgs[idx,...]).float(), torch.from_numpy(tmp).long()

val_ind = random.sample(range(patches_masks_train.shape[0]),int(np.floor(val_portion*patches_masks_train.shape[0])))

train_ind =  set(range(patches_masks_train.shape[0])) - set(val_ind)
train_ind = list(train_ind)

train_set = TrainDataset(patches_imgs_train[train_ind,...],patches_masks_train[train_ind,...])
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=4)

val_set = TrainDataset(patches_imgs_train[val_ind,...],patches_masks_train[val_ind,...])
val_loader = DataLoader(val_set, batch_size=batch_size,
                          shuffle=True, num_workers=4)

#========= Save a sample of what you're feeding to the neural network ==========
N_sample = min(patches_imgs_train.shape[0],40)
visualize(group_images(patches_imgs_train[0:N_sample,:,:,:],5),'../'+name_experiment+'/'+"sample_input_imgs")#.show()
visualize(group_images(patches_masks_train[0:N_sample,:,:,:],5),'../'+name_experiment+'/'+"sample_input_masks")#.show()

best_loss = np.Inf

# create a list of learning rate with epochs
lr_schedule = np.zeros(total_epoch)
for l in range(len(lr_epoch)):
    if l ==0:
        lr_schedule[0:lr_epoch[l]] = lr_value[l]
    else:
        lr_schedule[lr_epoch[l-1]:lr_epoch[l]] = lr_value[l]

if device == 'cuda':
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+check_path)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    IoU = []

    # get learning rate from learing schedule
    lr = lr_schedule[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("Learning rate = %4f\n" % lr)

    IU = []
    # train network
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print("Epoch %d: Train loss %4f\n" % (epoch, train_loss / np.float32(len(train_loader))))

def test(epoch, display=False):
    global best_loss
    net.eval()
    test_loss = 0
    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()

        print(
            'Valid loss: {:.4f}'.format(test_loss))
    # Save checkpoint.
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + check_path)
        best_loss = test_loss

for epoch in range(start_epoch,total_epoch):
    train(epoch)
    test(epoch,False)
