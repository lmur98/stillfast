
import torch
import os

import torch.nn as nn
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from stillfast.datasets import Ego4dShortTermAnticipationStillVideo #Import the dataset
from stillfast.config.defaults import get_cfg #Import the configurations
import stillfast.datasets.loader as loader #Import the dataloader
from stillfast.models.stillfast import StillFast #Import the model
import loren_utils as my_utils #Import my utils

#A. Load the configurations and setup the logs
experiment_name = "baseline_0" #"faster_rcnn_DINO_single_scale"
gpu_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', gpu_device)
writer_train = SummaryWriter(log_dir='/home/lmur/hum_obj_int/tensorboard_logs' + experiment_name + '/train')
writer_val = SummaryWriter(log_dir='/home/lmur/hum_obj_int/tensorboard_logs' + experiment_name + '/val')
cfg_file = '/home/lmur/hum_obj_int/stillfast/configs/sta/STILL_FAST_R50_X3DM_EGO4D_v2.yaml'
cfg = get_cfg() # Setup cfg.
cfg.merge_from_file(cfg_file)
#print('The configurations are:', cfg)

#B. Read the dataset
train_dataset = Ego4dShortTermAnticipationStillVideo(cfg, split = 'train')
val_dataset = Ego4dShortTermAnticipationStillVideo(cfg, split = 'val')
print('The len of the training dataset is:', train_dataset.__len__(), ', test:', val_dataset.__len__())

#C. Create the dataloaders (So far we are with a single devide)
train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=2, shuffle=True, 
                                               num_workers=2, pin_memory=True, 
                                               drop_last=True, collate_fn=loader.get_collate(cfg.TASK))
val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                               batch_size=2, shuffle=False, 
                                               num_workers=2, pin_memory=True, 
                                               drop_last=True, collate_fn=loader.get_collate(cfg.TASK))
print('The number of batches in the train datast is:', len(train_dataloader), 'validating:', len(val_dataloader))

#E. Define the model and the optimizer
model = StillFast(cfg)
model.to(gpu_device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
lr_policy = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
#We have to add the warmup, though with dino it might not be necesary
n_epochs = 10
init_epoch = 0

for epoch in range(init_epoch, n_epochs):
    #-------------------------Train one epoch--------------------------------
    model.train()
    bar = tqdm(total = len(train_dataloader))
    
    for batch, sample in enumerate(train_dataloader):
        #Send data to gpu!
        high_res_img = my_utils.list_to_gpu(sample['still_img'], gpu_device)
        low_res_video = my_utils.list_to_gpu(sample['fast_imgs'], gpu_device)
        targets = my_utils.targets_to_gpu(sample['targets'], gpu_device) 
        gpu_batch = {'still_img': high_res_img, 'fast_imgs': low_res_video, 'targets': targets}
        #Training loop
        output = model(gpu_batch)
        total_loss = sum(output.values())
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_policy.step()
        write_train_metrics(writer_train, output, total_loss, n_iter = epoch * len(train_dataloader) + batch)
        #Log the results

        
        print(pred, 'que got itt')
        break
    break
