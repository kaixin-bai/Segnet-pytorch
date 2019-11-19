from segnet import SegNet as segnet
import torch
import torch.nn as nn
from data_controller import SegDataset
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.ma as ma
from loss import Loss
import sys
sys.path.append("..")
from utils import setup_logger
import time 

model = segnet()
model = model.cuda()
model = nn.DataParallel(model, device_ids=[0])
model.load_state_dict(torch.load("./trained_models/model_20_0.11692727969866246.pth"))
model.eval()

test_dataset = SegDataset("../datasets/ycb/YCB_Video_Dataset", '../datasets/ycb/dataset_config/test_data_list.txt', False, 1000)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=24)
criterion = Loss()
test_all_cost=0
test_time = 0	
epoch = 0

debug_mode = 1
#logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
for j, data in enumerate(test_dataloader, 0):
    rgb, target = data
    print("img {0}".format(j))
    npimg_ori = torch.squeeze(rgb).numpy()  #(3,480,640)
    out_ori = np.transpose(npimg_ori, (1, 2, 0))    #(480,640,3)
    plt.imshow(out_ori/1000 )
    plt.title('out_ori')
    plt.show()
    rgb, target = Variable(rgb).cuda(), Variable(target).cuda()
    print('debug', rgb.shape)   #debug torch.Size([1, 3, 480, 640])
    semantic = model(rgb)   # semantic.shape=torch.Size([1,22,480,640])
    
    npimg = torch.squeeze(semantic).detach().cpu().numpy()  #shape={tuple}:(22,480,640)
    for i in range(22):
        print('debug npimg[i]',npimg[i])
        npimg_mask = npimg[i].copy()
        npimg_mask = np.where(npimg_mask<0, False, True)
        plt.imshow(npimg_mask)
        plt.title('npimg_mask')
        plt.show()
        plt.imshow(npimg[i])
        plt.title('bkx:debug')
        plt.show()
    out = np.transpose(npimg, (1, 2, 0))
    # print('out.shape',out.shape)    #out.shape (480, 640, 22)

    crops = []
    for i in range(22):
        if i == 0:
            arr = out[:,:,i]<0  # true false map
        else:
            arr = out[:,:,i]>0
        height = np.nonzero((arr).any(axis=0))[0]
        # print(len(height))
        if(len(height)<30):
            continue

        height_points = [height[0],height[-1]]
        width = np.nonzero(arr.any(axis=1))[0]
        if(len(width)<30):
            continue
        width_points = [width[0],width[-1]]
        crops.append([i,width_points,height_points])
        if debug_mode == 1:
            # print(crops[-1])
            fig,ax = plt.subplots(1)
            # Display the image
            ax.imshow(out[:,:,i],cmap='viridis')
            # Create a Rectangle patch
            rect = patches.Rectangle((height_points[0],width_points[0]),height_points[1]-height_points[0],width_points[1] - width_points[0],linewidth=1,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.show()

    # print(crops)
    for i in range(len(crops)):
        print(i)
        crop = crops[i]
        # print('crop',crop)
        plt.imshow(out_ori[crop[1][0]:crop[1][1],crop[2][0]:crop[2][1],:]/1000)
        plt.title('croped image')
        plt.show()
    semantic_loss = criterion(semantic, target)
    test_all_cost += semantic_loss.item()
    test_time += 1
    print('Batch {0} CEloss {1}'.format(test_time, semantic_loss.item()))

test_all_cost = test_all_cost / test_time
logger.info('Test Finish Avg CEloss: {0}'.format(test_all_cost))

