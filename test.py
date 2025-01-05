import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io

from time import time
'''
from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
'''
from networks.dinknet import DinkNet34_WithBranch
# BATCHSIZE_PER_CARD = 4

class TTAFrame():
    def __init__(self, net):
        self.net = net()
        #self.net = net().cuda()
        #self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))

    def test_one_img_without_TTAFrame(self, path, evalmode=True):
        if evalmode:
            self.net.eval()
        img = cv2.imread(path)
        img = np.expand_dims(np.array(img),axis=0)
        img = img.transpose(0,3,1,2)
        img = np.array(img, np.float32)/255.0 * 3.2 -1.6
        img = torch.Tensor(img)
        #img = V(torch.Tensor(img).cuda())
        mask, prob, posi, link = self.net.forward(img)
        mask = mask.squeeze().cpu().data.numpy()
        prob = prob.squeeze(0).cpu().data.numpy()
        posi = posi.squeeze().cpu().data.numpy()
        link = link.squeeze().cpu().data.numpy()
        return mask, prob, posi, link

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        #self.net.load_state_dict(torch.load(path))
        
source = 'dataset/test/sat/'
#source = 'dataset/train/sat/'
val = os.listdir(source)
solver = TTAFrame(DinkNet34_WithBranch)
solver.load('weights/log04_dink34.th')
tic = time()
target = 'dataset/test/pred/'
#target = 'dataset/train/pred/'
os.makedirs(target, exist_ok=True)
os.makedirs(target+'mask/', exist_ok=True)
os.makedirs(target+'mat/', exist_ok=True)
os.makedirs(target+'prob_posi_link/', exist_ok=True)
os.makedirs(target+'merge/', exist_ok=True)

total_accuracy = 0.0
num_samples = 0

for img_id,name in enumerate(val):
    if img_id % 10 == 0:
        print(img_id/10, '    ','%.2f'%(time()-tic))
    
    mask, prob, posi, link = solver.test_one_img_without_TTAFrame(source + name)
    mask[mask > 0.5] = 255
    mask[mask <= 0.5] = 0
    prob[prob > 0.5] = 1
    prob[prob <= 0.5] = 0
    link[link > 0.5] = 1
    link[link <= 0.5] = 0
    posi_final = np.zeros((2,64,64), np.uint64)
    for i in range(0, 64):
        for j in range(0, 64):
            if prob[0,i,j] == 1:
                posi_final[0,i,j] = int(posi[0,i,j]*15+0.5)+i*16
                posi_final[1,i,j] = int(posi[1,i,j]*15+0.5)+j*16
                if link[0,i,j]==1:
                    if i-1<0 or prob[0,i-1,j]!=1:
                        link[0,i,j]=0
                if link[1,i,j]==1:
                    if i-1<0 or j+1>=64 or prob[0,i-1,j+1]!=1:
                        link[1,i,j]=0
                if link[2,i,j]==1:
                    if j+1>=64 or prob[0,i,j+1]!=1:
                        link[2,i,j]=0
                if link[3,i,j]==1:
                    if i+1>=64 or j+1>=64 or prob[0,i+1,j+1]!=1:
                        link[3,i,j]=0
                if link[4,i,j]==1:
                    if i+1>=64 or prob[0,i+1,j]!=1:
                        link[4,i,j]=0
                if link[5,i,j]==1:
                    if i+1>=64 or j-1<0 or prob[0,i+1,j-1]!=1:
                        link[5,i,j]=0
                if link[6,i,j]==1:
                    if j-1<0 or prob[0,i,j-1]!=1:
                        link[6,i,j]=0
                if link[7,i,j]==1:
                    if i-1<0 or j-1<0 or prob[0,i-1,j-1]!=1:
                        link[7,i,j]=0

                if link[1,i,j] == 0 and i-1>=0 and j+1<64:
                    if prob[0,i-1,j+1] == 1 and prob[0,i-1,j] == 0 and prob[0,i,j+1] == 0:
                        link[1,i,j] = 1
                if link[3,i,j] == 0 and i+1<64 and j+1<64:
                    if prob[0,i+1,j+1] == 1 and prob[0,i,j+1] == 0 and prob[0,i+1,j] == 0:
                        link[3,i,j] = 1
                if link[5,i,j] == 0 and i+1<64 and j-1>=0:
                    if prob[0,i+1,j-1] == 1 and prob[0,i+1,j] == 0 and prob[0,i,j-1] == 0:
                        link[5,i,j] = 1
                if link[7,i,j] == 0 and i-1>=0 and j-1>=0:
                    if prob[0,i-1,j-1] == 1 and prob[0,i,j-1] == 0 and prob[0,i-1,j] == 0:
                        link[7,i,j] = 1
            else:
                posi_final[0,i,j] = -1
                posi_final[1,i,j] = -1
                for k in range(0,8):
                    link[k,i,j] == -1

    posi_cal = posi_final.astype(np.int64)
    for i in range(0, 63):
        for j in range(0, 63):
            if prob[0,i,j] == 1 and prob[0,i+1,j] == 1 and prob[0,i,j+1] == 1 and prob[0,i+1,j+1] == 1 and (link[4,i,j] == 1 or link[0,i+1,j] == 1) and (link[2,i+1,j] == 1 or link[6,i+1,j+1] == 1) and (link[0,i+1,j+1] == 1 or link[4,i,j+1] == 1) and (link[6,i,j+1] == 1 or link[2,i,j] == 1):
                # i,j --> i+1,j
                a = (posi_cal[0,i,j]-posi_cal[0,i+1,j])*(posi_cal[0,i,j]-posi_cal[0,i+1,j]) + (posi_cal[1,i,j]-posi_cal[1,i+1,j])*(posi_cal[1,i,j]-posi_cal[1,i+1,j])
                # i+1,j --> i+1,j+1
                b = (posi_cal[0,i+1,j]-posi_cal[0,i+1,j+1])*(posi_cal[0,i+1,j]-posi_cal[0,i+1,j+1]) + (posi_cal[1,i+1,j]-posi_cal[1,i+1,j+1])*(posi_cal[1,i+1,j]-posi_cal[1,i+1,j+1])
                # i,j --> i,j+1
                c = (posi_cal[0,i,j]-posi_cal[0,i,j+1])*(posi_cal[0,i,j]-posi_cal[0,i,j+1]) + (posi_cal[1,i,j]-posi_cal[1,i,j+1])*(posi_cal[1,i,j]-posi_cal[1,i,j+1])
                # i,j+1 --> i+1,j+1
                d = (posi_cal[0,i,j+1]-posi_cal[0,i+1,j+1])*(posi_cal[0,i,j+1]-posi_cal[0,i+1,j+1]) + (posi_cal[1,i,j+1]-posi_cal[1,i+1,j+1])*(posi_cal[1,i,j+1]-posi_cal[1,i+1,j+1])
                if a>=b and a>=c and a>=d:
                    link[4,i,j] = 0
                    link[0,i+1,j] = 0
                    link[2,i+1,j] = 1
                    link[6,i+1,j+1] = 1
                    link[2,i,j] = 1
                    link[6,i,j+1] = 1
                    link[4,i,j+1] = 1
                    link[0,i+1,j+1] = 1
                if b>=a and b>=c and b>=d:
                    link[4,i,j] = 1
                    link[0,i+1,j] = 1
                    link[2,i+1,j] = 0
                    link[6,i+1,j+1] = 0
                    link[2,i,j] = 1
                    link[6,i,j+1] = 1
                    link[4,i,j+1] = 1
                    link[0,i+1,j+1] = 1
                if c>=a and c>=b and c>=d:
                    link[4,i,j] = 1
                    link[0,i+1,j] = 1
                    link[2,i+1,j] = 1
                    link[6,i+1,j+1] = 1
                    link[2,i,j] = 0
                    link[6,i,j+1] = 0
                    link[4,i,j+1] = 1
                    link[0,i+1,j+1] = 1
                if d>=a and d>=b and d>=c:
                    link[4,i,j] = 1
                    link[0,i+1,j] = 1
                    link[2,i+1,j] = 1
                    link[6,i+1,j+1] = 1
                    link[2,i,j] = 1
                    link[6,i,j+1] = 1
                    link[4,i,j+1] = 0
                    link[0,i+1,j+1] = 0
    
    mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
    mask = mask.astype(np.uint8)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(target+'mask/'+name[:-7]+'mask.png',mask.astype(np.uint8))

    mat_savepath = target+'mat/' + name[:-7] + 'mask.mat'
    scipy.io.savemat(mat_savepath, mdict={'if_key_points': prob, 'all_key_points_position': posi_final, 'anchor_link': link})
    
    new_img = np.zeros((1024,1024,3), np.uint8)

    for i in range(0, 64):
        for j in range(0, 64):
            for m in range(16*i,16*i+16):
                for n in range(16*j,16*j+16):
                    new_img[m][n] = [255,255,255]
                    if prob[0,i,j] == 0:
                        new_img[m][n] = [0,255,255]

    
    for i in range(0, 64):
        for j in range(0, 64):
            if prob[0,i,j] == 1:
                if link[0,i,j]==1 and i-1>=0: # i,j ---> i-1,j
                    if prob[0,i-1,j]==1:
                        cv2.line(new_img, (posi_final[1,i,j],posi_final[0,i,j]), (posi_final[1,i-1,j],posi_final[0,i-1,j]), (0,255,0), 1)
                if link[1,i,j]==1 and i-1>=0 and j+1<64: # i,j ---> i-1,j+1
                    if prob[0,i-1,j+1]==1:
                        cv2.line(new_img, (posi_final[1,i,j],posi_final[0,i,j]), (posi_final[1,i-1,j+1],posi_final[0,i-1,j+1]), (0,255,0), 1)
                if link[2,i,j]==1 and j+1<64: # i,j ---> i,j+1
                    if prob[0,i,j+1]==1:
                        cv2.line(new_img, (posi_final[1,i,j],posi_final[0,i,j]), (posi_final[1,i,j+1],posi_final[0,i,j+1]), (0,255,0), 1)
                if link[3,i,j]==1 and i+1<64 and j+1<64: # i,j ---> i+1,j+1
                    if prob[0,i+1,j+1]==1:
                        cv2.line(new_img, (posi_final[1,i,j],posi_final[0,i,j]), (posi_final[1,i+1,j+1],posi_final[0,i+1,j+1]), (0,255,0), 1)
                if link[4,i,j]==1 and i+1<64: # i,j ---> i+1,j
                    if prob[0,i+1,j]==1:
                        cv2.line(new_img, (posi_final[1,i,j],posi_final[0,i,j]), (posi_final[1,i+1,j],posi_final[0,i+1,j]), (0,255,0), 1)
                if link[5,i,j]==1 and i+1<64 and j-1>=0: # i,j ---> i+1,j-1
                    if prob[0,i+1,j-1]==1:
                        cv2.line(new_img, (posi_final[1,i,j],posi_final[0,i,j]), (posi_final[1,i+1,j-1],posi_final[0,i+1,j-1]), (0,255,0), 1)
                if link[6,i,j]==1 and j-1>=0: # i,j ---> i,j-1
                    if prob[0,i,j-1]==1:
                        cv2.line(new_img, (posi_final[1,i,j],posi_final[0,i,j]), (posi_final[1,i,j-1],posi_final[0,i,j-1]), (0,255,0), 1)
                if link[7,i,j]==1 and i-1>=0 and j-1>=0: # i,j ---> i-1,j-1
                    if prob[0,i-1,j-1]==1:
                        cv2.line(new_img, (posi_final[1,i,j],posi_final[0,i,j]), (posi_final[1,i-1,j-1],posi_final[0,i-1,j-1]), (0,255,0), 1)
    for i in range(0, 64):
        for j in range(0, 64):
            for m in range(16*i,16*i+16):
                for n in range(16*j,16*j+16):
                    if (prob[0,i,j] == 1) and (posi_final[0,i,j]==m) and (posi_final[1,i,j]==n):
                        new_img[m][n] = [0,0,255]
    
    cv2.imwrite(target+'prob_posi_link/'+name[:-7]+'prob_posi_link.png', new_img)

    sat = cv2.imread(source + name)

    truth_folder = 'dataset/test/truth/'
    #truth_folder = 'dataset/train/mask/'
    # Load the ground truth mask
    truth_mask_path = os.path.join(truth_folder, name[:-7] + 'mask.png')  
    truth_mask = cv2.imread(truth_mask_path, cv2.IMREAD_GRAYSCALE) 
    
    # Create a new image to overlay the ground truth mask
    sat = cv2.imread(source + name)
    sat[truth_mask == 255] = [0, 0, 255]  # Set ground truth mask pixels to red
    sat_merge = cv2.addWeighted(sat, 0.6, mask, 0.4, 0)  # Original merge with predicted mask
    
    cv2.imwrite(target + 'merge/' + name[:-7] + 'merge_with_truth.png', sat_merge)

    # compute accuracy for a single prediction
    predicted_flat = mask_gray.flatten()
    ground_truth_flat = truth_mask.flatten()
    correct_pixels = np.sum(predicted_flat == ground_truth_flat)
    total_pixels = predicted_flat.size
    accuracy = correct_pixels / total_pixels
    total_accuracy += accuracy
    num_samples += 1

# compute average accuracy
average_accuracy = total_accuracy / num_samples
print(f"Average Accuracy: {average_accuracy * 100:.2f}%")