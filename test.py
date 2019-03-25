import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import get_loader 
from args import get_parser
from models import *
from tqdm import tqdm
import pdb
import torch.nn.functional as F
from triplet_loss import *
import pickle
from build_vocab import Vocabulary
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.utils as vutils

device = [0]
with open(opts.vocab_path, 'rb') as f:
    vocab = pickle.load(f)

image_model = ImageEmbedding()
image_model = torch.nn.DataParallel(image_model, device_ids=device).cuda()
image_model_pre = torch.load('acme/model_e045_v1.pkl')
image_model.load_state_dict(image_model_pre)

recipe_model = TextEmbedding()
recipe_model = torch.nn.DataParallel(recipe_model, device_ids=device).cuda()
recipe_model_pre = torch.load('acme/model_e045_v2.pkl')
recipe_model.load_state_dict(recipe_model_pre)

fc_sia = nn.Sequential(
            nn.Linear(opts.embDim, opts.embDim),
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh(),
        ).cuda()
fc_sia.load_state_dict(torch.load('acme/model_e045_v8.pkl'))

np.random.seed(opts.seed)

def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224)])

    val_loader = get_loader(opts.img_path, val_transform, vocab, opts.data_path, partition='test',
                            batch_size=opts.batch_size, shuffle=False,
                            num_workers=opts.workers, pin_memory=True)
    print('Validation loader prepared.')

    test(val_loader)

def test(test_loader):
    image_model.eval()
    recipe_model.eval()

    for i, data in enumerate(tqdm(test_loader)):

        img_emd_modal = image_model(data[0][0].cuda())
        recipe_emb_modal = recipe_model(data[0][1].cuda(), data[0][2].cuda(), data[0][3].cuda(), data[0][4].cuda())

        img_emd_modal = norm(fc_sia(img_emd_modal))
        recipe_emb_modal = norm(fc_sia(recipe_emb_modal))  
     
        if i==0:
            data0 = img_emd_modal.data.cpu().numpy()
            data1 = recipe_emb_modal.data.cpu().numpy()
        else:
            data0 = np.concatenate((data0,img_emd_modal.data.cpu().numpy()),axis=0)
            data1 = np.concatenate((data1,recipe_emb_modal.data.cpu().numpy()),axis=0)

    medR_i2t, recall_i2t = rank_i2t(opts, data0, data1)
    print('I2T Val medR {medR:.4f}\t'
          'Recall {recall}'.format(medR=medR_i2t, recall=recall_i2t))

    medR_t2i, recall_t2i = rank_t2i(opts, data0, data1)
    print('T2I Val medR {medR:.4f}\t'
          'Recall {recall}'.format(medR=medR_t2i, recall=recall_t2i))

    return recall_i2t, recall_t2i, medR_i2t, medR_t2i

def rank_i2t(opts, img_embeds, rec_embeds):
    random.seed(opts.seed)
    im_vecs = img_embeds 
    instr_vecs = rec_embeds 

    # Ranker
    N = 10000
    idxs = range(N)

    glob_rank = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(10):

        ids = random.sample(range(0,len(img_embeds)), N)
        im_sub = im_vecs[ids,:]
        instr_sub = instr_vecs[ids,:]

        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}

        for ii in idxs:
            distance = {}
            for j in range(N):
                distance[j] = np.linalg.norm(im_sub[ii] - instr_sub[j])
            distance_sorted = sorted(distance.items(), key=lambda x:x[1])
            pos = np.where(np.array(distance_sorted) == distance[ii])[0][0]

            if (pos+1) == 1:
                recall[1]+=1
            if (pos+1) <=5:
                recall[5]+=1
            if (pos+1)<=10:
                recall[10]+=1

            # store the position
            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i]=recall[i]/N

        med = np.median(med_rank)

        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10

    return np.average(glob_rank), glob_recall

def rank_t2i(opts, img_embeds, rec_embeds):
    random.seed(opts.seed)
    im_vecs = img_embeds 
    instr_vecs = rec_embeds 

    # Ranker
    N = 10000
    idxs = range(N)

    glob_rank = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(10):

        ids = random.sample(range(0,len(img_embeds)), N)
        im_sub = im_vecs[ids,:]
        instr_sub = instr_vecs[ids,:]

        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}

        for ii in idxs:
            distance = {}
            for j in range(N):
                distance[j] = np.linalg.norm(instr_sub[ii] - im_sub[j])
            distance_sorted = sorted(distance.items(), key=lambda x:x[1])
            pos = np.where(np.array(distance_sorted) == distance[ii])[0][0]

            if (pos+1) == 1:
                recall[1]+=1
            if (pos+1) <=5:
                recall[5]+=1
            if (pos+1)<=10:
                recall[10]+=1

            # store the position
            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i]=recall[i]/N

        med = np.median(med_rank)

        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10

    return np.average(glob_rank), glob_recall

if __name__ == '__main__':
    main()