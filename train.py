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
from torch.optim import lr_scheduler
from tqdm import tqdm
import pdb
import torch.nn.functional as F
from triplet_loss import *
import pickle
from build_vocab import Vocabulary
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.utils as vutils

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
device = [0]
with open(opts.vocab_path, 'rb') as f:
    vocab = pickle.load(f)
# =============================================================================

##load models
image_model = torch.nn.DataParallel(ImageEmbedding().cuda(), device_ids=device)
recipe_model = torch.nn.DataParallel(TextEmbedding().cuda(), device_ids=device)
netG = torch.nn.DataParallel(G_NET().cuda(), device_ids=device)
multi_label_net = torch.nn.DataParallel(MultiLabelNet().cuda(), device_ids=device)
cm_discriminator = torch.nn.DataParallel(cross_modal_discriminator().cuda(), device_ids=device)
text_discriminator = torch.nn.DataParallel(text_emb_discriminator().cuda(), device_ids=device)
netsD = torch.nn.DataParallel(D_NET128().cuda(), device_ids=device)

## load loss functions
triplet_loss = TripletLoss(device, margin=0.3)
img2text_criterion = nn.MultiLabelMarginLoss().cuda()

weights_class = torch.Tensor(opts.numClasses).fill_(1)
weights_class[0] = 0
class_criterion = nn.CrossEntropyLoss(weight=weights_class).cuda()

GAN_criterion = nn.BCELoss().cuda()

nz = opts.Z_DIM
noise = Variable(torch.FloatTensor(opts.batch_size, nz)).cuda()
fixed_noise = Variable(torch.FloatTensor(opts.batch_size, nz).normal_(0, 1)).cuda()
real_labels = Variable(torch.FloatTensor(opts.batch_size).fill_(1)).cuda()
fake_labels = Variable(torch.FloatTensor(opts.batch_size).fill_(0)).cuda()

fc_sia = nn.Sequential(
            nn.Linear(opts.embDim, opts.embDim),
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh(),
        ).cuda()
    
model_list = [image_model, recipe_model, netG, multi_label_net, cm_discriminator, text_discriminator, netsD, fc_sia]

optimizer = torch.optim.Adam([
                {'params': image_model.parameters()},
                {'params': recipe_model.parameters()},
                {'params': netG.parameters()},
                {'params': multi_label_net.parameters()}
            ], lr=opts.lr, betas=(0.5, 0.999))

optimizers_imgD = torch.optim.Adam(netsD.parameters(), lr=opts.lr, betas=(0.5, 0.999))
optimizer_cmD = torch.optim.Adam(cm_discriminator.parameters(), lr=opts.lr, betas=(0.5, 0.999))

label = list(range(0, opts.batch_size))
label.extend(label)
label = np.array(label)
label = torch.tensor(label).cuda().long()

method = 'acme'
save_folder = method
os.makedirs(save_folder, exist_ok=True) 
epoch_trace_f_dir = os.path.join(save_folder, "trace_" + method + ".csv")
with open(epoch_trace_f_dir, "w") as f:
    f.write("epoch,lr,I2R,R@1,R@5,R@10,R2I,R@1,R@5,R@10\n")

def main():

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([ 
        transforms.Resize(256),
        transforms.RandomCrop(224),   
        transforms.RandomHorizontalFlip()])
    val_transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224)])
    
    cudnn.benchmark = True

    # preparing the training laoder
    train_loader = get_loader(opts.img_path, train_transform, vocab, opts.data_path, partition='train',
                            batch_size=opts.batch_size, shuffle=True,
                            num_workers=opts.workers, pin_memory=True)
    print('Training loader prepared.')

    # preparing validation loader 
    val_loader = get_loader(opts.img_path, val_transform, vocab, opts.data_path, partition='test',
                            batch_size=opts.batch_size, shuffle=False,
                            num_workers=opts.workers, pin_memory=True)
    print('Validation loader prepared.')

    best_val_i2t = {1:0.0,5:0.0,10:0.0}
    best_val_t2i = {1:0.0,5:0.0,10:0.0}
    best_epoch_i2t = 0
    best_epoch_t2i = 0

    for epoch in range(0, opts.epochs):

        train(train_loader, epoch, val_loader)

        recall_i2t, recall_t2i, medR_i2t, medR_t2i = validate(val_loader)
        with open(epoch_trace_f_dir, "a") as f:
            lr = optimizer.param_groups[1]['lr']
            f.write("{},{},{},{},{},{},{},{},{},{}\n".format\
                (epoch,lr,medR_i2t,recall_i2t[1],recall_i2t[5],recall_i2t[10],\
                    medR_t2i,recall_t2i[1],recall_t2i[5],recall_t2i[10]))

        for keys in best_val_i2t:
            if recall_i2t[keys] > best_val_i2t[keys]:
                best_val_i2t = recall_i2t
                best_epoch = epoch+1
                model_num = 1
                for model_n in model_list:
                    filename = save_folder + '/model_e%03d_v%d.pkl' % (epoch+1, model_num)
                    torch.save(model_n.state_dict(), filename)
                    model_num += 1
                break  
        print("best: ", best_epoch, best_val_i2t)
        print('params lr: %f' % optimizer.param_groups[1]['lr'])
        
        if epoch == 30:
            optimizer.param_groups[0]['lr'] = 0.00001
            optimizer.param_groups[1]['lr'] = 0.00001
            optimizer.param_groups[2]['lr'] = 0.00001
            optimizer.param_groups[3]['lr'] = 0.00001
            optimizers_imgD.param_groups[0]['lr'] = 0.00001
            optimizer_cmD.param_groups[0]['lr'] = 0.00001

def train_Dnet(idx, real_imgs, fake_imgs, mu, label_class):
    netD = netsD
    real_imgs = real_imgs[idx]
    fake_imgs = fake_imgs[idx]

    real_logits = netD(real_imgs, mu.detach())
    fake_logits = netD(fake_imgs.detach(), mu.detach())

    lossD_real = GAN_criterion(real_logits[0], real_labels)
    lossD_fake = GAN_criterion(fake_logits[0], fake_labels)

    lossD = lossD_real + lossD_fake
    return lossD

def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD

def train_Gnet(idx, real_imgs, fake_imgs, mu, logvar, label_class):
    netD = netsD
    real_imgs = real_imgs[idx]
    fake_imgs = fake_imgs[idx]

    real_logits = netD(real_imgs, mu)
    fake_logits = netD(fake_imgs, mu)

    lossG_fake = GAN_criterion(fake_logits[0], real_labels)

    lossG_real_cond = class_criterion(real_logits[1], label_class)
    lossG_fake_cond = class_criterion(fake_logits[1], label_class)
    lossG_cond  = lossG_real_cond + lossG_fake_cond

    lossG = lossG_fake + lossG_cond

    kl_loss = KL_loss(mu, logvar) * 2
    lossG = kl_loss + lossG

    return lossG

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.autograd.Variable(torch.cuda.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,  # fack samples
        inputs=interpolates,   # real samples
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train(train_loader, epoch, val_loader):
    tri_losses = AverageMeter()

    img_losses = AverageMeter()
    text_losses = AverageMeter()
    cmG_losses = AverageMeter()

    image_model.train()
    recipe_model.train()

    for i, data in enumerate(tqdm(train_loader)):

        img_emd_modal = image_model(data[0][0].cuda())
        recipe_emb_modal = recipe_model(data[0][1].cuda(), data[0][2].cuda(), data[0][3].cuda(), data[0][4].cuda())

        ################################################################
        # modal-level fusion
        ################################################################
        real_validity = cm_discriminator(img_emd_modal.detach())
        fake_validity = cm_discriminator(recipe_emb_modal.detach())
        gradient_penalty = compute_gradient_penalty(cm_discriminator, img_emd_modal.detach(), recipe_emb_modal.detach())
        loss_cmD = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
        optimizer_cmD.zero_grad()
        loss_cmD.backward()
        optimizer_cmD.step()

        g_fake_validity = cm_discriminator(recipe_emb_modal)
        loss_cmG = -torch.mean(g_fake_validity)

        ################################################################
        # cross-modal retrieval
        ################################################################
        img_id_fea = norm(fc_sia(img_emd_modal))
        rec_id_fea = norm(fc_sia(recipe_emb_modal))
        tri_loss = global_loss(triplet_loss, torch.cat((img_id_fea, rec_id_fea)), label)[0]
        
        ################################################################
        # translation consistency
        label_class = data[1][7].cuda()
        real_imgs = []
        real_imgs.append(data[1][8].cuda())
        ingr_cap = data[1][5].cuda()
        lengths = torch.tensor(data[1][6]).cuda()
        targets = pack_padded_sequence(ingr_cap, lengths, batch_first=True)[0]
        one_hot_cap = data[1][9].cuda().long()
        ################################################################
        # img2text
        ################################################################
        recipe_out = multi_label_net(img_id_fea)
        loss_i2t = img2text_criterion(recipe_out[0], one_hot_cap)
        loss_t_class = class_criterion(recipe_out[1], label_class)
        loss_text = loss_i2t + loss_t_class

        ###############################################################
        # text2img
        ###############################################################
        noise.data.normal_(0, 1)
        fake_imgs, mu, logvar = netG(noise, rec_id_fea)

        lossD_total = 0
        lossD = train_Dnet(0, real_imgs, fake_imgs, mu, label_class)
        optimizers_imgD.zero_grad()
        lossD.backward()
        optimizers_imgD.step()

        lossG = train_Gnet(0, real_imgs, fake_imgs, mu, logvar, label_class)
        loss_img = lossG

        if loss_text.item() < loss_img.item():
            loss_img = (loss_text.item()/loss_img.item()) * loss_img
        else:
            loss_text = (loss_img.item()/loss_text.item()) * loss_text
        loss_g = loss_img + loss_text

        ###############################################################
        # back-propogate
        ###############################################################
        loss = tri_loss + 0.005 * loss_cmG + 0.002 * loss_g  

        tri_losses.update(tri_loss.item(), data[0][0].size(0))
        img_losses.update(loss_img.item(), data[0][0].size(0))
        text_losses.update(loss_text.item(), data[0][0].size(0))
        cmG_losses.update(loss_cmG.item(), data[0][0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(epoch)
    print('Epoch: {0}  '
              'tri loss {tri_loss.val:.4f} ({tri_loss.avg:.4f}),  '
              'cm loss {loss_cmG.val:.4f} ({loss_cmG.avg:.4f}),  '
              'img loss {img_losses.val:.4f} ({img_losses.avg:.4f}),  '
              'text loss {loss_text.val:.4f} ({loss_text.avg:.4f})'
              .format(
               epoch, tri_loss=tri_losses, loss_cmG=cmG_losses,
               img_losses=img_losses, loss_text=text_losses))
             

def validate(val_loader):

    # switch to evaluate mode
    image_model.eval()
    recipe_model.eval()

    end = time.time()
    for i, data in enumerate(tqdm(val_loader)):

        with torch.no_grad():

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
    N = 1000
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
    N = 1000
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
