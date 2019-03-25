from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import sys
import pickle
import numpy as np
import lmdb
import torch
import pdb
import torchvision.transforms as transforms
import nltk
from build_vocab import Vocabulary
from args import get_parser


parser = get_parser()
opts = parser.parse_args()


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except:
        return Image.new('RGB', (224, 224), 'white')


class ImagerLoader(data.Dataset):
    def __init__(self, img_path, transform=None, vocab=None,
                 loader=default_loader, data_path=None, partition=None):

        if data_path == None:
            raise Exception('No data path specified.')

        if partition is None:
            raise Exception('Unknown partition type %s.' % partition)
        else:
            self.partition = partition

        self.env = lmdb.open(os.path.join(img_path, partition + '_lmdb'), max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        
        with open(os.path.join('data/food_data/' + partition + '_ids.pkl'), 'rb') as f:
            self.ids = pickle.load(f, encoding='latin1')

        with open(os.path.join('data/food_data/' + partition + '_split.pkl'), 'rb') as f:
            self.split = pickle.load(f, encoding='latin1') 

        self.imgPath = img_path
        self.maxInst = 20

        self.transform = transform
        self.loader = loader

        self.vocab = vocab

    def __getitem__(self, index):

        # for background
        with self.env.begin(write=False) as txn:
            serialized_sample = txn.get(self.ids[index].encode())
        sample = pickle.loads(serialized_sample, encoding='latin1')
        imgs = sample['imgs']
        food_id = self.ids[index]
        
        if self.partition != 'train':
            imgIdx = 0
        else:
            imgIdx = np.random.choice(range(min(5, len(imgs))))
        loader_path = [imgs[imgIdx]['id'][i] for i in range(4)]
        loader_path = os.path.join(*loader_path)
        path = os.path.join(self.imgPath, self.partition, loader_path, imgs[imgIdx]['id'])

        # instructions
        instrs = sample['intrs']
        itr_ln = len(instrs)
        t_inst = np.zeros((self.maxInst, np.shape(instrs)[1]), dtype=np.float32)
        t_inst[:itr_ln][:] = instrs
        instrs = torch.FloatTensor(t_inst)

        # ingredients
        ingrs = sample['ingrs'].astype(int)
        ingrs = torch.LongTensor(ingrs)
        igr_ln = max(np.nonzero(sample['ingrs'])[0]) + 1

        # image
        img = self.loader(path)
        img = self.transform(img)
        normalize = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        re_img = transforms.Resize(128)(img)
        img = normalize(img)
        ret = normalize(re_img)
        
        class_label = sample['classes'] - 1

        vocab = self.vocab
        selec_ingrs = set(self.split[food_id]['ingredients'])
        ingr_cap = []
        ingr_cap.append(vocab('<start>'))

        one_hot_vec = torch.zeros(4102)
        for i in list(selec_ingrs):
            one_hot_vec[vocab(str(i).lower())] = 1

        ingr_cap = torch.Tensor(ingr_cap)

        # output
        ## also output the length of captions, which could be used in LSTM prediction
        return img, instrs, itr_ln, ingrs, igr_ln, \
            ingr_cap, class_label, ret, one_hot_vec, food_id

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).

    data.sort(key=lambda x: len(x[5]), reverse=True)
    img, instrs, itr_ln, ingrs, igr_ln,\
    ingr_cap, class_label, ret, one_hot_vec, food_id = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(img, 0)
    instrs = torch.stack(instrs, 0)
    itr_ln = torch.LongTensor(list(itr_ln))
    ingrs = torch.stack(ingrs, 0)
    igr_ln = torch.LongTensor(list(igr_ln))
    class_label = torch.LongTensor(list(class_label))
    ret = torch.stack(ret, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in ingr_cap]
    targets = torch.zeros(len(ingr_cap), max(lengths)).long()
    for i, cap in enumerate(ingr_cap):
        end = lengths[i]
        targets[i, :end] = cap[:end] 

    one_hot_vec = torch.stack(one_hot_vec, 0)

    return [images, instrs, itr_ln, ingrs, igr_ln, list(food_id)], \
    [images, instrs, itr_ln, ingrs, igr_ln, targets, lengths, class_label, ret, one_hot_vec]

def get_loader(img_path, transform, vocab, data_path, partition, batch_size, shuffle, num_workers, pin_memory):
    
    data_loader = torch.utils.data.DataLoader(ImagerLoader(img_path, transform, vocab,
                                                data_path=data_path, partition=partition), 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory,
                                              drop_last=True,
                                              collate_fn=collate_fn)
    return data_loader