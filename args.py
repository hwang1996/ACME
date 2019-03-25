import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='tri-joint parameters')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--device', default=[0], type=list)

    # data
    parser.add_argument('--img_path', default='../im2recipe-Pytorch/data')
    parser.add_argument('--data_path', default='data/food_data/')
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--vocab_path', type=str, default='data/new_word_dict.pkl', help='path for vocabulary wrapper')

    # model
    parser.add_argument('--batch_size', default=64, type=int)

    # im2recipe model
    parser.add_argument('--embDim', default=1024, type=int)
    parser.add_argument('--nRNNs', default=1, type=int)
    parser.add_argument('--srnnDim', default=1024, type=int)
    parser.add_argument('--irnnDim', default=300, type=int)
    parser.add_argument('--imfeatDim', default=2048, type=int)
    parser.add_argument('--stDim', default=1024, type=int)
    parser.add_argument('--ingrW2VDim', default=300, type=int)
    parser.add_argument('--maxSeqlen', default=20, type=int)
    parser.add_argument('--maxIngrs', default=20, type=int)
    parser.add_argument('--maxImgs', default=5, type=int)
    parser.add_argument('--numClasses', default=1048, type=int)

    #img-text
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    #text-img
    parser.add_argument('--Z_DIM', type=int , default=100, help='noise dimension for image generation')
    parser.add_argument('--DF_DIM', type=int , default=64, help='D dimension')
    parser.add_argument('--GF_DIM', type=int , default=64, help='G dimension')
    parser.add_argument('--EMBEDDING_DIM', type=int , default=128, help='embedding dimension')
    parser.add_argument('--R_NUM', type=int , default=2, help='resudial unit number')
    parser.add_argument('--BRANCH_NUM', type=int , default=3, help='the number of the stages')
    parser.add_argument('--B_CONDITION', type=bool , default=True, help='if use condition loss')

    # training 
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--ingrW2V', default='data/vocab.bin',type=str)  

    # dataset
    parser.add_argument('--maxlen', default=20, type=int)
    parser.add_argument('--vocab', default = 'vocab.txt', type=str)
    parser.add_argument('--dataset', default = '../data/recipe1M/', type=str)
    parser.add_argument('--sthdir', default = '../data/', type=str)

    return parser
