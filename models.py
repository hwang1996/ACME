import torch
import torch.nn as nn
import torch.nn.parallel
import torch.legacy as legacy
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchwordemb
from args import get_parser
import pdb
import torch.nn.functional as F
from torch.autograd import Variable

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# # =============================================================================

##############################################################
## some codes modified from Han Zhang's Stack-GAN (https://github.com/hanzhanggit/StackGAN)
## and Salvador's im2recipe (https://github.com/torralba-lab/im2recipe-Pytorch)
##############################################################
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule, self).__init__()
        
    def forward(self, x, dim):
        y = torch.cat(x, dim)
        return y

def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

# Skip-thoughts LSTM
class stRNN(nn.Module):
    def __init__(self):
        super(stRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=opts.stDim, hidden_size=opts.srnnDim, bidirectional=False, batch_first=True)
                
    def forward(self, x, sq_lengths):
        # here we use a previous LSTM to get the representation of each instruction 
        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx\
                .view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the lstm
        out, hidden = self.lstm(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        unsorted_idx = original_idx.view(-1,1,1).expand_as(unpacked)
        # we get the last index of each sequence in the batch
        idx = (sq_lengths-1).view(-1,1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        # we sort and get the last element of each sequence
        output = unpacked.gather(0, unsorted_idx.long()).gather(1,idx.long())
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output 

class ingRNN(nn.Module):
    def __init__(self):
        super(ingRNN, self).__init__()

        self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)
        _, vec = torchwordemb.load_word2vec_bin(opts.ingrW2V)
        self.embs = nn.Embedding(vec.size(0), opts.ingrW2VDim, padding_idx=0) # not sure about the padding idx 
        self.embs.weight.data.copy_(vec)

        # self.embs = nn.Embedding(3122, opts.ingrW2VDim)
        # self.irnn = nn.LSTM(input_size=opts.ingrW2VDim, hidden_size=opts.irnnDim, bidirectional=True, batch_first=True)

    def forward(self, x, sq_lengths):

        # we get the w2v for each element of the ingredient sequence
        x = self.embs(x)  # torch.Size([64, 20, 300])

        # sort sequence according to the length
        sorted_len, sorted_idx = sq_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx\
                .view(-1,1,1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long())
        # pack sequence
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs, sorted_len.cpu().data.numpy(), batch_first=True)
        # pass it to the rnn
        out, hidden = self.irnn(packed_seq)

        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)

        # LSTM
        # bi-directional
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        # 2 directions x batch_size x num features, we transpose 1st and 2nd dimension
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous() # torch.Size([64, 2, 300])
        
        output = output.view(output.size(0),output.size(1)*output.size(2))

        return output


class TextEmbedding(nn.Module):
    def __init__(self):
        super(TextEmbedding, self).__init__()

        self.recipe_embedding = nn.Sequential(
            nn.Linear(opts.irnnDim*2 + opts.srnnDim, opts.embDim),
            nn.Tanh(),
        )

        self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)
        
        self.fc_recipe = nn.Sequential(
            nn.Linear(opts.embDim, opts.embDim),
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh(),
        )

        self.stRNN_     = stRNN()
        self.ingRNN_    = ingRNN()
        self.table      = TableModule()
        
    def forward(self, y1, y2, z1, z2, ingrs_emb=None):
        if torch.is_tensor(ingrs_emb):
            recipe_emb = self.table([self.stRNN_(y1,y2), ingrs_emb],1) # joining on the last dim 
        else:
            ingrs_emb = self.ingRNN_(z1,z2)
            recipe_emb = self.table([self.stRNN_(y1,y2), ingrs_emb],1) # joining on the last dim 
        recipe_emb_domain = self.recipe_embedding(recipe_emb)
        
        output = recipe_emb_domain
        return output

class ImageEmbedding(nn.Module):
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # we do not use the last fc layer.
        self.visionMLP = nn.Sequential(*modules)

        self.visual_embedding = nn.Sequential(
            nn.Linear(opts.imfeatDim, opts.embDim),
            nn.Tanh(),
        )

        self.semantic_branch = nn.Linear(opts.embDim, opts.numClasses)

        self.fc_visual = nn.Sequential(
            nn.Linear(opts.embDim, opts.embDim),
            nn.BatchNorm1d(opts.embDim),
            nn.Tanh(),
        )
 
    def forward(self, x):

        visual_emb = self.visionMLP(x)
        visual_emb = visual_emb.view(visual_emb.size(0), -1) # batch_size * 2048
        visual_emb_domain = self.visual_embedding(visual_emb)
        
        output = visual_emb_domain
        return output


class MultiLabelNet(nn.Module):
    def __init__(self):
        super(MultiLabelNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opts.embDim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 4102)
        )

        self.ingrs_linear = nn.Sequential(
            nn.Linear(opts.embDim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, opts.numClasses)
        )

    def forward(self, features):
        output = self.model(features)

        return [nn.Sigmoid()(output), self.ingrs_linear(features)]


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=100):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=opts.irnnDim, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(opts.irnnDim, vocab_size)
        self.ingrs_linear = nn.Linear(opts.irnnDim, opts.irnnDim*2)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True) 
        outs, hidden = self.lstm(packed)

        _, original_idx = lengths.sort(0, descending=False)
        unsorted_idx = original_idx.view(1,-1,1).expand_as(hidden[0])
        output = hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        # output = output.view(output.size(0),output.size(1)*output.size(2))
        output_fea = self.ingrs_linear(output.view(output.size(0),output.size(1)*output.size(2)))

        outputs = self.linear(outs[0])
        return [outputs, output_fea]
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            outs, states = self.lstm(inputs, states)             # outs: (batch_size, 1, hidden_size)
            outputs = self.linear(outs.squeeze(1))               # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


class text_emb_discriminator(nn.Module):
    def __init__(self):
        super(text_emb_discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opts.irnnDim * 2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, opts.numClasses * 2)
        )

    def forward(self, fea):
        output = self.model(fea)

        return output

class cross_modal_discriminator(nn.Module):
    def __init__(self):
        super(cross_modal_discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opts.embDim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, fea):
        output = self.model(fea)

        return output
        

class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = opts.embDim
        self.ef_dim = opts.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        
        with torch.cuda.device(logvar.get_device()):
            std = logvar.mul(0.5).exp_()
            eps = torch.FloatTensor(std.size()).normal_().cuda()
            eps = Variable(eps)
        
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if opts.B_CONDITION:
            self.in_dim = opts.Z_DIM + opts.EMBEDDING_DIM
        else:
            self.in_dim = opts.Z_DIM
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code=None):
        if opts.B_CONDITION and c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)
        else:
            in_code = z_code
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=opts.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if opts.B_CONDITION:
            self.ef_dim = opts.EMBEDDING_DIM
        else:
            self.ef_dim = opts.Z_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code

class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = opts.GF_DIM
        self.define_module()

    def define_module(self):
        if opts.B_CONDITION:
            self.ca_net = CA_NET()

        if opts.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if opts.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if opts.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)
        if opts.BRANCH_NUM > 3: # Recommended structure (mainly limited by GPU memory), and not test yet
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 4, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 8)
        if opts.BRANCH_NUM > 4:
            self.h_net4 = NEXT_STAGE_G(self.gf_dim // 8, num_residual=1)
            self.img_net4 = GET_IMAGE_G(self.gf_dim // 16)

    def forward(self, z_code, text_embedding=None):
        if opts.B_CONDITION and text_embedding is not None:
            c_code, mu, logvar = self.ca_net(text_embedding)
        else:
            c_code, mu, logvar = z_code, None, None
        fake_imgs = []
        h_code1 = self.h_net1(z_code, c_code)
        h_code2 = self.h_net2(h_code1, c_code)
        fake_img2 = self.img_net2(h_code2)

        fake_imgs.append(fake_img2)

        return fake_imgs, mu, logvar

# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = opts.DF_DIM
        self.ef_dim = opts.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())
        self.logits_class = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if opts.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_branch = nn.Sequential(
                nn.Conv2d(ndf * 8, ndf * 4, kernel_size=4, stride=4),
                # nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True))
            self.semantic_branch = nn.Linear(ndf * 4, opts.numClasses)

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        if opts.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if opts.B_CONDITION:
            out_uncond = self.uncond_branch(x_code)
            out_uncond = out_uncond.view(out_uncond.size(0), -1)
            out_uncond = self.semantic_branch(out_uncond)
            return [output.view(-1), out_uncond]
        else:
            return [output.view(-1)]
