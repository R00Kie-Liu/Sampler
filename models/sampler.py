import torch
import torch.nn as nn
import math
import mobula
import torch.nn.functional as F
import perturbations
import shufflenetv2 as shufflenetv2
from torchvision import transforms
from attention_sampler.attsampler_th import AttSampler
mobula.op.load('models/attention_sampler')


def order_topk(x, k=8):
    if len(x.shape) > 2:
        x = x[:,:,0]
    return F.one_hot(torch.sort(torch.topk(x, k=args.k, dim=-1)[-1])[0], list(x.shape)[-1]).transpose(-1,-2).float()


def amplify(maps, data, size=224):
    '''
    Input:
    maps: B, w_m, h_m
    data: B, 3, w, h
    return: B, 3, w, h
    '''
    maps = F.interpolate(maps.unsqueeze(1), (size, size), mode='bilinear', align_corners=True).squeeze(1)
    assert maps.size(-1) == data.size(-1)
    map_sx = torch.unsqueeze(torch.max(maps, 2)[0], dim=2)
    map_sy = torch.unsqueeze(torch.max(maps, 1)[0], dim=2)
    sum_sx = torch.sum(map_sx, dim=(1, 2), keepdim=True)
    sum_sy = torch.sum(map_sy, dim=(1, 2), keepdim=True)
    map_sx /= sum_sx
    map_sy /= sum_sy

    data_pred = AttSampler(scale=1, dense=4)(data, map_sx, map_sy)

    return data_pred

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.dim = d_model
        self.seq_len = max_len
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.view(-1, self.seq_len, self.dim)
        x = x + self.pe[:,:]
        return x.view(-1, self.dim)

class Generator(nn.Module):
    def __init__(self, insize=1024, outsize=512, z_dim=64, bias=False):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.z_dim = z_dim
        self.bias = bias
        self.encoder = nn.Linear(insize, z_dim * 2)
        if bias:
            self.gen = nn.Linear(z_dim, outsize + 1)
        else:
            self.gen = nn.Linear(z_dim, outsize)

    def forward(self, task_context):
        # task_context: , insize
        distribution = self.encoder(task_context) # , mu_size * 2
        mu = distribution[:self.z_dim]
        log_var = distribution[self.z_dim:]

        z_signal = torch.randn(1, self.z_dim).cuda()
        z_mu = mu.unsqueeze(0) # 1, z_dim
        z_log_var = log_var.unsqueeze(0) # 1, z_dim
        z = z_mu + torch.exp(z_log_var/2) * z_signal # 1, z_dim

        weight_bias = self.gen(z) # 1, out_size * 2
        weight = weight_bias[:, :self.outsize] # 1, out_size
        weight = weight / torch.norm(weight, 2) # normalize 
        if self.bias:
            bias = weight_bias[0, self.outsize:] # ,1
            return weight, bias
        else:
            return weight
  
class Attention(nn.Module):
    def __init__(self, args, out_size=48):
        super().__init__()
        self.att_size = args.img_size
        self.out_size = out_size
        #self.conv_q = nn.Conv2d(192, out_size, 1, 1)
        #self.conv_v = nn.Conv2d(192, out_size, 1, 1)
        self.softmax  = nn.Softmax(dim=-1)

        #self.init_conv(self.conv_q)
        #self.init_conv(self.conv_v)
        
    def init_conv(self, conv, glu=True):
        nn.init.xavier_uniform_(conv.weight)
        if conv.bias is not None:
            conv.bias.data.zero_()

    def upsample(self, att, size):
        return F.interpolate(att, (size, size), mode='bilinear', align_corners=True).squeeze(1)

    def forward(self, features):
        '''
        f1: B, 24, size/4, size/4
        f2: B, 48, size/8, size/8
        f3: B, 96, size/16, size/16
        f4: B, 192, size/32, size/32
        '''
        f1, f2, f3, f4 = features
        batch = f2.size(0)
        max_scale = f2.size(-1)
        f4_up = self.upsample(f4, max_scale).mean(dim=1) # B, 192, max_scale=w, max_scale=h
        f3_up = self.upsample(f3, max_scale).mean(dim=1)
        att = (f2.mean(dim=1) + f3_up + f4_up) / 3

        att = self.upsample(att.unsqueeze(1), self.att_size).squeeze(1)

        return att
        
class Selector(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.resize = transforms.Resize(args.mini_size)
        if self.args.lightCNN == 'shufflev2_0_5':
            lightCNN = shufflenetv2.shufflenet_v2_x0_5(pretrained=True)
        elif self.args.lightCNN == 'shufflev2_1_0':
            lightCNN = shufflenetv2.shufflenet_v2_x1_0(pretrained=True)
        lightCNN.fc = nn.Identity()
        self.backbone = lightCNN
        self.generator = Generator()
        self.attention = Attention(self.args)
        if self.args.ada:
            self.evaluator = nn.Sequential(PositionalEmbedding(d_model=2048, max_len=self.args.seq_len),
                                            nn.Linear(2048, 512),
                                            nn.ReLU(),
                                            )
        else:
            self.evaluator = nn.Sequential(PositionalEmbedding(d_model=2048, max_len=self.args.seq_len),
                                            nn.Linear(2048, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 1),
                                            )
        self.func = order_topk
    
    def return_topk_func(self):
        iteration = self.args.current_iter
        if iteration == 1:
            print('Initial sigma as', self.args.sigma)
            self.args.current_sigma = self.args.sigma
        sigma_min = 0.01
        step = self.args.sigma_step
        if self.args.sigma_decay and (iteration + 1) % step == 0:
            self.args.current_sigma = self.args.sigma * ((0.8) ** ((iteration + 1) // step))
            if self.args.current_sigma < sigma_min:
                self.args.current_sigma = sigma_min
            print('Decay sigma, the current sigma is {:.2}:'.format(self.args.current_sigma))
        topk_func = perturbations.perturbed(self.func, 
                                            num_samples=self.args.num_samples,
                                            sigma=self.args.current_sigma,
                                            noise=self.args.noise,
                                            batched=True)
        return topk_func


    def fast_topk(self, x):
        return F.one_hot(torch.sort(torch.topk(x, k=self.args.k, dim=-1)[-1])[0], list(x.shape)[-1]).transpose(-1,-2).float()


    def forward(self, candidates, weight=None, category='support'):
        
        x = self.resize(candidates) # way * shot * frame, 3, mini_w, mini_h
        x, f1, f2, f3, f4 = self.backbone(x) # way * shot * frame, dim
        n, dim = x.size()
        x = x.view(-1, self.args.seq_len, dim) # way * shot, frame, dim
        feature = x


        # Calulate spatial attention map
        maps = self.attention([f1,f2,f3,f4]) # way * shot * frame, 16 , 16

        # Calculate the global feature of the whole video as a kind of guidance
        if self.args.shot > 1:
            info_g = x.view(self.args.way, -1, *x.shape[1:]) # way, shot, frame, dim
            shot = info_g.size(1)
            info_g = info_g.mean(dim=-3, keepdim=True).mean(dim=-2) # way, 1, dim
            info_g = info_g.repeat(shot, 1, 1).expand(-1, self.args.seq_len, -1) # way * shot, 1, dim --> way * shot, frame, dim
        else:
            info_g = x.mean(dim=-2, keepdim=True).expand(-1, self.args.seq_len, -1) # (way * shot, 1, dim) expand to --> (way * shot, frame, dim)

        # Calculate the task feature
        task_f = info_g[:,0,:].mean(dim=0) # ,dim

        # Feed into Evaluator to get scores
        x = torch.cat((x, info_g), dim=-1) # way * shot, frame, 2*dim
        score = self.evaluator(x.view(n, dim*2)) # way * shot * frame, 1 (if task_ada, way * shot * frame, weight_dim)
        #==== Dynamic linear weight generation =====
        if self.args.ada:
            if category == 'support':
                weight = self.generator(task_f) # weight: 1, weight_dim, bias: weight_dim
            else:
                assert weight!=None # If query samples, weight should be given
            score = F.linear(score, weight)

        # Normalize score with min-max
        score = score.view(-1, self.args.seq_len) # way * shot, frame
        safe_min_value = 1e-4
        norm_score = (score - score.min(dim=1, keepdim=True)[0]) / (score.max(dim=1, keepdim=True)[0]- score.min(dim=1, keepdim=True)[0] + safe_min_value) # way * shot, frame

        if self.training:
            score = norm_score.unsqueeze(-1).expand(-1,-1,self.args.k) # way * shot, frame, k
            topk_func = self.return_topk_func()
            indices = topk_func(score) # way * shot, frame, k
        else:
            indices = self.fast_topk(norm_score)
        #seleted_score = torch.bmm(indices.transpose(-1,-2), )
        indices = indices.transpose(-1,-2) # way * shot, k, frame
        selected_feature = torch.bmm(indices, feature)

        #===== Spatial amplifier =====
        # indices: way * shot, k, frame
        # maps: way * shot * frame, 16, 16
        maps_scale = maps.size(-1)
        maps = maps.view(-1, self.args.seq_len, maps_scale*maps_scale) # way * shot, frame, 16 * 16
        #maps = F.softmax(maps, dim=-1)
        # Norm attention maps
        max_value = maps.max(dim=-1,keepdim=True)[0]
        min_value = maps.min(dim=-1,keepdim=True)[0]
        maps = (maps - min_value) / (max_value - min_value + 1e-4) # way * shot, frame, 16*16
        # De-noise
        #zeros = torch.zeros_like(maps).cuda()
        #maps = torch.where(maps<0.3, zeros, maps)
        selected_maps = torch.bmm(indices, maps).view(-1, self.args.k, maps_scale, maps_scale) # way * shot, k, 16 , 16

        return indices, selected_feature, info_g, weight, selected_maps, feature


if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.way = 5
            self.shot = 1
            self.query_per_class = 1
            self.query_per_class_test = 1
            self.trans_dropout = 0.1
            self.seq_len = 16
            self.img_size = 224
            self.backbone = "resnet50"
            self.num_gpus = 1
            self.metric = 'cos'
            self.k = 8
            self.sigma = 0.05
            self.mini_size = 64
            self.sigma_decay = True
            self.num_samples = 500
            self.noise = 'normal'
            self.current_iter = 1
            self.lightCNN = 'shufflev2_0_5'
            self.sigma_step = 1000
            self.dropout = 0.1
            self.ada = True

    args = ArgsObject()
    S = Selector(args).cuda()
    input = torch.rand(args.way*args.shot*args.seq_len, 3, args.img_size, args.img_size).cuda() # Input: way*shot*frame, c, w, h
    print('Input Data shape:', input.shape)
    n, c, w, h = input.size()
    indices,_,_,_,_,_ = S(input) # Indice: way*shot, k, len
    input = input.view(args.way*args.shot, args.seq_len, -1) # Input: way*shot, len, c*w*h
    subset = torch.bmm(indices, input) # Output: way*shot, k, c*w*h
    subset = subset.view(-1, c, w, h) # Output: way*shot*k, c, w, h
    print('Data shape output by sampler:', subset.shape)