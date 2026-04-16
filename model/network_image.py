import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import numpy as np
from types import SimpleNamespace
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet18_Weights
from model.clstm import*
from model.vae import*
from model.pose_encoder import PoseEvidenceEncoder
from model.evidence_accumulator import LeakyDecisionAccumulator





def _check_finite_tensor(tensor, name):
    if tensor is None:
        return
    if torch.isnan(tensor).any():
        raise ValueError(f'{name} contains NaN values.')
    if torch.isinf(tensor).any():
        raise ValueError(f'{name} contains Inf values.')


class PTINet(nn.Module):
    def __init__(self, args):
        super(PTINet, self).__init__()

        if args.dataset=='jaad':
            self.size = 4
            self.ped_attribute_size=3
            self.ped_behavior_size=4
            self.scene_attribute_size=10

        elif args.dataset=='pie':
            self.size = 4
            self.ped_attribute_size=2
            self.ped_behavior_size=3
            self.scene_attribute_size=4

        elif args.dataset=='titan':
            self.size = 4
            self.ped_behavior_size=3
        else:
            'wrong dataset'

       
        self.num_layers=1
        self.latent_size=args.hidden_size
        self.use_pose = getattr(args, 'use_pose', False)
        self.use_decision_accumulator = getattr(args, 'use_decision_accumulator', False)
        self.use_fused_decoder_input = getattr(args, 'use_fused_decoder_input', False)
        
        self.speed_encoder = LSTMVAE(input_size=self.size, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)
        self.pos_encoder = LSTMVAE(input_size=self.size, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)
        self.loc_feat_proj = nn.Linear(self.size, args.hidden_size)
        self.debug_last_features = {}
        if self.use_pose:
            self.pose_encoder = PoseEvidenceEncoder(num_joints=17, out_dim=args.hidden_size)
            self.pose_direct_to_crossing = nn.Linear(args.hidden_size, args.hidden_size)
            self.fc_intention_pose_direct = nn.Linear(args.hidden_size, 2)
            self.pose_accumulator = LeakyDecisionAccumulator(feature_dim=args.hidden_size)
            self.belief_to_crossing = nn.Linear(args.hidden_size, args.hidden_size)
            self.fc_intention_belief = nn.Linear(args.hidden_size, 2)
        else:
            self.pose_encoder = None
            self.pose_direct_to_crossing = None
            self.fc_intention_pose_direct = None
            self.pose_accumulator = None
            self.belief_to_crossing = None
            self.fc_intention_belief = None
        


        if args.use_attribute == True:
            self.ped_behavior_encoder = LSTMVAE(input_size=self.ped_behavior_size, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)
            if args.dataset == 'jaad' or args.dataset == 'pie':         
                self.scene_attribute_encoder   =LSTMVAE(input_size=self.scene_attribute_size, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)
                self.mlp = nn.Sequential( nn.Linear(self.ped_attribute_size, 64),nn.ReLU(),nn.Linear(64, args.hidden_size),nn.ReLU() )

        if args.use_image==True:
            if args.image_network== 'resnet50':
                self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                self.resnet.fc = nn.Identity()
                self.img_encoder   = LSTMVAE(input_size=2048, hidden_size=args.hidden_size, latent_size=self.latent_size, device=args.device)

            elif args.image_network== 'resent18':
                self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
                self.resnet.fc = nn.Identity()
                self.img_encoder   = nn.LSTM(input_size=512, hidden_size=args.hidden_size,num_layers=self.num_layers,batch_first=True)
            elif args.image_network == 'clstm':
                self.clstm=ConvLSTM(input_channels=3, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, conv_stride=1,pool_kernel_size=(2, 2), step=5, effective_step=[4])
                self.pooling_h = nn.AdaptiveAvgPool2d((1, 1))
                self.pooling_c = nn.AdaptiveAvgPool2d((1, 1))
                self.linear_c= nn.Linear(in_features=32, out_features=512)
                self.linear_h = nn.Linear(in_features=32, out_features=512)

        if args.use_opticalflow== True:
            self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Identity()
            self.op_encoder   = nn.LSTM(input_size=2048, hidden_size=args.hidden_size,num_layers=self.num_layers,batch_first=True)

      
        
        self.pos_embedding = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=self.size),
                                           nn.ReLU())
        
        self.speed_decoder    = nn.LSTMCell(input_size=self.size, hidden_size=args.hidden_size)
        self.crossing_decoder = nn.LSTMCell(input_size=self.size, hidden_size=args.hidden_size)
        self.attrib_decoder = nn.LSTMCell(input_size=self.size, hidden_size=args.hidden_size)
        
        self.fc_speed    = nn.Linear(in_features=args.hidden_size, out_features=self.size)
        self.fc_crossing = nn.Linear(in_features=args.hidden_size, out_features=2)
        self.fc_attrib = nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=3), nn.ReLU())
        self.fc_intention = nn.Linear(in_features=args.hidden_size, out_features=2)
        
        self.hardtanh = nn.Hardtanh(min_val=-1*args.hardtanh_limit, max_val=args.hardtanh_limit)
        
        self.args = args
        
    def forward(self, speed=None, pos=None,ped_attribute=None,ped_behavior=None,scene_attribute=None,images=None,optical=None, pose=None, pose_conf=None, average=False):


        # 轨迹速度、位置：
        sloss, x_hat, zsp,hsp, (recon_loss, kld_loss) = self.speed_encoder(speed)
        hsp = hsp[0].squeeze(0)
        zsp=torch.mean(zsp,axis=1)
        # csp = csp.squeeze(0)
        
        ploss, x_hat, zpo,hpo, (recon_loss, kld_loss) = self.pos_encoder(pos)
        hpo = hpo[0].squeeze(0)
        zpo=torch.mean(zpo,axis=1)
        # cpo = cpo.squeeze(0)

        loc_feat_seq = self.loc_feat_proj(pos.float())
        app_feat_seq = None

        pose_evidence_seq = None
        pose_evidence_last = None
        belief_seq = None
        belief_last = None
        if self.use_pose:
            if pose is None or pose_conf is None:
                raise ValueError('pose and pose_conf must be provided when use_pose=True.')
            pose_evidence_seq = self.pose_encoder(pose, pose_conf)
            pose_evidence_last = pose_evidence_seq[:, -1, :]
            _check_finite_tensor(pose_evidence_seq, 'pose_evidence_seq')
            _check_finite_tensor(pose_evidence_last, 'pose_evidence_last')

            if self.use_decision_accumulator:
                belief_seq = self.pose_accumulator(pose_evidence_seq)
                belief_last = belief_seq[:, -1, :]
                _check_finite_tensor(belief_seq, 'belief_seq')
                _check_finite_tensor(belief_last, 'belief_last')

        self.debug_last_features = {
            'loc_feat_seq': loc_feat_seq.detach(),
            'app_feat_seq': app_feat_seq.detach() if app_feat_seq is not None else None,
            'pose_feat_seq': pose_evidence_seq.detach() if pose_evidence_seq is not None else None,
            'pose_evidence_seq': pose_evidence_seq.detach() if pose_evidence_seq is not None else None,
            'pose_evidence_last': pose_evidence_last.detach() if pose_evidence_last is not None else None,
            'belief_seq': belief_seq.detach() if belief_seq is not None else None,
            'belief_last': belief_last.detach() if belief_last is not None else None,
            'fused_feat_seq': None,
        }


        if self.args.use_attribute == True:
            pbloss, x_hat, zpa,hpa, (recon_loss, kld_loss)  = self.ped_behavior_encoder (ped_behavior)
            hpa = hpa[0].squeeze(0)
            zpa = torch.mean(zpa,axis=1)

            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':  

                psloss, x_hat, zsa,hsa, (recon_loss, kld_loss)  = self.scene_attribute_encoder(scene_attribute)
                hsa = hsa[0].squeeze(0)
                zsa =torch.mean(zsa,axis=1)

                pb=self.mlp(ped_attribute)

        if self.args.use_image==True:
            batch_size, seq_len, c, h, w = images.size()

            if self.args.image_network=='clstm':
                batch_size, seq_len, c, h, w = images.size()
                _,(himg, cimg)=self.clstm(images)
                himg=self.pooling_h(himg).view(himg.size(0), -1)
                himg=self.linear_h(himg)

                cimg=self.pooling_c(cimg).view(cimg.size(0), -1)
                cimg=self.linear_c(cimg)
            else:
                images = images.view(batch_size * seq_len, c, h, w)
                img_feats = self.resnet(images)
                img_feats = img_feats.view(batch_size, seq_len, -1)

                imgloss, x_hat, zim,him, (recon_loss, kld_loss) = self.img_encoder(img_feats)
                him = him[0].squeeze(0)
                zim = torch.mean(zim,axis=1)
        if self.args.use_opticalflow==True:
            batch_size_op, seq_len_op, c_op, h_op, w_op = optical.size()
            optical = optical.view(batch_size * seq_len_op, c_op, h_op, w_op)
            op_feats = self.resnet(optical)
            op_feats = op_feats.view(batch_size, seq_len_op, -1)

            _, (himg_op, cimg_op) = self.op_encoder(op_feats)
            himg_op = himg_op[-1,:,:].squeeze(0)
            cimg_op = cimg_op[-1,:,:].squeeze(0)


        outputs=[]

        if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   

            outputs.append(ploss+sloss+pbloss+psloss)
        else:
            outputs.append(ploss+sloss+pbloss)



        #  _, (hsp, csp) = self.speed_encoder(speed.permute(1,0,2))
        # hsp = hsp.squeeze(0)
        # csp = csp.squeeze(0)
        
        # _, (hpo, cpo) = self.pos_encoder(pos.permute(1,0,2))
        # hpo = hpo.squeeze(0)
        # cpo = cpo.squeeze(0)
        # outputs = []

        #  if self.args.use_attribute == True:
        #     _, (hpa, cpa) = self.ped_behavior_encoder (ped_behavior)
        #     hpa = hpa[-1,:,:].squeeze(0)
        #     cpa = cpa[-1,:,:].squeeze(0)

        #     _, (hsa, csa) = self.scene_attribute_encoder(scene_attribute)
        #     hsa = hsa[-1,:,:].squeeze(0)
        #     csa = csa[-1,:,:].squeeze(0)

        #     pb=self.mlp(ped_attribute)

        # if self.args.use_image==True:
        #     batch_size, seq_len, c, h, w = images.size()
        #     images = images.view(batch_size * seq_len, c, h, w)
        #     img_feats = self.resnet(images)
        #     img_feats = img_feats.view(batch_size, seq_len, -1)

        #     _, (himg, cimg) = self.img_encoder(img_feats)
        #     himg = himg[-1,:,:].squeeze(0)
        #     cimg = cimg[-1,:,:].squeeze(0)

        # outputs = []
        
        
        speed_outputs    = torch.tensor([], device=self.args.device)
        in_sp = speed[:,-1,:]
        # multimodal fusion
        #  hidden / latent 向量直接相加：
        hds = hpo + hsp
        zds = zpo + zsp

        if self.args.use_attribute == True:
            # init
            hds = hds + hpa  
            # add 
            zds = zds +zpa 
            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':  
                hds = hds+ hsa  + hpa  + pb 
                zds = zds +zpa + zsa + pb

        if self.args.use_image ==True:
            hds=hds + himg
            zds=zds + cimg 

        if self.args.use_opticalflow ==True:
            hds=hds + himg_op
            zds=zds + cimg_op 

        for i in range(self.args.output//self.args.skip):
            hds, zds         = self.speed_decoder(in_sp, (hds, zds))
            speed_output     = self.hardtanh(self.fc_speed(hds))
            speed_outputs    = torch.cat((speed_outputs, speed_output.unsqueeze(1)), dim = 1)
            in_sp            = speed_output.detach()
            
        outputs.append(speed_outputs)

        
        crossing_outputs = torch.tensor([], device=self.args.device)
        in_cr = pos[:,-1,:]
        
        hdc = hpo + hsp
        zdc = zpo + zsp

        if self.args.use_attribute == True:
            hdc = hdc  + hpa  
            zdc = zdc+ zpa 
            if self.args.dataset == 'jaad' or self.args.dataset == 'pie':   
                hdc = hdc+ hsa  + hpa  + pb 
                zdc = zdc +zpa + zsa + pb


        if self.args.use_image ==True:
            hdc=hdc + himg
            zdc=zdc + cimg 

        if self.args.use_opticalflow ==True:
            hdc=hdc + himg_op
            zdc=zdc + cimg_op 

        if belief_last is not None:
            hdc = hdc + self.belief_to_crossing(belief_last)
        elif pose_evidence_last is not None:
            hdc = hdc + self.pose_direct_to_crossing(pose_evidence_last)

        crossing_hidden_states = []

        for i in range(self.args.output//self.args.skip):
            hdc, zdc         = self.crossing_decoder(in_cr, (hdc, zdc))
            crossing_hidden_states.append(hdc.unsqueeze(1))
            crossing_output  = self.fc_crossing(hdc)
            in_cr            = self.pos_embedding(hdc).detach()
            crossing_outputs = torch.cat((crossing_outputs, crossing_output.unsqueeze(1)), dim = 1)

        intention_context = torch.cat(crossing_hidden_states, dim=1).max(dim=1)[0]
        _check_finite_tensor(crossing_outputs, 'crossing_outputs')
        if belief_last is not None:
            intention_logits = self.fc_intention_belief(belief_last)
        elif pose_evidence_last is not None:
            intention_logits = self.fc_intention_pose_direct(pose_evidence_last)
        else:
            intention_logits = self.fc_intention(intention_context)
        _check_finite_tensor(intention_logits, 'intention_logits')

        outputs.append(crossing_outputs)
        outputs.append(intention_logits)
    
        if average:
            intention = torch.argmax(intention_logits, dim=1)
            outputs.append(intention)
        


        return tuple(outputs)


def _build_smoke_args(use_pose, use_decision_accumulator):
    return SimpleNamespace(
        dataset='jaad',
        hidden_size=128,
        device=torch.device('cpu'),
        use_pose=use_pose,
        use_decision_accumulator=use_decision_accumulator,
        use_fused_decoder_input=False,
        use_attribute=True,
        use_image=False,
        use_opticalflow=False,
        output=4,
        skip=1,
        hardtanh_limit=1.0,
    )


def _build_smoke_inputs():
    batch_size = 2
    time_steps = 5
    return {
        'speed': torch.randn(batch_size, time_steps, 4),
        'pos': torch.randn(batch_size, time_steps, 4),
        'ped_attribute': torch.randn(batch_size, 3),
        'ped_behavior': torch.randn(batch_size, time_steps, 4),
        'scene_attribute': torch.randn(batch_size, time_steps, 10),
        'images': torch.randn(batch_size, time_steps, 3, 32, 32),
        'pose': torch.randn(batch_size, time_steps, 17, 2),
        'pose_conf': torch.rand(batch_size, time_steps, 17),
    }


def _smoke_test():
    torch.manual_seed(0)

    baseline_args = _build_smoke_args(use_pose=False, use_decision_accumulator=False)
    baseline_model = PTINet(baseline_args)
    baseline_model.eval()
    baseline_inputs = _build_smoke_inputs()

    with torch.no_grad():
        baseline_outputs = baseline_model(
            speed=baseline_inputs['speed'],
            pos=baseline_inputs['pos'],
            ped_attribute=baseline_inputs['ped_attribute'],
            ped_behavior=baseline_inputs['ped_behavior'],
            scene_attribute=baseline_inputs['scene_attribute'],
            images=baseline_inputs['images'],
        )

    baseline_crossing = baseline_outputs[2]
    baseline_intention = baseline_outputs[3]
    assert baseline_crossing.shape == (2, 4, 2), (
        f'unexpected baseline crossing shape: {tuple(baseline_crossing.shape)}'
    )
    assert baseline_intention.shape == (2, 2), (
        f'unexpected baseline intention shape: {tuple(baseline_intention.shape)}'
    )
    _check_finite_tensor(baseline_crossing, 'baseline_crossing')
    _check_finite_tensor(baseline_intention, 'baseline_intention')

    direct_args = _build_smoke_args(use_pose=True, use_decision_accumulator=False)
    direct_model = PTINet(direct_args)
    direct_model.eval()
    direct_inputs = _build_smoke_inputs()

    with torch.no_grad():
        direct_outputs = direct_model(
            speed=direct_inputs['speed'],
            pos=direct_inputs['pos'],
            ped_attribute=direct_inputs['ped_attribute'],
            ped_behavior=direct_inputs['ped_behavior'],
            scene_attribute=direct_inputs['scene_attribute'],
            images=direct_inputs['images'],
            pose=direct_inputs['pose'],
            pose_conf=direct_inputs['pose_conf'],
        )

    direct_crossing = direct_outputs[2]
    direct_intention = direct_outputs[3]
    direct_pose_evidence_seq = direct_model.debug_last_features['pose_evidence_seq']
    direct_pose_evidence_last = direct_model.debug_last_features['pose_evidence_last']
    direct_belief_seq = direct_model.debug_last_features['belief_seq']
    direct_belief_last = direct_model.debug_last_features['belief_last']

    assert direct_crossing.shape == (2, 4, 2), (
        f'unexpected direct crossing shape: {tuple(direct_crossing.shape)}'
    )
    assert direct_intention.shape == (2, 2), (
        f'unexpected direct intention shape: {tuple(direct_intention.shape)}'
    )
    assert direct_pose_evidence_seq.shape == (2, 5, 128), (
        f'unexpected direct pose_evidence_seq shape: {tuple(direct_pose_evidence_seq.shape)}'
    )
    assert direct_pose_evidence_last.shape == (2, 128), (
        f'unexpected direct pose_evidence_last shape: {tuple(direct_pose_evidence_last.shape)}'
    )
    assert direct_belief_seq is None, 'direct belief_seq should be None.'
    assert direct_belief_last is None, 'direct belief_last should be None.'
    _check_finite_tensor(direct_crossing, 'direct_crossing')
    _check_finite_tensor(direct_intention, 'direct_intention')
    _check_finite_tensor(direct_pose_evidence_seq, 'direct_pose_evidence_seq')
    _check_finite_tensor(direct_pose_evidence_last, 'direct_pose_evidence_last')

    accumulator_args = _build_smoke_args(use_pose=True, use_decision_accumulator=True)
    accumulator_model = PTINet(accumulator_args)
    accumulator_model.eval()
    accumulator_inputs = _build_smoke_inputs()

    with torch.no_grad():
        accumulator_outputs = accumulator_model(
            speed=accumulator_inputs['speed'],
            pos=accumulator_inputs['pos'],
            ped_attribute=accumulator_inputs['ped_attribute'],
            ped_behavior=accumulator_inputs['ped_behavior'],
            scene_attribute=accumulator_inputs['scene_attribute'],
            images=accumulator_inputs['images'],
            pose=accumulator_inputs['pose'],
            pose_conf=accumulator_inputs['pose_conf'],
        )

    accumulator_crossing = accumulator_outputs[2]
    accumulator_intention = accumulator_outputs[3]
    accumulator_pose_evidence_seq = accumulator_model.debug_last_features['pose_evidence_seq']
    accumulator_pose_evidence_last = accumulator_model.debug_last_features['pose_evidence_last']
    accumulator_belief_seq = accumulator_model.debug_last_features['belief_seq']
    accumulator_belief_last = accumulator_model.debug_last_features['belief_last']

    assert accumulator_crossing.shape == (2, 4, 2), (
        f'unexpected accumulator crossing shape: {tuple(accumulator_crossing.shape)}'
    )
    assert accumulator_intention.shape == (2, 2), (
        f'unexpected accumulator intention shape: {tuple(accumulator_intention.shape)}'
    )
    assert accumulator_pose_evidence_seq.shape == (2, 5, 128), (
        f'unexpected accumulator pose_evidence_seq shape: {tuple(accumulator_pose_evidence_seq.shape)}'
    )
    assert accumulator_pose_evidence_last.shape == (2, 128), (
        f'unexpected accumulator pose_evidence_last shape: {tuple(accumulator_pose_evidence_last.shape)}'
    )
    assert accumulator_belief_seq.shape == (2, 5, 128), (
        f'unexpected accumulator belief_seq shape: {tuple(accumulator_belief_seq.shape)}'
    )
    assert accumulator_belief_last.shape == (2, 128), (
        f'unexpected accumulator belief_last shape: {tuple(accumulator_belief_last.shape)}'
    )
    _check_finite_tensor(accumulator_crossing, 'accumulator_crossing')
    _check_finite_tensor(accumulator_intention, 'accumulator_intention')
    _check_finite_tensor(accumulator_pose_evidence_seq, 'accumulator_pose_evidence_seq')
    _check_finite_tensor(accumulator_pose_evidence_last, 'accumulator_pose_evidence_last')
    _check_finite_tensor(accumulator_belief_seq, 'accumulator_belief_seq')
    _check_finite_tensor(accumulator_belief_last, 'accumulator_belief_last')

    has_nan = bool(
        torch.isnan(baseline_crossing).any()
        or torch.isnan(baseline_intention).any()
        or torch.isnan(direct_crossing).any()
        or torch.isnan(direct_intention).any()
        or torch.isnan(direct_pose_evidence_seq).any()
        or torch.isnan(direct_pose_evidence_last).any()
        or torch.isnan(accumulator_crossing).any()
        or torch.isnan(accumulator_intention).any()
        or torch.isnan(accumulator_pose_evidence_seq).any()
        or torch.isnan(accumulator_pose_evidence_last).any()
        or torch.isnan(accumulator_belief_seq).any()
        or torch.isnan(accumulator_belief_last).any()
    )
    has_inf = bool(
        torch.isinf(baseline_crossing).any()
        or torch.isinf(baseline_intention).any()
        or torch.isinf(direct_crossing).any()
        or torch.isinf(direct_intention).any()
        or torch.isinf(direct_pose_evidence_seq).any()
        or torch.isinf(direct_pose_evidence_last).any()
        or torch.isinf(accumulator_crossing).any()
        or torch.isinf(accumulator_intention).any()
        or torch.isinf(accumulator_pose_evidence_seq).any()
        or torch.isinf(accumulator_pose_evidence_last).any()
        or torch.isinf(accumulator_belief_seq).any()
        or torch.isinf(accumulator_belief_last).any()
    )

    print(f'baseline crossing shape: {tuple(baseline_crossing.shape)}')
    print(f'baseline intention shape: {tuple(baseline_intention.shape)}')
    print(f'direct crossing shape: {tuple(direct_crossing.shape)}')
    print(f'direct intention shape: {tuple(direct_intention.shape)}')
    print(f'direct pose_evidence_seq shape: {tuple(direct_pose_evidence_seq.shape)}')
    print(f'direct pose_evidence_last shape: {tuple(direct_pose_evidence_last.shape)}')
    print(f'direct belief_seq: {direct_belief_seq}')
    print(f'direct belief_last: {direct_belief_last}')
    print(f'accumulator crossing shape: {tuple(accumulator_crossing.shape)}')
    print(f'accumulator intention shape: {tuple(accumulator_intention.shape)}')
    print(f'accumulator pose_evidence_seq shape: {tuple(accumulator_pose_evidence_seq.shape)}')
    print(f'accumulator pose_evidence_last shape: {tuple(accumulator_pose_evidence_last.shape)}')
    print(f'accumulator belief_seq shape: {tuple(accumulator_belief_seq.shape)}')
    print(f'accumulator belief_last shape: {tuple(accumulator_belief_last.shape)}')
    print(f'has_nan: {has_nan}')
    print(f'has_inf: {has_inf}')


if __name__ == '__main__':
    _smoke_test()
