# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


import math
import torch.nn.functional as F
from utils.general import xyxy2xywh, xywh2xyxy,xywhn2xyxy,xywh2xyxy_
import warnings

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch



class mask_loss(nn.Module):
    def __init__(self,T = 1,w_fg = 0.02, w_bg = 0.01 ,r=10):
        super().__init__()
        self.T = T
        self.w_fg =w_fg
        self.w_bg = w_bg
        self.r = r
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1,padding=1)

    def forward(self, fea_s, fea_t, target, device):
        mask_fg = self.Fg_mask_generator(fea_s, target, device,self.r)
        # fea_t_m = self.maxpool(fea_t)#
        # fea_s_m = self.maxpool(fea_s)
        # loss_max = self.loss_compute(fea_s_m, fea_t_m, mask_fg)
        loss_ori = self.loss_compute(fea_s  , fea_t  , mask_fg)
        #loss = (loss_ori + loss_max) / 2
        return loss_ori #loss



    def loss_compute(self, fea_s, fea_t, mask_fg):

        mask_bg = 1 - mask_fg
        loss_abs = torch.abs(fea_t - fea_s)  # (2,3,1024,1024)

        loss_bg = torch.mul(mask_bg, loss_abs)

        loss_bg[loss_bg < torch.mean(loss_bg)] = 0  # 背景部分差不多得了##

        loss_fg = torch.mul(mask_fg, loss_abs)

        loss = self.w_fg * loss_fg + self.w_bg * loss_bg  # (B,C,H,W)
        loss = torch.mean(loss)
        #print(torch.mean(loss_bg),'...')##
        return loss#

    def loss_compute_attention(self, fea_s, fea_t, mask_fg):
        B, C, H, W = fea_s.shape
        #mask_fg = self.Fg_mask_generator(fea_s, target, device, r=10)#
        # spatial_attention_t = fea_t.mean(axis=1, keepdim=True)
        # spatial_attention_t = (H * W * F.softmax((spatial_attention_t / self.T).view(B, -1), dim=1)).view(B, 1, H, W)
        spatial_attention_s = fea_s.mean(axis=1, keepdim=True)
        spatial_attention_s = (H * W * F.softmax((spatial_attention_s / self.T).view(B, -1), dim=1)).view(B, 1, H, W)        #from  FGD

        # channel_attention_t = fea_t.mean(axis=2, keepdim=False).mean(axis=2,keepdim=False)  # value.mean(axis=2,keepdim=False)对第二个维度H求均值，且不维持维度，对其结果的第二个维度W再求均值，且也不位置维度  (2,3)
        # channel_attention_t = C * F.softmax(channel_attention_t / self.T, dim=1).view(B, C, 1, 1)
        channel_attention_s = fea_s.mean(axis=2, keepdim=False).mean(axis=2,keepdim=False)
        channel_attention_s = C * F.softmax(channel_attention_s / self.T, dim=1).view(B, C, 1, 1)

        #fea_t_attention = fea_t * spatial_attention_t * channel_attention_t
        fea_s_attention = fea_s * spatial_attention_s * channel_attention_s

        mask_bg = 1 - mask_fg
        #loss_abs = torch.abs(fea_t_attention - fea_s_attention)  # (2,3,1024,1024)
        loss_abs =  torch.abs(fea_t - fea_s)

        loss_bg = torch.mul(mask_bg, loss_abs)
        loss_bg[loss_bg < torch.mean(fea_s_attention*loss_bg)] = 0  # 背景部分差不多得了

        loss_fg = torch.mul(mask_fg, loss_abs)

        loss = self.w_fg * loss_fg + self.w_bg * loss_bg  #(B,C,H,W)
        loss = torch.mean(loss)
        return loss


    def Fg_mask_generator(self, x, targets, device, r=10):
        B, C, H, W = x.shape
        mask = torch.zeros([B, 1, H, W]).to(device)
        targets_labels = targets.clone()
        warnings.filterwarnings("ignore", message="__floordiv__ is deprecated")
        for i in range(B):

            labels = targets_labels[targets_labels[:, 0] == i]  # 选择对应batch的标签
            labels = labels[..., 2:]
            labels[:, 0] *= W
            labels[:, 2] *= W
            labels[:, 1] *= H
            labels[:, 3] *= H
            for label in labels:  # x.shape是torch.Size([4])  这里他不是(1,4)
                r_w = label[2] // r if label[2] // r != 0 else 1
                r_h = label[3] // r if label[3] // r != 0 else 1
                label = xywh2xyxy_(label)
                label = label.tolist()
                x1 = int(math.floor(label[0]) - r_w if label[0] - r_w > 0 else 0)
                y1 = int(math.floor(label[1]) - r_h if label[1] - r_h > 0 else 0)
                x2 = int(math.ceil (label[2]) + r_w if label[2] + r_w < W else W)
                y2 = int(math.ceil (label[3]) + r_h if label[3] + r_h < H else H)
                mask[i, :, y1:y2, x1:x2] = 1  # 第i个batch三个维度  i,1,1024,1024     前景部分全为1
        # x_fg = torch.mul(mask, x)  # 前景图
        return mask