# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.

Models and datasets download automatically from the latest YOLOv5 release.
Models: https://github.com/ultralytics/yolov5/tree/master/models
Datasets: https://github.com/ultralytics/yolov5/tree/master/data
Tutorial: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (RECOMMENDED)
    $ python path/to/train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch
"""

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
#from models.yolo import Model
from models.yolo_1024_fpn_cbamconcat import Model as Model_1024  #hxc
from models.yolo_512 import Model as Model_512
#from models.yolo_256 import Model as Model_256

from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, check_amp, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss,mask_loss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_labels
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data,  cfg_low,cfg_high, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg_low,opt.cfg_high,\
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # Directories
    # w = save_dir / 'weights'  # weights dir
    # (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    w_1024 = save_dir / 'weights_1024'  # weights dir
    (w_1024.parent if evolve else w_1024).mkdir(parents=True, exist_ok=True)  # make dir
    last_1024, best_1024 = w_1024 / 'last.pt', w_1024 / 'best.pt'

    w_512 = save_dir / 'weights_512'  # weights dir
    (w_512.parent if evolve else w_512).mkdir(parents=True, exist_ok=True)  # make dir
    last_512, best_512 = w_512 / 'last.pt', w_512 / 'best.pt'

    # w_256 = save_dir / 'weights_256'  # weights dir
    # (w_256.parent if evolve else w_256).mkdir(parents=True, exist_ok=True)  # make dir
    # last_256, best_256 = w_256 / 'last.pt', w_256 / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.clearml:
            data_dict = loggers.clearml.data_dict  # None if no ClearML dataset or filled in by ClearML
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    #k=int(opt.imgsz[0]/opt.imgsz[1])
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak

        model_1024 = Model_1024(cfg_1024 or ckpt['model_1024'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model_512 = Model_512(cfg_512  or ckpt['model_512'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        #model_256 = Model_512(cfg_256  or ckpt['model_256'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)

        exclude_1024 = ['anchor'] if (cfg_1024 or hyp.get('anchors')) and not resume else []  # exclude keys
        csd_1024 = ckpt['model_1024'].float().state_dict()  # checkpoint state_dict as FP32
        csd_1024 = intersect_dicts(csd_1024, model_1024.state_dict(), exclude=exclude_1024)  # intersect
        model_1024.load_state_dict(csd_1024, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd_1024)}/{len(model_1024.state_dict())} items from {weights}')  # report

        exclude_512 = ['anchor'] if (cfg_512 or hyp.get('anchors')) and not resume else []  # exclude keys
        csd_512= ckpt['model_512'].float().state_dict()  # checkpoint state_dict as FP32
        csd_512 = intersect_dicts(csd_512, model_512.state_dict(), exclude=exclude_512)  # intersect
        model_512.load_state_dict(csd_512, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd_512)}/{len(model_512.state_dict())} items from {weights}')  # report

        # exclude_256 = ['anchor'] if (cfg_256 or hyp.get('anchors')) and not resume else []  # exclude keys
        # csd_256 = ckpt['model_256'].float().state_dict()  # checkpoint state_dict as FP32
        # csd_256 = intersect_dicts(csd_256, model_256.state_dict(), exclude=exclude_256)  # intersect
        # model_256.load_state_dict(csd_256, strict=False)  # load
        # LOGGER.info(f'Transferred {len(csd_256)}/{len(model_256.state_dict())} items from {weights}')  # report

    else:
        model_1024 = Model_1024(cfg_high, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        model_512 = Model_512(cfg_low, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        #model_256 = Model_256(cfg_256, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model_1024)&check_amp(model_512)#&check_amp(model_256) # check AMP


    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze   #没动
    for k, v in model_512.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs_1024 = max(int(model_1024.stride.max()), 32)  # grid size (max stride)  #最小的也要大于32
    gs_512 = max(int(model_512.stride.max()), 32)  # grid size (max stride)  #最小的也要大于32
    #gs_256 = max(int(model_256.stride.max()), 32)  # grid size (max stride)  #最小的也要大于32
    #imgsz  = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    imgsz, imgsz_test = [check_img_size(x, gs_512) for x in opt.imgsz]   #要保证三个模型输入都满足
    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model_512, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model_1024,model_512, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])  #用512的模型优化器

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema_1024 = ModelEMA(model_1024) if RANK in {-1, 0} else None
    ema_512 = ModelEMA(model_512) if RANK in {-1, 0} else None
    #ema_256 = ModelEMA(model_256) if RANK in {-1, 0} else None

    # Resume
    best_fitness_1024 = 0.0
    best_fitness_512 = 0.0
    #best_fitness_256,
    start_epoch = 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema_512, weights, epochs, resume)
        del ckpt, csd_512

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model_1024 = torch.nn.DataParallel(model_1024)
        model_512 = torch.nn.DataParallel(model_512)
        #model_256 = torch.nn.DataParallel(model_256)
    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model_1024 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_1024).to(device)
        model_512  = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_512).to(device)
        #model_256 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_256).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs_1024,                     #看看要训练哪一个
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in {-1, 0}:
        val_loader_1024 = create_dataloader(val_path,
                                       imgsz,
                                       #batch_size // WORLD_SIZE * 2,
                                       batch_size // WORLD_SIZE ,
                                       gs_1024,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
        val_loader_512 = create_dataloader(val_path,
                                       imgsz_test,
                                       #batch_size // WORLD_SIZE * 2,
                                       batch_size // WORLD_SIZE ,
                                       gs_512,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
        # val_loader_256 = create_dataloader(val_path,
        #                                imgsz_test/2,
        #                                #batch_size // WORLD_SIZE * 2,
        #                                batch_size // WORLD_SIZE ,
        #                                gs_256,
        #                                single_cls,
        #                                hyp=hyp,
        #                                cache=None if noval else opt.cache,
        #                                rect=True,
        #                                rank=-1,
        #                                workers=workers * 2,
        #                                pad=0.5,
        #                                prefix=colorstr('val: '))[0]

        if not resume:
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model_1024, thr=hyp['anchor_t'], imgsz=imgsz)  #这里是imgze还是img_test
                check_anchors(dataset, model=model_512, thr=hyp['anchor_t'], imgsz=imgsz/2)
                #check_anchors(dataset, model=model_256, thr=hyp['anchor_t'], imgsz=imgsz/4)
            model_1024.half().float()
            model_512. half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end')

    # DDP mode
    if cuda and RANK != -1:
        model_1024 = smart_DDP(model_1024)
        model_512 = smart_DDP(model_512)
        #model_256 = smart_DDP(model_256)
    # Model attributes
    nl = de_parallel(model_512).model[-1].nl  # number of detection layers (to scale hyps)   3个是一样的  所以一次就够
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model_1024.nc = nc  # attach number of classes to model
    model_512.nc = nc  # attach number of classes to model
    #model_256.nc = nc  # attach number of classes to model

    model_1024.hyp = hyp  # attach hyperparameters to model
    model_512.hyp = hyp
    #model_256.hyp = hyp

    model_1024.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model_512.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    #model_256.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    model_1024.names = names
    model_512.names = names
    #model_256.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    #results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    results_1024 = (0, 0, 0, 0, 0, 0, 0)
    results_512 = (0, 0, 0, 0, 0, 0, 0)
    #results_256 = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss_1024 = ComputeLoss(model_1024)  # init loss class+
    compute_loss_512 = ComputeLoss(model_512)
    distill_loss = mask_loss(T=1, w_fg=2, w_bg=1, r=10)
    #compute_loss_256 = ComputeLoss(model_256)

    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz_test} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model_1024.train()
        model_512.train()
    #    model_256.train()
        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model_1024.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights  3个模型的class_weights是一样的
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss_1024 = torch.zeros(3, device=device)  # mean losses
        mloss_512 = torch.zeros(3, device=device)
        #mloss_256 = torch.zeros(3, device=device)
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        #LOGGER.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'distill','box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            
            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs_1024) // gs_1024 * gs_1024  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs_1024) * gs_1024 for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
                    
            imgs = imgs.to(device, non_blocking=True).float() / 255 # uint8 to float32, 0-255 to 0.0-1.0 
            
            imgs_1024 = imgs                       #2,3,1024,1024#hxc
            imgs_512 = F.interpolate(imgs_1024, size=[i // 2 for i in imgs_1024.size()[2:]], mode='bilinear',align_corners=True)      #2,3,512,512  #hxc
            #imgs_256 = F.interpolate(imgs_512,  size=[i // 2 for i in imgs_512 .size()[2:]], mode='bilinear',align_corners=True)
            #print(imgs_1024.shape,imgs_512.shape,imgs_256.shape)

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # # Multi-scale
            # if opt.multi_scale:
            #     sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs_1024) // gs_1024 * gs_1024  # size
            #     sf = sz / max(imgs.shape[2:])  # scale factor
            #     if sf != 1:
            #         ns = [math.ceil(x * sf / gs_1024) * gs_1024 for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
            #         imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):
                #pred = model(imgs)  # forward   pred:list[Tensor:(1,3,64,64,13)]  sr_out:{Tensor:(1,3,512,512)}
                #pred, sr_out, _ = model(imgs)
                pred_1024, feature_1024 = model_1024(imgs_1024)  # forward  #feature最后一层为一个列表其中包含的是检测头输出的结果[1,3,64,64,13],其中13是5+8 前五个值为框位置和类别
                pred_512, feature_512 = model_512(imgs_512)
                #pred_256, feature_256 = model_256(imgs_256)
                loss_1024, loss_items_1024 = compute_loss_1024(pred_1024 , targets.to(device))  # loss scaled by batch_size
                loss_512, loss_items_512 = compute_loss_512(pred_512, targets.to(device))
                #loss_256, loss_items_256 = compute_loss_256(pred_256, targets.to(device))#

                #loss_distill_total = 0.01 * torch.nn.L1Loss()(feature_512[-2], feature_1024[-2])  #
                #loss_distill_total = 1 * torch.nn.L1Loss()(feature_512[-2], feature_1024[-2])  #
                loss_distill_total = torch.nn.L1Loss()(feature_512[-2], feature_1024[-2])  ##
                # loss_distill_total = 0.0
                # 
                # for i in range(len(feature_512) - 1):
                #     #print(feature_256[i].shape,feature_512[i+1].shape,'.........................................................')
                #     #loss_fea_256_512 = 0.5 * torch.nn.L1Loss()(feature_256[i], feature_512[i+1])#
                #     #loss_fea_512_1024= 0.5 * torch.nn.L1Loss()(feature_512[i], feature_1024[i+1])#单向连接
                #     if opt.mask:n
                #         loss_fea_layer = 0.1 * (1 + 0.1 * i) * distill_loss(feature_512[i], feature_1024[i + 1], targets, device)
                #     else:
                #         loss_fea_layer = 0.1 * (1 + 0.1 * i) * torch.nn.L1Loss()(feature_512[i], feature_1024[i + 1])
                #     #loss_fea_layer =  distill_loss(feature_512[i], feature_1024[i + 1], targets, device)
                #     #loss_fea_512_1024 = 0.1 * (1 + 0.1 * i) * torch.nn.L1Loss()(feature_512[i],feature_1024[i + 1])  # 单向连接
                #     loss_distill_total += loss_fea_layer / len(feature_512)
                #print(loss_distill_total)
                loss = 1.5 * loss_1024 + loss_512 + opt.alpha * loss_distill_total#

                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode

                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()


            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients     optimizer
                torch.nn.utils.clip_grad_norm_(model_1024.parameters(), max_norm=10.0)
                torch.nn.utils.clip_grad_norm_(model_512.parameters(), max_norm=10.0)
                #torch.nn.utils.clip_grad_norm_(model_256.parameters(), max_norm=10.0)# clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema_512:
                    ema_1024.update(model_1024)
                    ema_512.update(model_512)
                #    ema_256.update(model_256)
                last_opt_step = ni



            # Log
            if RANK in {-1, 0}:
                mloss_1024 = (mloss_1024 * i + loss_items_1024) / (i + 1)  # update mean losses
                mloss_512 = (mloss_512 * i + loss_items_512) / (i + 1)
                #mloss_256 = (mloss_256 * i + loss_items_256) / (i + 1)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                # pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                #                      (f'{epoch}/{epochs - 1}', mem, *mloss_1024, targets.shape[0], imgs.shape[-1]))n
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss_1024, targets.shape[0], imgs.shape[-1]))
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss_512, targets.shape[0], imgs.shape[-1]))
                # pbar.set_description(('%10s' * 2 + '%10.4g' * 5) %
                #                      (f'{epoch}/{epochs - 1}', mem, *mloss_256, targets.shape[0], imgs.shape[-1]))

                callbacks.run('on_train_batch_end', ni, model_1024, imgs, targets, paths, plots)#!!!!
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema_1024.update_attr(model_1024, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results_1024, maps_1024, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE ,
                                           imgsz=imgsz,
                                           half=amp,
                                           model=ema_1024.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader_1024,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss_1024,)
                                           #sr=opt.sr)

            ema_512.update_attr(model_512, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:  # Calculate mAP
                results_512, maps_512, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE ,
                                           imgsz=imgsz_test,
                                           half=amp,
                                           model=ema_512.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader_512 ,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss_512,)

            # ema_256.update_attr(model_256, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # if not noval or final_epoch:  # Calculate mAP
            #     results_256, maps_256, _ = val.run(data_dict,
            #                                batch_size=batch_size // WORLD_SIZE ,
            #                                imgsz=imgsz_test/2,
            #                                half=amp,
            #                                model=ema_256.ema,
            #                                single_cls=single_cls,
            #                                dataloader= val_loader_256 ,
            #                                save_dir=save_dir,
            #                                plots=False,
            #                                callbacks=callbacks,
            #                                compute_loss=compute_loss_256,)

            # Update best mAP
            fi_1024 = fitness(np.array(results_1024).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            #stop = stopper(epoch=epoch, fitness=fi_1024)  # early stop check
            if fi_1024 > best_fitness_1024:
                best_fitness_1024 = fi_1024

            fi_512 = fitness(np.array(results_512).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi_512)  # early stop check
            if fi_512 > best_fitness_512:
                best_fitness_512 = fi_512

            # fi_256 = fitness(np.array(results_256).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            # #stop = stopper(epoch=epoch, fitness=fi_256)  # early stop check
            # if fi_256 > best_fitness_256:
            #     best_fitness_256 = fi_256

            # if opt.sr == False:
            #     sr_loss = torch.tensor([0.0])
            log_vals_1024 = list(mloss_1024) + list(results_1024) + lr                     #hxc
            log_vals_512 = list(mloss_512) + list(results_512) + lr
            #log_vals_256 = list(mloss_256) + list(results_256) + lr
            # log_vals.insert(0, loss_fea)
            # log_vals_1024.insert(0, loss_distill_total)#
            # log_vals_512.insert(0, loss_distill_total)#
            # log_vals.insert(0, loss)
            callbacks.run('on_fit_epoch_end', log_vals_1024, epoch,best_fitness_1024, fi_1024)
            callbacks.run('on_fit_epoch_end', log_vals_512, epoch, best_fitness_1024, fi_512)
            #callbacks.run('on_fit_epoch_end', log_vals_256, epoch, best_fitness_1024, fi_256)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt_1024 = {
                    'epoch': epoch,
                    'best_fitness': best_fitness_1024,
                    'model': deepcopy(de_parallel(model_1024)).half(),
                    'ema': deepcopy(ema_1024.ema).half(),
                    'updates': ema_1024.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()}

                ckpt_512 = {
                    'epoch': epoch,
                    'best_fitness': best_fitness_512,
                    'model': deepcopy(de_parallel(model_512)).half(),
                    'ema': deepcopy(ema_512.ema).half(),
                    'updates': ema_512.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()}

                # ckpt_256 = {
                #     'epoch': epoch,
                #     'best_fitness': best_fitness_256,
                #     'model': deepcopy(de_parallel(model_256)).half(),
                #     'ema': deepcopy(ema_256.ema).half(),
                #     'updates': ema_256.updates,
                #     'optimizer': optimizer.state_dict(),
                #     'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                #     'opt': vars(opt),
                #     'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt_1024, last_1024)  #Save last
                torch.save(ckpt_512, last_512)

                if best_fitness_1024 == fi_1024:     #Save best
                    torch.save(ckpt_1024, best_1024)

                if best_fitness_512 == fi_512:
                    torch.save(ckpt_512, best_512)   # 这个best是啥
                # if best_fitness_256 == fi_256:
                #     torch.save(ckpt_256, best_256)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt_1024, w_1024 / f'epoch{epoch}.pt')
                    torch.save(ckpt_512, w_512 / f'epoch{epoch}.pt')
                #    torch.save(ckpt_256, w_256 / f'epoch{epoch}.pt')
                del ckpt_1024,ckpt_512#,ckpt_256
                callbacks.run('on_model_save', last_1024, epoch, final_epoch, best_fitness_1024, fi_1024)
                callbacks.run('on_model_save', last_512,  epoch, final_epoch, best_fitness_512,  fi_512 )
                #callbacks.run('on_model_save', last_256,  epoch, final_epoch, best_fitness_256,  fi_256 )

#改到这里了！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！






        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

        for f_1024 in last_1024, best_1024:
            if f_1024.exists():
                strip_optimizer(f_1024)  # strip optimizers
                if f_1024 is best_1024:
                    LOGGER.info(f'\nValidating {f_1024}...')
                    results_1024, _, _ = val.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE ,
                        imgsz=imgsz,
                        model=attempt_load(f_1024, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                        single_cls=single_cls,
                        dataloader=val_loader_1024,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss_1024)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss_1024) + list(results_1024) + lr, epoch, best_fitness_1024, fi_1024)
        callbacks.run('on_train_end', last_1024, best_1024, plots, epoch, results_1024)

        for f_512 in last_512, best_512:
            if f_512.exists():
                strip_optimizer(f_512)  # strip optimizers
                if f_512 is best_512:
                    LOGGER.info(f'\nValidating {f_512}...')
                    results_512, _, _ = val.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE,
                        imgsz=imgsz_test,
                        model=attempt_load(f_512, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                        single_cls=single_cls,
                        dataloader=val_loader_512,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss_512)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss_512) + list(results_512) + lr, epoch, best_fitness_512, fi_512)
        callbacks.run('on_train_end', last_512, best_512, plots, epoch, results_512)

        # for f_256 in last_256, best_256:
        #     if f_256.exists():
        #         strip_optimizer(f_256)  # strip optimizers
        #         if f_256 is best_256:
        #             LOGGER.info(f'\nValidating {f_256}...')
        #             results_256, _, _ = val.run(
        #                 data_dict,
        #                 batch_size=batch_size // WORLD_SIZE,
        #                 imgsz=imgsz_test/2,
        #                 model=attempt_load(f_256, device).half(),
        #                 iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
        #                 single_cls=single_cls,
        #                 dataloader=val_loader_256,
        #                 save_dir=save_dir,
        #                 save_json=is_coco,
        #                 verbose=True,
        #                 plots=plots,
        #                 callbacks=callbacks,
        #                 compute_loss=compute_loss_256)  # val best model with plots
        #             if is_coco:
        #                 callbacks.run('on_fit_epoch_end', list(mloss_256) + list(results) + lr, epoch, best_fitness,fi_256)
        # callbacks.run('on_train_end', last_1024, best_1024, plots, epoch, results)



    torch.cuda.empty_cache()
    return results_1024,results_512#,results_256


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    #parser.add_argument('--cfg_256', type=str, default='models/DOTA/yolov5s_256.yaml', help='model.yaml path')
    # parser.add_argument('--cfg_512', type=str, default='models/DOTA/v5s_512_31.yaml', help='model.yaml path')
    # parser.add_argument('--cfg_1024', type=str, default='models/DOTA/v5s_1024_32_31.yaml', help='model.yaml path')#
    parser.add_argument('--cfg_low', type=str, default='models/DOTA/full/cfg_low.yaml', help='model.yaml path')
    parser.add_argument('--cfg_high', type=str, default='models/DOTA/full/cfg_high.yaml', help='model.yaml path')#
    # parser.add_argument('--cfg_512', type=str, default='models/NWPU/full/512_31_neck1_2_lesschannel_subpixl.yaml', help='model.yaml path')
    # parser.add_argument('--cfg_1024', type=str, default='models/NWPU/full/1024_32_31_neck1_2_lesschannel_subpixl.yaml', help='model.yaml path')#
    # parser.add_argument('--cfg', type=str, default='models/yolov5s_student.yaml', help='model.yaml path')#目前最好效果
    # parser.add_argument('--cfg_teacher', type=str, default='models/yolov5s_teacher_pro.yaml', help='model.yaml path')
    # parser.add_argument('--cfg', type=str, default='models/yolov5s_student_tqf.yaml', help='model.yaml path')
    # parser.add_argument('--cfg_teacher', type=str, default='models/yolov5s_teacher_tf.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')##
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=[1024,512], help='train, val image size (pixels)')
    #parser.add_argument('--factor', type=int, default=2)
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='W&B: Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')
    # parser.add_argument('--sr', action='store_true', help='super resolution')
    # parser.add_argument('--T', type=float, default=0.5, help='sr_T')
    parser.add_argument('--mask', action='store_true', help='是否用mask')
    parser.add_argument('--alpha', type=float, default=0.1, help='特征损失权重')
    #parser.add_argument('--gmma', type=int, default=1, help='super resolution weight')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()


    # Resume
    if opt.resume and not (check_wandb_resume(opt) or opt.evolve):  # resume from specified or most recent last.pt
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg_low, opt.cfg_high, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg_low), check_yaml(opt.cfg_high), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg_high) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.imgsz.extend([opt.imgsz[-1]] * (2 - len(opt.imgsz)))
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {evolve_csv}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
