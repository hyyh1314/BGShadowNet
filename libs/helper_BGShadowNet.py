import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from logging import getLogger
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.lib.shape_base import apply_along_axis
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .VGG_loss import VGGNet
from .meter import AverageMeter, ProgressMeter
from .metric import calc_accuracy

__all__ = ["train", "evaluate"]

logger = getLogger(__name__)

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
vgg = VGGNet().cuda().eval()
def perceptual_loss(x, y):
    c = nn.MSELoss()
    fx1, fx2 = vgg(x)
    fy1, fy2 = vgg(y)
    m1 = c(fx1, fy1)
    m2 = c(fx2, fy2)
    loss = (m1+m2)*0.06
    return loss

def do_one_iteration(
    sample: Dict[str, Any],
    firstStage_BGShadowNet: nn.Module,
    secondStage_BGShadowNet: nn.Module,
    discriminator: nn.Module,
    cbeNet:nn.Module,
    criterion: Any,
    device: str,
    iter_type: str,
    lambda_dict: Dict,
    optimizerG: Optional[optim.Optimizer] = None,
    optimizerD: Optional[optim.Optimizer] = None,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:

    if iter_type not in ["train", "evaluate"]:
        message = "iter_type must be either 'train' or 'evaluate'."
        logger.error(message)
        raise ValueError(message)

    if iter_type == "train" and (optimizerG is None or optimizerD is None):
        message = "optimizer must be set during training."
        logger.error(message)
        raise ValueError(message)

    Tensor = torch.cuda.FloatTensor if device != torch.device("cpu") else torch.FloatTensor

    x = sample["img"].to(device)
    gt = sample["gt"].to(device)
    background,featureMap = cbeNet(x.to(device))
    background = background.detach()

    batch_size, c, h, w = x.shape

    # compute output and loss
    # train discriminator
    if iter_type == "train" and optimizerD is not None:
        set_requires_grad([discriminator], True)
        optimizerD.zero_grad()

    
    confuse_result,confuseFeatureMap = firstStage_BGShadowNet(x.to(device),featureMap)#
    confuse_result = confuse_result
    # refine_input = torch.cat([confuse_result,background],dim=1)
    shadow_removal_image ,_= secondStage_BGShadowNet(confuse_result,background,x,confuseFeatureMap)
    fake = torch.cat([x, shadow_removal_image], dim=1)
    real = torch.cat([x, gt], dim=1)

    out_D_fake = discriminator(fake.detach())
    out_D_real = discriminator(real.detach())

    label_D_fake = Variable(Tensor(np.zeros(out_D_fake.size())), requires_grad=True)
    label_D_real = Variable(Tensor(np.ones(out_D_fake.size())), requires_grad=True)

    loss_D_fake = criterion[1](out_D_fake, label_D_fake)
    loss_D_real = criterion[1](out_D_real, label_D_real)

    D_L_GAN = loss_D_fake + loss_D_real

    D_loss = lambda_dict["lambda2"] * D_L_GAN

    if iter_type == "train" and optimizerD is not None:
        D_loss.backward()
        optimizerD.step()

    # train firstStage_BGShadowNet
    if iter_type == "train" and optimizerD is not None:
        set_requires_grad([discriminator], False)
        optimizerG.zero_grad()

    fake = torch.cat([x, shadow_removal_image], dim=1)
    out_D_fake = discriminator(fake.detach())

    G_L_GAN = criterion[1](out_D_fake, label_D_real)
    G_L_data = criterion[0](gt, shadow_removal_image)
    G_L_confuse = criterion[0](gt,confuse_result)
    G_L_VGG = perceptual_loss(gt, shadow_removal_image)


    G_loss = lambda_dict["lambda1"] * G_L_data + lambda_dict["lambda2"] * G_L_GAN+G_L_VGG+0.2*G_L_confuse#粗网络的loss真的有意义吗?

    if iter_type == "train" and optimizerG is not None:
        G_loss.backward()
        optimizerG.step()

    # measure PSNR and SSIM TODO
    gt = gt.to("cpu").numpy()
    pred = shadow_removal_image.detach().to("cpu").numpy()
    background = background.detach().to("cpu").numpy()
    confuse_result = confuse_result.detach().to("cpu").numpy()


    return batch_size, G_loss.item(), D_loss.item(), gt, pred,confuse_result


def train(
    loader: DataLoader,
    firstStage_BGShadowNet: nn.Module,
    secondStage_BGShadowNet: nn.Module,
    discriminator: nn.Module,
    cbeNet:nn.Module,
    warmup_scheduler,
    criterion: Any,
    lambda_dict: Dict,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
    epoch: int,
    device: str,
    interval_of_progress: int = 50,
) -> Tuple[float, float, float]:

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    g_losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter("Loss", ":.4e")
    #top1 = AverageMeter("Acc@1", ":6.2f")

    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, g_losses, d_losses],#, top1
        prefix="Epoch: [{}]".format(epoch),
    )

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to train mode
    firstStage_BGShadowNet.train()
    discriminator.train()
    secondStage_BGShadowNet.train()
    cbeNet.eval()

    end = time.time()
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size, g_loss, d_loss, gt, pred ,_= do_one_iteration(
            sample, firstStage_BGShadowNet,secondStage_BGShadowNet, discriminator,cbeNet, criterion, device, "train", lambda_dict, optimizerG, optimizerD
        )
        

        g_losses.update(g_loss, batch_size)
        d_losses.update(d_loss, batch_size)
        if epoch<4:
            warmup_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # show progress bar per 50 iteration
        if i != 0 and i % interval_of_progress == 0:
            progress.display(i)

    return g_losses.get_average(), d_losses.get_average()


def evaluate(
    loader: DataLoader, firstStage_BGShadowNet: nn.Module, discriminator: nn.Module, cbeNet:nn.Module,criterion: Any, lambda_dict: Dict, device: str
) -> Tuple[float, float]:
    g_losses = AverageMeter("Loss", ":.4e")
    d_losses = AverageMeter("Loss", ":.4e")

    # keep predicted results and gts for calculate F1 Score
    gts = []
    preds = []

    # switch to evaluate mode
    firstStage_BGShadowNet.eval()
    discriminator.eval()

    with torch.no_grad():
        for sample in loader:
            batch_size, g_loss, d_loss, gt, pred = do_one_iteration(
                sample, firstStage_BGShadowNet, discriminator, cbeNet, criterion, device, "evaluate", lambda_dict
            )

            g_losses.update(g_loss, batch_size)
            d_losses.update(d_loss, batch_size)

            # save the ground truths and predictions in lists
            #gts += list(gt)
            #preds += list(pred)

    return g_losses.get_average(), d_losses.get_average()