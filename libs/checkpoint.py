
import os
from logging import getLogger
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = getLogger(__name__)


def save_checkpoint(
    result_path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    torch.save(save_states, os.path.join(result_path, "checkpoint.pth"))
    logger.debug("successfully saved the ckeckpoint.")

def save_checkpoint_BGShadowNet(
    result_path: str,
    epoch: int,
    firstStage_BGShadowNet: nn.Module,
    discriminator: nn.Module,
    optimizerG: optim.Optimizer,
    optimizerD: optim.Optimizer,
    best_g_loss: float,
    best_d_loss: float,
) -> None:

    save_states = {
        "epoch": epoch,
        "state_dictG": firstStage_BGShadowNet.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "best_g_loss": best_g_loss,
    }

    torch.save(save_states, os.path.join(result_path, "g_checkpoint.pth"))
    logger.debug("successfully saved the firstStage_BGShadowNet's ckeckpoint.")

    save_states = {
        "epoch": epoch,
        "state_dictD": discriminator.state_dict(),
        "optimizerD": optimizerG.state_dict(),
        "best_d_loss": best_d_loss,
    }

    torch.save(save_states, os.path.join(result_path, "d_checkpoint.pth"))
    logger.debug("successfully saved the discriminator's ckeckpoint.")
