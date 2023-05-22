# I am very grateful to the author of this code, which is used for reading datasets and other operations
# https://github.com/IsHYuhi/BEDSR-Net_A_Deep_Shadow_Removal_Network_from_a_Single_Document_Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import datetime
from libs.fix_weight_dict import fix_model_state_dict
import time
from logging import DEBUG, INFO, basicConfig, getLogger
import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import _LRScheduler
from albumentations import (
    Compose,
    RandomResizedCrop,
    HorizontalFlip,
    Normalize,
)
from albumentations.pytorch import ToTensorV2
from libs.models.CBENet import *
from libs.models.stageI import *
from libs.models.stageII import *
from libs.models.models import Discriminator
from libs.checkpoint import save_checkpoint_BGShadowNet
from libs.config import get_config
from libs.dataset import get_dataloader
from libs.device import get_device
from libs.helper_BGShadowNet import evaluate, train
from libs.logger import TrainLoggerBGShadowNet
from libs.loss_fn import get_criterion
from libs.seed import set_seed
logger = getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """parse all the arguments from command line inteface return a list of
    parsed arguments."""

    parser = argparse.ArgumentParser(
        description="""
        train a network for image classification with Flowers Recognition Dataset.
        """
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Add --use_wandb option if you want to use wandb.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Add --debug option if you want to see debug-level logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    return parser.parse_args()


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    # lr =lr/2**(epoch//100)
    if epoch > 200:
        lr = lr * (0.7 ** ((epoch - 150) // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def main() -> None:
    args = get_arguments()

    # save log files in the directory which contains config file.
    result_path = os.path.dirname(args.config)
    experiment_name = os.path.basename(result_path)

    # setting logger configuration
    logname = os.path.join(result_path, f"{datetime.datetime.now():%Y-%m-%d}_train.log")
    basicConfig(
        level=DEBUG if args.debug else INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=logname,
    )

    # fix seed
    set_seed()

    # configuration
    config = get_config(args.config)

    # cpu or cuda
    device = get_device(allow_only_gpu=False)

    # Dataloader
    train_transform = Compose(
        [
            RandomResizedCrop(config.height, config.width),
            HorizontalFlip(),
            Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ]
    )

    train_loader = get_dataloader(
        config.dataset_name,
        config.model,
        "train",
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        transform=train_transform,
    )

    # the number of classes
    n_classes = 1

    # define a model
    cbeNet = CBENet(3)  # 背景估计网络
    cbeNet_weights = torch.load('./pretrained/pretrained_CBENet.prm')
    cbeNet.load_state_dict(fix_model_state_dict(cbeNet_weights))
    firstStage_BGShadowNet = BGShadowNet1(3)  # 第一阶段网络
    secondStage_BGShadowNet = BGShadowNet2(6)  # 第二阶段网络
    discriminator = Discriminator(6)
    if config.pretrained == True:
        firstStage_BGShadowNet_weights = torch.load('./pretrained/pretrained_firstStage_for_BGShadowNet.prm')
        firstStage_BGShadowNet.load_state_dict(fix_model_state_dict(firstStage_BGShadowNet_weights))
        refine_weights = torch.load('./pretrained/pretrained_secondStage_for_BGShadowNet.prm')
        secondStage_BGShadowNet.load_state_dict(fix_model_state_dict(refine_weights))
        discriminator_weights = torch.load('./pretrained/pretrained_discriminator_for_BGShadowNet.prm')
        discriminator.load_state_dict(fix_model_state_dict(discriminator_weights))
    # send the model to cuda/cpu
    cbeNet.to(device)
    firstStage_BGShadowNet.to(device)
    discriminator.to(device)
    secondStage_BGShadowNet.to(device)

    optimizerG = optim.Adam(
        [{'params': firstStage_BGShadowNet.parameters()}, {'params': secondStage_BGShadowNet.parameters()}],
        lr=config.learning_rate, betas=(config.beta1, config.beta2))
    optimizerD = optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2))
    warmup_epoch = 4
    iter_per_epoch = 4371 // config.batch_size
    warmup_scheduler = WarmUpLR(optimizerG, iter_per_epoch * warmup_epoch)
    lambda_dict = {"lambda1": config.lambda1, "lambda2": config.lambda2}

    # keep training and validation log
    begin_epoch = 0
    best_g_loss = float("inf")
    best_d_loss = float("inf")

    log_path = os.path.join(result_path, "log.csv")
    train_logger = TrainLoggerBGShadowNet(log_path, resume=args.resume)

    # criterion for loss
    criterion = get_criterion(config.loss_function_name, device)

    # Weights and biases
    if args.use_wandb:
        wandb.init(
            name=experiment_name,
            config=config,
            project="BGShadowNet",
            job_type="training",
            # dirs="./wandb_result/",
        )
        # Magic
        # wandb.watch(model, log="all")
        wandb.watch(firstStage_BGShadowNet, log="all")
        wandb.watch(discriminator, log="all")

    # train and validate model
    logger.info("Start training.")

    for epoch in range(begin_epoch, config.max_epoch):
        # training

        start = time.time()
        train_g_loss, train_d_loss = train(
            train_loader, firstStage_BGShadowNet, secondStage_BGShadowNet, discriminator, cbeNet, warmup_scheduler,
            criterion, lambda_dict, optimizerG, optimizerD, epoch, device
        )
        train_time = int(time.time() - start)
        if epoch >= warmup_epoch:
            adjust_learning_rate(optimizerG, epoch, config.learning_rate)
        print('learn rate', optimizerG.param_groups[0]['lr'])

        # validation
        start = time.time()
        val_g_loss, val_d_loss = 1.0, 1.0
        val_time = int(time.time() - start)

        if epoch % 20 == 0 and epoch > 200:
            torch.save(discriminator.state_dict(),
                       os.path.join(result_path, "pretrained_discriminator_for_BGShadowNet" + str(epoch) + ".prm"), )
            torch.save(firstStage_BGShadowNet.state_dict(),
                       os.path.join(result_path, "pretrained_firstStage_for_BGShadowNet" + str(epoch) + ".prm"), )
            torch.save(
                secondStage_BGShadowNet.state_dict(),
                os.path.join(result_path, "pretrained_secondStage_for_BGShadowNet" + str(epoch) + ".prm"),
            )
        # save a model if top1 acc is higher than ever
        if best_g_loss > train_g_loss:
            best_g_loss = train_g_loss
            best_d_loss = train_d_loss
            torch.save(
                firstStage_BGShadowNet.state_dict(),
                os.path.join(result_path, "pretrained_firstStage_for_BGShadowNet.prm"),
            )
            torch.save(
                secondStage_BGShadowNet.state_dict(),
                os.path.join(result_path, "pretrained_secondStage_for_BGShadowNet.prm"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(result_path, "pretrained_discriminator_for_BGShadowNet.prm"),
            )

        # save checkpoint every epoch
        save_checkpoint_BGShadowNet(result_path, epoch, firstStage_BGShadowNet, discriminator, optimizerG, optimizerD,
                                    best_g_loss, best_d_loss)

        # write logs to dataframe and csv file
        train_logger.update(
            epoch,
            optimizerG.param_groups[0]["lr"],
            optimizerD.param_groups[0]["lr"],
            train_time,
            train_g_loss,
            train_d_loss,
            val_time,
            val_g_loss,
            val_d_loss
        )

        # save logs to wandb
        if args.use_wandb:
            wandb.log(
                {
                    "lrG": optimizerG.param_groups[0]["lr"],
                    "lrD": optimizerD.param_groups[0]["lr"],
                    "train_time[sec]": train_time,
                    "train_g_loss": train_g_loss,
                    "train_d_loss": train_d_loss,
                    "val_time[sec]": val_time,
                    "val_g_loss": val_g_loss,
                    "val_d_loss": val_d_loss,
                },
                step=epoch,
            )

    # save models
    torch.save(secondStage_BGShadowNet.state_dict(), os.path.join(result_path, "refine_checkpoint.prm"))
    torch.save(firstStage_BGShadowNet.state_dict(), os.path.join(result_path, "g_checkpoint.prm"))
    torch.save(discriminator.state_dict(), os.path.join(result_path, "d_checkpoint.prm"))

    # delete checkpoint
    os.remove(os.path.join(result_path, "g_checkpoint.pth"))
    os.remove(os.path.join(result_path, "d_checkpoint.pth"))

    logger.info("Done")


if __name__ == "__main__":
    main()
