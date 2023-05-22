#I am very grateful to the author of this code, which is used for reading datasets and other operations
#https://github.com/IsHYuhi/BEDSR-Net_A_Deep_Shadow_Removal_Network_from_a_Single_Document_Image
import cv2
import numpy as np
import os
from libs.models.models import Discriminator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from libs.fix_weight_dict import fix_model_state_dict
from libs.models.CBENet import *
from libs.models.stageI import *
from albumentations import (
    Compose,
    Normalize,
    Resize
)
from libs.models.stageII import *
from albumentations.pytorch import ToTensorV2
from utils.visualize import visualize, reverse_normalize
from libs.dataset import get_dataloader
from libs.loss_fn import get_criterion
from libs.helper_BGShadowNet import do_one_iteration
if __name__ == '__main__':

    def convert_show_image(tensor, idx=None):
        if tensor.shape[1]==3:
            img = reverse_normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif tensor.shape[1]==1:
            img = tensor*0.5+0.5

        if idx is not None:
            img = (img[idx].transpose(1, 2, 0)*255).astype(np.uint8)
        else:
            img = (img.squeeze(axis=0).transpose(1, 2, 0)*255).astype(np.uint8)

        return img

    test_transform = Compose([Resize(256,256), Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])
    test_loader = get_dataloader(
            "RDD",
            "BGShadowNet",
            "test",
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            transform=test_transform,
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cbeNet = CBENet(3)
    firstStage_BGShadowNet = BGShadowNet1(3)
    discriminator = Discriminator(6)
    cbeNet_weights = torch.load('./pretrained/pretrained_CBENet.prm')
    cbeNet.load_state_dict(fix_model_state_dict(cbeNet_weights))

    firstStage_BGShadowNet_weights = torch.load('./pretrained/pretrained_firstStage_for_BGShadowNet.prm')
    firstStage_BGShadowNet.load_state_dict(fix_model_state_dict(firstStage_BGShadowNet_weights))

    discriminator_weights = torch.load('./pretrained/pretrained_discriminator_for_BGShadowNet.prm')
    discriminator.load_state_dict(fix_model_state_dict(discriminator_weights))

    secondStage_BGShadowNet =BGShadowNet2(6)
    refine_weights = torch.load('./pretrained/pretrained_secondStage_for_BGShadowNet.prm')
    secondStage_BGShadowNet.load_state_dict(fix_model_state_dict(refine_weights))
    firstStage_BGShadowNet.eval()
    discriminator.eval()
    cbeNet.eval()
    secondStage_BGShadowNet.eval()
    cbeNet = cbeNet.to(device)
    firstStage_BGShadowNet.to(device)
    secondStage_BGShadowNet.to(device)
    discriminator.to(device)
    criterion = get_criterion("GAN", device)
    lambda_dict = {"lambda1": 1.0, "lambda2": 0.01}

    def check_dir():
        if not os.path.exists('./test_result'):
            os.mkdir('./test_result')
        if not os.path.exists('./test_result/img'):
            os.mkdir('./test_result/img')
        if not os.path.exists('./test_result/shadow_removal_image'):
            os.mkdir('./test_result/shadow_removal_image')
        if not os.path.exists('./test_result/grid'):
            os.mkdir('./test_result/grid')
        if not os.path.exists('./test_result/imtarget'):
            os.mkdir('./test_result/imtarget')

    check_dir()

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            _, _, _, gt, pred,coares_result= do_one_iteration(sample, firstStage_BGShadowNet, secondStage_BGShadowNet,discriminator, cbeNet,criterion, device, "evaluate", lambda_dict)
            img_path = sample['img_path'][0].split('/')[-1][:-4]+'.png'#
            shadow_removal = convert_show_image(np.array(pred))
            cv2.imwrite('./test_result/shadow_removal_image/' + img_path, shadow_removal)

