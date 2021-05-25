from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img, imgtensor2numpy
import csv
import cv2
from glob import glob
import numpy as np
from PIL import Image

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=40, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint_0309/netG_model_epoch_{}.pth".format(opt.nepochs)

net_g = torch.load(model_path).to(device)

# if opt.direction == "a2b":
#     image_dir = "dataset/{}/test/a/".format(opt.dataset)
# else:
#     image_dir = "dataset/{}/test/b/".format(opt.dataset)



# ====================== 测试同一姿态的图片 没有子文件夹 ==========================================
# image_dir = "/home/zhang/zydDataset/faceRendererData/renderData/PIFu_front_render/0324/"
# gt_dir = "/home/zhang/zydDataset/faceRendererData/rawscan_masked/"
# save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/0_facerenderer-pix2pix/0324/"

image_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/glw2zyd/PRNet_data_7180/"
save_dir = "/run/user/1000/gvfs/afp-volume:host=CITE-3D.local,user=anonymous,volume=share/zhangyidi/FaceRendererData/testResults/0_facerenderer-pix2pix/0517/"

lis = sorted(glob(image_dir + "*_*/*.png"))
image_filenames = lis


transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for i in range(0, len(image_filenames)):  # len(image_filenames)
    image_name = image_filenames[i]
    name = image_name.split("/")[-1]
    subdir = image_name.split("/")[-2]

    img = load_img(image_name)

    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    # save_name = image_name.replace(image_dir, save_dir)
    # save_path = "/".join(save_name.split("/")[:-1])
    # save_path = save_dir + pose_ind + "/"
    save_path = save_dir
    save_name = save_path + subdir + "_" + name

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    o_im = imgtensor2numpy(out_img)

    i_im = np.array(Image.open(image_name))
    index = np.where(np.any(i_im < 255, axis=-1))
    o_im[index[0], index[1], :] = i_im[index[0], index[1], :]

    cv2.imwrite(save_name, o_im[:, :, ::-1])
    print(save_name)

    # save_img(out_img, "result/{}".format(image_name[0]))
# ==========================================================================================



# # ==================== 测试从10~50度的图片的代码 =================================================
# image_dir = "/home/zhang/zydDataset/faceRendererData/renderData/TUFaceImages_Pose/0325/"
# gt_dir = "/home/zhang/zydDataset/faceRendererData/rawscan_masked/"
# save_dir = "/home/zhang/zydDataset/faceRendererData/testResults/0_facerenderer-pix2pix/0325/"
#
# lis = sorted(glob(image_dir + "*/*_*/*.png"))
# image_filenames = lis
#
# transform_list = [transforms.ToTensor(),
#                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#
# transform = transforms.Compose(transform_list)
#
# for i in range(0, len(image_filenames)):  # len(image_filenames)
#     image_name = image_filenames[i]
#     name = image_name.split("/")[-1]
#     subdir = image_name.split("/")[-2]
#     pose_ind = image_name.split("/")[-3]
#     img = load_img(image_name)
#
#     img = transform(img)
#     input = img.unsqueeze(0).to(device)
#     out = net_g(input)
#     out_img = out.detach().squeeze(0).cpu()
#
#     save_path = save_dir + pose_ind + "/"
#     save_name = save_path + subdir + "_" + name
#
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#
#     o_im = imgtensor2numpy(out_img)
#
#     i_im = np.array(Image.open(image_name))
#     index = np.where(np.all(i_im < 255, axis=-1))
#     o_im[index[0], index[1], :] = i_im[index[0], index[1], :]
#
#     cv2.imwrite(save_name, o_im[:, :, ::-1])
#     print(save_name)
