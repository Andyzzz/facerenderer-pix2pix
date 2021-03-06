from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img
import csv

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/netG_model_epoch_{}.pth".format(opt.nepochs)

net_g = torch.load(model_path).to(device)

# if opt.direction == "a2b":
#     image_dir = "dataset/{}/test/a/".format(opt.dataset)
# else:
#     image_dir = "dataset/{}/test/b/".format(opt.dataset)

image_dir = "/home/zhang/zydDataset/faceRendererData/TUfaceImages/"

test_csv_file = "./dataset/data_test.csv"
image_filenames = []
with open(test_csv_file, 'r') as f:
    reader = csv.reader(f)
    image_filenames = list(reader)

# image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    lis = image_name.split("/")
    dir = "/".join(lis[0: len(lis)-1])


    if not os.path.exists(os.path.join("result", dir)):
        os.makedirs(os.path.join("result", dir))
    save_img(out_img, "result/{}".format(image_name))
