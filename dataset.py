from os import listdir
from os.path import join
import random

from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img

import csv

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction, path_csv_file):
        super(DatasetFromFolder, self).__init__()
        self.path_csv_file = path_csv_file
        self.direction = direction
        self.a_path = join(image_dir, "TUfaceImages")
        self.b_path = join(image_dir, "rawscan_masked")
        self.image_filenames = []
        with open(self.path_csv_file, 'r') as f:
            reader = csv.reader(f)
            self.image_filenames = list(reader)
        # self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        finalsize = 512
        a = Image.open(join(self.a_path, self.image_filenames[index][0])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index][0])).convert('RGB')
        a = a.resize((finalsize+30, finalsize+30), Image.BICUBIC)
        b = b.resize((finalsize+30, finalsize+30), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, finalsize+30 - finalsize - 1))
        h_offset = random.randint(0, max(0, finalsize+30 - finalsize - 1))
    
        a = a[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]
        b = b[:, h_offset:h_offset + finalsize, w_offset:w_offset + finalsize]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        # if random.random() < 0.5:
        #     idx = [i for i in range(a.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     a = a.index_select(2, idx)
        #     b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)
