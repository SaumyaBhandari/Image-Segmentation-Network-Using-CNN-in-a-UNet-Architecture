from torchvision import transforms
import torch
from PIL import Image
import os
import numpy as np
import cv2


class DataLoader():

    def __init__(self):
        self.transform =  transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


    def make_data(self):
        traindata, testdata = [], []

        for image, label  in zip(os.listdir('segmented-images/images'), os.listdir('segmented-images/masks')):
            traindata.append([f'segmented-images/images/{image}', f'segmented-images/masks/{label}'])

        for image, label in zip(os.listdir('segmented-images/images'), os.listdir('segmented-images/masks')):
            testdata.append([f'segmented-images/images/{image}', f'segmented-images/masks/{label}'])
        
        return traindata, testdata


    def load_data(self, batch_size):

        train_image, test_image = self.make_data()

        train_dataset = Dataset(train_image, self.transform)
        test_dataset = Dataset(test_image, self.transform)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        return train_dataloader, test_dataloader



class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, transforms=None):
        self.dataset = data
        self.transform = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, mask_path = self.dataset[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        mask = Image.open(mask_path).convert('L')
        mask = self.transform(mask)
        return image, mask


# class ValDataset(Dataset):
#     def __init__(self, data, transforms=None):
#         self.dataset = data
#         self.transform = transforms


# create = DataLoader()
# l1 = create.load_data(4)
# for items in l1:
#     for item in range(0, len(items), 2):
#         image= items[item][0]
#         mask = items[item+1][0]
#         image = np.concatenate((image, mask), axis=2)
#         img = np.moveaxis(image, 0, 2)
#         cv2.imshow('img', img)
#         cv2.waitKey(1)
