import torch
from unet import UNet
from PIL import Image
from torchvision import transforms
import numpy as np
import random
import os
from dataset import DataLoader
import cv2


class Inference():

    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = self.__load_network()
        self.batch_size = 4


    def __load_network(self):
        model = UNet(in_channels=3, out_channels=1)
        check = torch.load('saved_model/model.tar')
        model.load_state_dict(check['model_state_dict'])
        model = model.to(self.device)
        return model


    def __view_sample(self, x, y, y_pred):

        x = torch.squeeze(x, dim=0)
        y = torch.squeeze(y, dim=0)
        y = torch.cat((torch.cat((y, y), dim=0), y), dim=0)
        y_pred = torch.squeeze(y_pred, dim=0)
        y_pred = torch.cat((torch.cat((y_pred, y_pred), dim=0), y_pred), dim=0)
        tensor = torch.cat((x, torch.cat((y_pred, y), dim=2)), dim=2)
        tensor = tensor.cpu()
        image = tensor.detach().numpy()
        image = np.transpose(image, (1, 2, 0))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', image)
        cv2.waitKey(1)


    def infer(self):
        traindata, testdata = DataLoader().load_data(1)
        for i, (image, mask) in enumerate(testdata):
            image, mask = image.to(self.device), mask.to(self.device)
            model = self.network
            mask_pred = model(image)
            self.__view_sample(image, mask, mask_pred)
    

seg = Inference()
seg.infer()