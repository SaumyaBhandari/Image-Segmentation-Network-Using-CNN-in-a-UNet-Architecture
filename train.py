import torch
import torch.optim as optim
from loss import DiceLoss
from unet import UNet
from dataset import DataLoader
from tqdm import tqdm
import random
from torchvision import transforms

class Trainer():

    def __init__(self) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.network = self.load_network()
        self.batch_size = 4

    def load_network(self):
        model = UNet(in_channels=3, out_channels=1)
        # check = torch.load('saved_model/model.tar')
        # model.load_state_dict(check['model_state_dict'])
        model = model.to(self.device)
        return model
    
    def save_sample(self, epoch, x, y, y_pred):

        #just load the image from dataloader class, not like this stupid!
        x = x[0:1, :, :, :]
        y = y[0:1, :, :, :]
        y_pred = y_pred[0:1, :, :, :]
        x = torch.squeeze(x)
        y = torch.squeeze(y)
        y_pred = torch.squeeze(y_pred)
        x = transforms.ToPILImage()(x)
        y = transforms.ToPILImage()(y)
        y_pred = transforms.ToPILImage()(y_pred)

        image = x.save(f"sneakpeeks/{epoch}_input.jpg")
        actual = y.save(f"sneakpeeks/{epoch}_actual.jpg")
        pred = y_pred.save(f"sneakpeeks/{epoch}_predicted.jpg")


    def train(self, epochs, lr=0.0001):

        print("Loading Datasets...")
        self.train_dataloader = DataLoader().load_data(self.batch_size)
        print("Dataset Loaded... initializing parameters...")
        model = self.network
        optimizer = optim.AdamW(model.parameters(), lr)
        dsc_loss = DiceLoss()

        loss_train = []
        start = 0
        # epochs += 40
        print(f"Starting to train for {epochs} epochs.")
        for epoch in range(start, epochs):
            print(f"Epoch no: {epoch+1}")
            _loss = 0
            num = random.randint(0, 100)
            for i, (x, y) in enumerate(tqdm(self.train_dataloader)):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = dsc_loss(y_pred, y)
                _loss += loss.item()
                loss.backward()
                optimizer.step()
                if i == num:
                    self.save_sample(epoch+1, x, y, y_pred)

            loss_train.append(_loss)



            print(f"Epoch: {epoch+1}, Training loss: {_loss}")

            if loss_train[-1] == min(loss_train):
                print('Saving Model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_train
                }, 'saved_model/model.tar')
            print('\nProceeding to the next epoch...')
    


seg = Trainer()
seg.train(epochs=20)