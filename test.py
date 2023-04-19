"""
Same as audio_noise_reduction.ipynb, but can run from command line
"""

from torch.utils.data import Dataset, DataLoader
import torchaudio
import glob
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

# hyperparameters
class config:
    target_sample_rate=48000
    duration=4
    n_fft=1024
    hop_length=512
    n_mels=64
    batch_size=128
    learning_rate=1e-6
    epochs=50

# loading data
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, target_sample_rate=config.target_sample_rate, duration=config.duration):
        self.data_path = data_path
        self.transform = transform
        self.target_sample_rate = target_sample_rate
        self.num_samples = target_sample_rate*duration
            
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        try:
            audio_path_clean = self.data_path[index]
            
            signal, sr = torchaudio.load(audio_path_clean)
            if sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
                signal = resampler(signal)
            
            if signal.shape[0] > 1:
                signal = torch.mean(signal, axis=0, keepdim=True)
            
            if signal.shape[1] > self.num_samples:
                signal = signal[:, :self.num_samples]
            
            if signal.shape[1] < self.num_samples:
                num_missing_samples = self.num_samples - signal.shape[1]
                signal = F.pad(signal, (0, num_missing_samples))
            
            mel = self.transform(signal)
            if torch.Size([1, 64, 376]) != mel.shape:
                print(f"Mel Spec shape = {mel.shape}; Index = {index}")
            # image = mel / torch.abs(mel).max()
            return mel
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {str(e)}")
            print(audio_path_clean, signal, sr)
            raise e

def get_dataloaders():

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.target_sample_rate, n_fft=config.n_fft, 
        hop_length=config.hop_length, n_mels=config.n_mels
    )

    real_train_paths = glob.glob("/home/cameron/voice_data/for-norm/training/real/*.wav")
    real_val_paths = glob.glob("/home/cameron/voice_data/for-norm/validation/real/*.wav")

    train_dataset = CustomDataset(real_train_paths, mel_spectrogram)
    val_dataset = CustomDataset(real_val_paths, mel_spectrogram)

    trainloader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True)
    validloader = DataLoader(val_dataset, batch_size=config.batch_size, drop_last=True)
    return trainloader, validloader

# define U-Net
class UNetGenerator(nn.Module):
    def __init__(self, chnls_in=1, chnls_out=1, device="cuda"):
        super(UNetGenerator, self).__init__()
        self.down_conv_layer_1 = DownConvBlock(chnls_in, 64, norm=False, device=device).to(device)
        self.down_conv_layer_2 = DownConvBlock(64, 128, device=device).to(device)
        self.down_conv_layer_3 = DownConvBlock(128, 256, device=device).to(device)
        self.down_conv_layer_4 = DownConvBlock(256, 256, dropout=0.5, device=device).to(device)
        self.down_conv_layer_5 = DownConvBlock(256, 256, dropout=0.5, device=device).to(device)
        self.down_conv_layer_6 = DownConvBlock(256, 256, dropout=0.5, device=device).to(device)

        self.up_conv_layer_1 = UpConvBlock(256, 256, kernel_size=(2,3), stride=2, padding=0, dropout=0.5, device=device).to(device)# 256+256 6 5 kernel_size=(2, 3), stride=2, padding=0
        self.up_conv_layer_2 = UpConvBlock(512, 256, kernel_size=(2,3), stride=2, padding=0, dropout=0.5, device=device).to(device) # 256+256 1 4
        self.up_conv_layer_3 = UpConvBlock(512, 256, kernel_size=(2,3), stride=2, padding=0, dropout=0.5, device=device).to(device) # 2 3
        self.up_conv_layer_4 = UpConvBlock(512, 128, dropout=0.5, device=device).to(device) # 3 2
        self.up_conv_layer_5 = UpConvBlock(256, 64, device=device).to(device) # 4 1
        self.up_conv_layer_6 = UpConvBlock(512, 128, device=device).to(device)
        self.up_conv_layer_7 = UpConvBlock(256, 64, device=device).to(device)
        self.upsample_layer = nn.Upsample(scale_factor=2).to(device)
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0)).to(device)
        self.conv_layer_1 = nn.Conv2d(128, chnls_out, 4, padding=1).to(device)
        self.activation = nn.Tanh().to(device)
    
    def forward(self, x):
        #print('x', x.shape)
        enc1 = self.down_conv_layer_1(x) # [4, 64, 32, 188]
        # print('1', enc1.shape)
        enc2 = self.down_conv_layer_2(enc1) # [4, 128, 16, 94]
        # print('2', enc2.shape)
        enc3 = self.down_conv_layer_3(enc2) # [4, 256, 8, 47]
        # print('3', enc3.shape)
        enc4 = self.down_conv_layer_4(enc3) # [4, 256, 4, 23]
        # print('4', enc4.shape)
        enc5 = self.down_conv_layer_5(enc4) # [4, 256, 2, 11]
        # print('5', enc5.shape)
        enc6 = self.down_conv_layer_6(enc5) # [4, 256, 1, 5]
        #print('6', enc6.shape)
 
        dec1 = self.up_conv_layer_1(enc6, enc5)# enc6: 256 + enc5: 256 [4, 512, 2, 11]
        #print('d1', dec1.shape)
        dec2 = self.up_conv_layer_2(dec1, enc4)# enc4: 256 + dec1=enc5*2: [4, 512, 4, 23]
        #print('d2', dec2.shape)
        dec3 = self.up_conv_layer_3(dec2, enc3)# enc3: 256 + dec2=enc4*2: [4, 512, 8, 47]
        #print('d3', dec3.shape)
        dec4 = self.up_conv_layer_4(dec3, enc2)# enc2: 128 + dec3=enc3*2: [4, 256, 16, 94]
        #print('d4', dec4.shape)
        dec5 = self.up_conv_layer_5(dec4, enc1)# enc1: 64 + dec4=enc1*2: [4, 128, 32, 188]
        #print('d5', dec5.shape)
      
        final = self.upsample_layer(dec5)
        final = self.zero_pad(final)
        final = self.conv_layer_1(final)
        # print(f"Got to final, shape = {final.shape}")
        return final

class UpConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, kernel_size=4, stride= 2, padding=1 ,dropout=0.0, device="cuda"):
        super(UpConvBlock, self).__init__()
        self.layers = [
            nn.ConvTranspose2d(ip_sz, op_sz, kernel_size=kernel_size, stride=stride, padding=padding).to(device),
            nn.InstanceNorm2d(op_sz).to(device),
            nn.ReLU().to(device),
        ]
        if dropout:
            self.layers += [nn.Dropout(dropout).to(device)]
    def forward(self, x, enc_ip):
        x = nn.Sequential(*(self.layers))(x)
        #print('x', x.shape)
        #print('enc', enc_ip.shape)
        op = torch.cat((x, enc_ip), 1)
        return op


class DownConvBlock(nn.Module):
    def __init__(self, ip_sz, op_sz, kernel_size=4, norm=True, dropout=0.0, device="cuda"):
        super(DownConvBlock, self).__init__()
        self.layers = [nn.Conv2d(ip_sz, op_sz, kernel_size, 2, 1).to(device)]
        if norm:
            self.layers.append(nn.InstanceNorm2d(op_sz).to(device))
        self.layers += [nn.LeakyReLU(0.2).to(device)]
        if dropout:
            self.layers += [nn.Dropout(dropout).to(device)]
    def forward(self, x):
        op = nn.Sequential(*(self.layers))(x)
        return op

# train the model
def train(dataloader, model, epoch, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    # print(len(dataloader))
    for i, audio in enumerate(tqdm(dataloader)):
        audio = audio.to(device)
        
        optimizer.zero_grad()
        pred = model(audio)
        curr_loss = loss_fn(pred, audio)
        curr_loss.backward()
        optimizer.step()

        total_loss += curr_loss
        if i % 1000 == 0:
            print('[Epoch number : %d, Mini-batches: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, total_loss / 200))
            total_loss = 0.0
            
def val(dataloader, model, epoch, loss_fn, device):
    model.eval()
    total_loss = 0.0
    print('-------------------------')
    with torch.no_grad():
        for i, audio in enumerate(tqdm(dataloader)):
            audio = audio.to(device)
        
            output = model(audio)
            loss = loss_fn(output, audio)
            total_loss += loss
            if i % 100 == 0:
                print('[Valid Epoch number : %d, Mini-batches: %5d] loss: %.3f' %
                      (epoch + 1, i + 1, total_loss / 200))
                total_loss = 0.0

if __name__ == "__main__":

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    model = UNetGenerator(device=device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.functional.mse_loss

    trainloader, validloader = get_dataloaders()
    for epoch in range(config.epochs):
        train(trainloader, model, epoch, loss_fn, optimizer, device)
        val(validloader, model, epoch, loss_fn, device)
