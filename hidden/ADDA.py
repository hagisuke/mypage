import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 1.データセットを用意する
training_epochs = 250            # エポック数
batch_size = 8                # バッチサイズ
n_class = 2
#class_weighting = torch.tensor(np.array([0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614], dtype="float32"))
img_size = (480, 640)
img_normalization = False
LRN = False
dropout = True

strain_img_path = "/dataset/hand_s_syn/train"
strain_gt_path = "/dataset/hand_s_syn/trainannot"
stest_img_path = "/dataset/hand_s_syn/test"
stest_gt_path = "/dataset/hand_s_syn/testannot"
ttrain_img_path = "/dataset/hand_t_real/train"
ttrain_gt_path = "/dataset/hand_t_real/trainannot"
ttest_img_path = "/dataset/hand_t_real/test"
ttest_gt_path = "/dataset/hand_t_real/testannot"

# dataloader
class MakeDataset(Dataset):
    def __init__(self, img_path, gt_path, transform = None, src = True):
        self.transform = transform

        imgs_names = os.listdir(img_path)
        imgs_names.sort()
        imgs_names = np.asarray(imgs_names)
        if src: # make number of data 1/6
            imgs_names = np.random.choice(imgs_names, int(len(imgs_names)/6))

        gt_names = os.listdir(gt_path)
        gt_names.sort()
        gt_names = np.asarray(gt_names)

        self.datanum = len(imgs_names)

        imgs = np.empty((len(imgs_names), img_size[0], img_size[1], 3), dtype="float32")
        for i in range(len(imgs_names)):
            img = cv2.imread(os.path.join(img_path, imgs_names[i]))
            img = cv2.resize(img, (img_size[1],img_size[0]))
            img = img.astype("float32")
            imgs[i] = img
        
        gts = np.empty((len(imgs_names), img_size[0], img_size[1]), dtype="int32")
        for i in range(len(imgs_names)):
            gt = cv2.imread(os.path.join(gt_path, imgs_names[i]))
            gt = cv2.resize(gt, (img_size[1],img_size[0]))
            gt = gt.astype("int32")
            gt = gt[:,:,0] #gt[:,:,0] == gt[:,:,1] == gt[:,:,2]
            gt = gt.reshape(img_size[0], img_size[1]) #which is better it or flatten()?
            gts[i] = gt

        self.data = imgs
        self.label = gts

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label

strainset = MakeDataset(strain_img_path, strain_gt_path, transform=transforms.ToTensor(), src=False)
strainloader = torch.utils.data.DataLoader(strainset, batch_size=batch_size, shuffle=True, drop_last=True)
stestset = MakeDataset(stest_img_path, stest_gt_path, transform=transforms.ToTensor(), src=False)
stestloader = torch.utils.data.DataLoader(stestset, batch_size=1, shuffle=False, drop_last=False)
ttrainset = MakeDataset(ttrain_img_path, ttrain_gt_path, transform=transforms.ToTensor(), src=False)
ttrainloader = torch.utils.data.DataLoader(ttrainset, batch_size=batch_size, shuffle=True, drop_last=True)
ttestset = MakeDataset(ttest_img_path, ttest_gt_path, transform=transforms.ToTensor(), src=False)
ttestloader = torch.utils.data.DataLoader(ttestset, batch_size=1, shuffle=False, drop_last=False)
print("dataset prepared")

# image normalize
def imagenormalize(imgs):
    imgs = imgs.clone().numpy()
    num_imgs = len(imgs)
    imgs = imgs.astype("uint8")
    normalized_imgs = np.zeros((num_imgs, 3, img_size[0], img_size[1]), dtype="float32")
    for i in range(num_imgs):
        normalized_imgs[i,0,:,:] = cv2.equalizeHist(imgs[i,0,:,:])
        normalized_imgs[i,1,:,:] = cv2.equalizeHist(imgs[i,1,:,:])
        normalized_imgs[i,2,:,:] = cv2.equalizeHist(imgs[i,2,:,:])

    return torch.tensor(normalized_imgs)

# 2.モデルの作成
class FeatureExtractor(nn.Module):
    def __init__(self, n_class=n_class):
        super(FeatureExtractor, self).__init__()   
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3,  64, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64,  128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128,  128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128,  128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128,  128, kernel_size=1, stride=1, padding=0)

    def forward(self, x): #x = (batchsize, 3, 480, 640) 
        x = x/255.0
        h = F.relu(self.conv1(x))
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = self.conv5(h)
        return h

class Classifier(nn.Module):
    def __init__(self, n_class=n_class):
        super(Classifier, self).__init__()
        self.deconv6 = nn.ConvTranspose2d(128, n_class, 32, stride=16, padding=8, bias=False)
        
    def forward(self, h):
        h = self.deconv6(h) 
        return h
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=(14,24), stride=16, padding=0),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64, 2),
        )
    
    def forward(self, h):
        y = self.layer(h)
        return y

# 3.実行する
# GPUに対応させる
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Fmodel = FeatureExtractor().to(device)
Cmodel = Classifier().to(device)

# 誤差逆伝播法アルゴリズムを選択する
CE = nn.CrossEntropyLoss() # 損失関数を選択
Pre_opt = optim.Adam(list(Fmodel.parameters()) + list(Cmodel.parameters()), lr=lr)]

# STEP1: train sourceCNN and classifier
for epoch in range(training_epochs):
    running_closs = 0.0
    for i, (src_inputs, labels) in enumerate(strainloader):
        src_inputs, labels = src_inputs.to(device), labels.to(device)

        Pre_opt.zero_grad()
        h = Fmodel(src_inputs)
        c = Cmodel(h)
        Lc = CE(c, labels.long())
        Lc.backward()
        Pre_opt.step()

        # 損失を定期的に出力する
        running_closs += Lc.item()
        if i % 20 == 19:
            print('[{:d}, {:5d}] loss: {:3f}'
                    .format(epoch + 1, i + 1, running_closs / 20))
            running_closs = 0.0

print("Finished STEP1")

correct = 0
total = 0

with torch.no_grad():
    for (images, labels) in ttestloader:
        images, labels = images.to(device), labels.to(device)
        h = Fmodel(images)
        outputs = Cmodel(h)
        outputs = outputs.data.reshape((n_class, img_size[0], img_size[1]))
        _, predicted = torch.max(outputs, 0)
        total += labels.shape[1]*labels.shape[2]
        correct += (predicted == labels).sum().item()
print('Accuracy(Target): {:.2f} %%'.format(100 * float(correct/total)))

# STEP2: train targetCNN and discriminator
Tmodel = FeatureExtractor().to(device)
Tmodel.load_state_dict(Fmodel.state_dict())
#Tmodel.train()
#for param in Fmodel.parameters():
#    param.requires_grad = False
Fmodel.eval()
Dmodel = Discriminator().to(device)
T_opt = optim.RMSprop(Tmodel.parameters(), lr=lr, weight_decay=weight_decay)
D_opt = optim.RMSprop(Dmodel.parameters(), lr=lr, weight_decay=weight_decay)

step = 0
n_batches = len(ttrainset)//batch_size
target_set = iter(ttrainloader)

def sample_target(step, n_batches):
    global target_set
    if step % n_batches == 0:
        target_set = iter(ttrainloader)
    return target_set.next()

for epoch in range(training_epochs):
    running_gloss = 0.0
    running_dloss = 0.0
    for i, (src_inputs, labels) in enumerate(strainloader):
        tgt_inputs, _ = sample_target(step, n_batches)
        src_inputs, labels, tgt_inputs = src_inputs.to(device), labels.to(device), tgt_inputs.to(device)

        s = Fmodel(src_inputs)
        t = Tmodel(tgt_inputs)
        ys = Dmodel(s.detach())
        yt = Dmodel(t.detach())
        Ld = - torch.sum(F.log_softmax(ys,dim=1)[:, 0])/batch_size - torch.sum(F.log_softmax(yt,dim=1)[:, 1])/batch_size
        D_opt.zero_grad()
        Ld.backward()
        D_opt.step()

        tg = Tmodel(tgt_inputs)
        ytg = Dmodel(tg)
        Lg = -torch.sum(F.log_softmax(ytg,dim=1)[:, 0])/batch_size
        T_opt.zero_grad()
        Lg.backward()
        T_opt.step()

        # 損失を定期的に出力する
        running_gloss += Lg.item()
        running_dloss += Ld.item()
        if i % 20 == 19:
            print('[{:d}, {:5d}] loss: {:3f}, {:3f}'
                    .format(epoch + 1, i + 1, running_gloss / 20, running_dloss / 20))
            running_gloss = 0.0
            running_dloss = 0.0

        if i % 500 == 0:
            correct = 0
            total = 0

            with torch.no_grad():
                for (images, labels) in ttestloader:
                    images, labels = images.to(device), labels.to(device)
                    h = Tmodel(images)
                    outputs = Cmodel(h)
                    outputs = outputs.data.reshape((n_class, img_size[0], img_size[1]))
                    _, predicted = torch.max(outputs, 0)
                    total += labels.shape[1]*labels.shape[2]
                    correct += (predicted == labels).sum().item()
            print('Accuracy(Target): {:.2f} %%'.format(100 * float(correct/total)))

        step += 1

print("Finished STEP2")

torch.save(Fmodel.to('cpu').state_dict(), 'Fmodel.pth')
torch.save(Tmodel.to('cpu').state_dict(), 'Tmodel.pth')
torch.save(Cmodel.to('cpu').state_dict(), 'Cmodel.pth')
torch.save(Dmodel.to('cpu').state_dict(), 'Dmodel.pth')
Fmodel.eval()
Tmodel.eval()
Cmodel.eval()
Dmodel.eval()
dummy_input1 = torch.randn(1, 3, 480, 640)
torch.onnx.export(Tmodel.to('cpu'), dummy_input1.to('cpu'), "Tmodel.onnx", export_params=True, verbose=True, input_names=["input"], output_names=["mid"])
dummy_input2 = torch.randn(1, 128, 30, 40)
torch.onnx.export(Cmodel.to('cpu'), dummy_input2.to('cpu'), "Cmodel.onnx", export_params=True, verbose=True, input_names=["mid"], output_names=["prediction"])
print('Finished Training')

Fmodel = Fmodel.to(device)
Tmodel = Tmodel.to(device)
Cmodel = Cmodel.to(device)
Dmodel = Dmodel.to(device)

# テストする
palette = [
    [255,0,0],
    [0,0,255]]

def draw_png(img):
    seg_img = np.zeros((img_size[0], img_size[1], 3))
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            color_number = img[i,j]
            seg_img[i,j,0] = palette[color_number][0]
            seg_img[i,j,1] = palette[color_number][1]
            seg_img[i,j,2] = palette[color_number][2]
    seg_img = Image.fromarray(np.uint8(seg_img))
    return seg_img

i = 0
correct = 0
total = 0
iou = 0.0
with torch.no_grad():
    for (images, labels) in stestloader:
        images, labels = images.to(device), labels.to(device)
        h = Fmodel(images)
        outputs = Cmodel(h)
        outputs = outputs.data.reshape((n_class, img_size[0], img_size[1]))
        _, predicted = torch.max(outputs, 0)
        total += labels.shape[1]*labels.shape[2]
        correct += (predicted == labels).sum().item()
        intersection = ((1-predicted) & (1-labels)).float().sum((1, 2))
        union = ((1-predicted) | (1-labels)).float().sum((1, 2))
        if union==0.0:
          continue
        iou += float(intersection/union)
        i = i + 1
        #colored_seg_img = draw_png(predicted)

print('Source-> pixAcc: {:.2f} %% mIoU: {:.2f} %%'.format(100 * float(correct/total), 100 * float(iou/i)))

i = 0
correct = 0
total = 0
iou = 0.0
colorlist = []
with torch.no_grad():
    for (images, labels) in ttestloader:
        images, labels = images.to(device), labels.to(device)
        h = Tmodel(images)
        outputs = Cmodel(h)
        outputs = outputs.data.reshape((n_class, img_size[0], img_size[1]))
        _, predicted = torch.max(outputs, 0)
        total += labels.shape[1]*labels.shape[2]
        correct += (predicted == labels).sum().item()
        intersection = ((1-predicted) & (1-labels)).float().sum((1, 2))
        union = ((1-predicted) | (1-labels)).float().sum((1, 2))
        if union==0.0:
          continue
        iou += float(intersection/union)
        i = i + 1
        colored_seg_img = draw_png(predicted)
        colorlist.append(colored_seg_img)

print('Target-> pixAcc: {:.2f} %% mIoU: {:.2f} %%'.format(100 * float(correct/total), 100 * float(iou/i)))
#colorlist[0]

# for t-SNE
rslnp = []
with torch.no_grad():
    for (images, labels) in stestloader:
        images, labels = images.to(device), labels.to(device)
        h = Fmodel(images)
        outputs = Cmodel(h)
        outputs = outputs.data.reshape((n_class*img_size[0]*img_size[1]))
        rslnp.append(outputs.to('cpu').detach().numpy().copy())

rslnp = np.array(rslnp)
rslnp = rslnp.reshape([len(rslnp),2,img_size[0],img_size[1]])
rslnp = rslnp.transpose(0,2,3,1)
rslnp = rslnp.reshape([len(rslnp)*img_size[0]*img_size[1],2])
print(rslnp.shape)
np.save("ADDA_s", rslnp)

rslnp = []
with torch.no_grad():
    for (images, labels) in ttestloader:
        images, labels = images.to(device), labels.to(device)
        h = Tmodel(images)
        outputs = Cmodel(h)
        outputs = outputs.data.reshape((n_class*img_size[0]*img_size[1]))
        rslnp.append(outputs.to('cpu').detach().numpy().copy())

rslnp = np.array(rslnp)
rslnp = rslnp.reshape([len(rslnp),2,img_size[0],img_size[1]])
rslnp = rslnp.transpose(0,2,3,1)
rslnp = rslnp.reshape([len(rslnp)*img_size[0]*img_size[1],2])
print(rslnp.shape)
np.save("ADDA_t", rslnp)
