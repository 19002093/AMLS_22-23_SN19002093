#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[7]:


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Load the dataset

    # In[8]:


    train_data = []
    train_label = []
    test_data = []
    test_label = []

    # transform images to tensor
    custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                           transforms.Resize((64, 64)),
                                           transforms.ToTensor()])

    # load the train data
    train_data_path = "C:/Users/CJyM2/Desktop/AMLS_22-23_SN19002093/Datasets/celeba/img/"
    train_label_path = "C:/Users/CJyM2/Desktop/AMLS_22-23_SN19002093/Datasets/celeba/labels.csv"
    train_name2smile = {}
    with open(train_label_path, "r") as read_file:
        lines = read_file.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            values = line.strip().split("\t")
            name = values[1]
            smile = 1 if values[3] == "1" else 0# smiling = 1, non-smiling = 0
            train_name2smile[name] = smile
    images = os.listdir(train_data_path)
    for file in images:
        path = os.path.join(train_data_path, file)
        img = custom_transform(Image.open(path))
        label = train_name2smile[file]
        train_data.append(img)
        train_label.append(label)

    # load the test data
    test_data_path = "C:/Users/CJyM2/Desktop/AMLS_22-23_SN19002093/Datasets/celeba_test/img/"
    test_label_path = "C:/Users/CJyM2/Desktop/AMLS_22-23_SN19002093/Datasets/celeba_test/labels.csv"
    test_name2smile = {}
    with open(test_label_path, "r") as read_file:
        lines = read_file.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            values = line.strip().split("\t")
            name = values[1]
            smile = 1 if values[3] == "1" else 0
            test_name2smile[name] = smile
    images = os.listdir(test_data_path)
    for file in images:
        path = os.path.join(test_data_path, file)
        img = custom_transform(Image.open(path))
        label = test_name2smile[file]
        test_data.append(img)
        test_label.append(label)


    # Define the Dataset

    # In[9]:


    class CelebaDataset(Dataset):
        #Custom Dataset for loading CelebA face images

        def __init__(self, x, y):
            self.x = x
            self.y = y
        def __getitem__(self, index):
            image = self.x[index]
            label = self.y[index]
            return image, torch.tensor(label)

        def __len__(self):
            return len(self.y)

    train_dataset = CelebaDataset(train_data, train_label)
    test_dataset = CelebaDataset(test_data, test_label)


    # Define the model

    # In[10]:


    def accuracy_score(truths, predictions):
        num = len(truths)
        correct_num = 0
        for i in range(num):
            if truths[i] == predictions[i]:
                correct_num += 1
        return correct_num / num


    # In[11]:


    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.mlp = torch.nn.Sequential(torch.nn.Linear(3* 64 * 64, 2048),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(2048),
                torch.nn.Linear(2048, 768))
            hidden_size = 768
            vocab_size = 2  # num classes
            self.c_layer = torch.nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            # Forward propagation

            embedding = self.mlp(x)
            logits = self.c_layer(embedding)

            y_hat = logits.argmax(-1)

            return logits, y_hat


    # Train and test

    # In[12]:


    # create the model
    model = Net()

    # hyper-parameters
    batch_size = 128
    epoch = 50
    lr = 1e-5

    #allocate storage space for the accuracy and loss for each epoch
    trains = []
    tests = []
    losses = []

    # optimzer and loss function
    optim = torch.optim.Adam(model.parameters(), lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # set seed
    seed = 2023
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    # train
    for i in range(epoch):
        print("Epoch: {}".format(i))
        loss_list = []
        model.train()
        train_predictions = []
        train_truths = []
        for sample in train_dataloader:
            x, y = sample
            train_truths.extend(y.tolist())
            x = x.view(x.shape[0], -1)
            logit, y_hat = model(x)
            y_hat = y_hat.view(-1).tolist()
            train_predictions.extend(y_hat)
            loss = loss_func(logit, y)
            loss_list.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        mean_loss = np.mean(loss_list)
        print("Epoch: {}  Loss: {}".format(i, mean_loss))
        losses.append(mean_loss)

        # calculate accuracy on the train data
        accuracy = accuracy_score(train_truths, train_predictions)
        trains.append(accuracy)
        print("Epoch: {}  Train Accuracy: {}".format(i, accuracy))

        # test
        model.eval()
        predictions = []
        truths = []
        with torch.no_grad():
            for sample in test_dataloader:
                x, y = sample
                truths.extend(y.tolist())
                x = x.view(x.shape[0], -1)
                _, prediction = model(x)
                prediction = prediction.view(-1)
                predictions.extend(prediction.tolist())

        # calculate accuracy on the test data
        accuracy = accuracy_score(truths, predictions)
        tests.append(accuracy)
        print("Epoch: {}  Test Accuracy: {}".format(i, accuracy))


    #plot training accuracy and testing accuracy for each epoch on the same graph
    No=range(1,epoch+1)# from 1 to 50
    fig=plt.figure
    ax1=plt.subplot(2,1,1)# first graph
    plt.plot(No,trains,label='training accuracy',color='g')# plot a green line for train accuracy
    plt.plot(No,tests,label='testing accuracy',color='b')# plot a blue line for train accuracy
    #put text
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('training and testing accuracy vs epoch with learning rate = {}'.format(lr))
    plt.legend()
    #show the plot
    #plt.savefig("A1_accuracies.jpg")
    #plt.show()


    # plot mean loss vs epoch 
    ax2=plt.subplot(2,1,2)#second graph
    plt.plot(No, losses,label='loss',color='r')# plot a red line for loss
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss vs epoch with learning rate = {}'.format(lr))
    plt.legend()
    #show the plot
    plt.tight_layout()#Adjust the padding between and around subplots to avoid overlapping
    plt.savefig("A2_plots.jpg")
    plt.show()


    # 
