#!/usr/bin/env python
# coding: utf-8

# # Import necessary packages

# In[1]:


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
    # # Load the dataset

    # In[2]:


    train_data = []
    train_label = []
    test_data = []
    test_label = []

    # transform images to tensor
    custom_transform = transforms.Compose([## focused on the centre of the image, and cut it into 178x178
                                           transforms.CenterCrop((178, 178)),
                                           ## scale the image to be h=64, w=64, but not actually cut
                                           transforms.Resize((64, 64)),
                                           ##transform an image of (H, W, C) to be a tensor of (C, H, W)
                                           transforms.ToTensor()])

    # load the train data
    train_data_path = "C:/Users/CJyM2/Desktop/AMLS_22-23_SN19002093/Datasets/celeba/img/"
    train_label_path = "C:/Users/CJyM2/Desktop/AMLS_22-23_SN19002093/Datasets/celeba/labels.csv"
    train_name2gender = {}
    with open(train_label_path, "r") as read_file:
        lines = read_file.readlines()# read the entire line from the file
        for i in range(1, len(lines)):# run len(lines) times
            line = lines[i]# get one piece
            # strip: delete spaces at the head and the tail. 
            # split: split string "line" into substrings if it meets "\t"
            values = line.strip().split("\t")
            # name = the serial number of images
            name = values[1]
            # third column (index=2) represents gender, with male=1, female=0
            gender = 1 if values[2] == "1" else 0 
            # store the value of gender to train_name2gender at index=name
            train_name2gender[name] = gender

    # load a lift of names of images in "../Datasets/celeba/img/"
    images = os.listdir(train_data_path)

    for file in images:# variable "file", from 0 to 4999 in A1
        # the exact path is the combination of train_data_path and the value of "file"
        path = os.path.join(train_data_path, file)
        # change the size of images to be 64x64, and assign this value to "img"
        img = custom_transform(Image.open(path))
        # assign the value of train_name2gender at index=file to "label", in this case, label=1 or 0
        label = train_name2gender[file]
        # add "img" to the end of "train_data"
        train_data.append(img)
        # add "label" to the end of "train_label"
        train_label.append(label)

    # load the test data
    test_data_path = "C:/Users/CJyM2/Desktop/AMLS_22-23_SN19002093/Datasets/celeba_test/img/"
    test_label_path = "C:/Users/CJyM2/Desktop/AMLS_22-23_SN19002093/Datasets/celeba_test/labels.csv"
    test_name2gender = {}
    with open(test_label_path, "r") as read_file:
        lines = read_file.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            values = line.strip().split("\t")
            name = values[1]
            gender = 1 if values[2] == "1" else 0
            test_name2gender[name] = gender
    images = os.listdir(test_data_path)
    for file in images:
        path = os.path.join(test_data_path, file)
        img = custom_transform(Image.open(path))
        label = test_name2gender[file]
        test_data.append(img)
        test_label.append(label)





    # # Define the Dataset

    # In[3]:


    class CelebaDataset(Dataset):
        #Custom Dataset for loading Celeba  images

        def __init__(self, x, y):## allocate storage space and initialize
            self.x = x
            self.y = y
        def __getitem__(self, index):## return the image data and label data with the given index of input 
            image = self.x[index]
            label = self.y[index]
            return image, torch.tensor(label)

        def __len__(self):## return the length of label of input
            return len(self.y)

    train_dataset = CelebaDataset(train_data, train_label)##
    test_dataset = CelebaDataset(test_data, test_label)


    # # Define the network

    # In[4]:


    class Net(torch.nn.Module):#inherit from torch.nn.Module
        def __init__(self):
            super().__init__()

            self.mlp = torch.nn.Sequential(
                #input layer
                torch.nn.Linear(3* 64 * 64, 2048),
                #hidden layer
                torch.nn.ReLU(),#rectified linear unit function
                torch.nn.BatchNorm1d(2048),#normalization
                torch.nn.Linear(2048, 768))# a linear layer
            #output layer, two outputs
            hidden_size = 768
            vocab_size = 2  # num classes 
            #nn.Linear () initialize random values for weights and bias output
            self.c_layer = torch.nn.Linear(hidden_size, vocab_size)

        #forward propagation
        def forward(self, x):
            # Forward propagation

            embedding = self.mlp(x)
            logits = self.c_layer(embedding)
            # -1 means adapt automatically
            y_hat = logits.argmax(-1)

            return logits, y_hat


    # # Train and test

    # accuracy

    # In[5]:


    def accuracy_score(truths, predictions):
        num = len(truths)
        correct_num = 0

        #get the number of correct results compared to the label listed in input "truths"
        for i in range(num):
            if truths[i] == predictions[i]:
                correct_num += 1

        return correct_num / num # ratio of correct/overall


    # In[6]:


    # create an instance of this neural network
    model = Net()

    # hyper-parameters
    batch_size = 128
    epoch = 50
    lr = 1e-5 #learning rate

    #allocate storage space for the accuracy and loss for each epoch
    trains = []
    tests = []
    losses = []

    # optimzer and loss function
    optim = torch.optim.Adam(model.parameters(), lr)#use Adam optimizer
    loss_func = torch.nn.CrossEntropyLoss()

    # dataloader
    #Split train_dataset randomly, pick up 128 images each time until all images in the train data is picked
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    #pick up 128 images in test_dataset each time sequentially, until all images in the test data is picked
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



    # set seed to get constant values everytime when random function is used
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
        loss_list = []#initialise a list to store the loss each batch
        model.train#train the model
        train_predictions = []#initialise a list to store the prediction (male or female) each epoch
        train_truths = []#initialise a list to store the loss each batch
        for sample in train_dataloader:
            x, y = sample #x, y represents the image data and label 
            train_truths.extend(y.tolist())
            # reshape multi-dimension Tensor to adapt required dimentions
            #-1 means automatically adapting
            x = x.view(x.shape[0], -1)
            logit, y_hat = model(x)# y_hat is the prediction
            y_hat = y_hat.view(-1).tolist()# -1 is to adapt the dimension automatically
            train_predictions.extend(y_hat)
            loss = loss_func(logit, y)# calculate loss for this batch
            loss_list.append(loss.item())# record loss for this batch
            optim.zero_grad()# reset gradient
            loss.backward()# backward propagation, calculate current gradient for this batch
            optim.step()## update parameters
        mean_loss = np.mean(loss_list)#loss_list stores the loss for each epoch, mean_loss: the average of loss for this epoch
        print("Epoch: {}  Loss: {}".format(i, mean_loss))# print the average loss for this epoch
        losses.append(mean_loss)#record the average loss for this epoch

        # calculate accuracy on the train data
        accuracy = accuracy_score(train_truths, train_predictions)
        #add calculated value of train accuracy in this epoch to the end of "trains"
        trains.append(accuracy)
        print("Epoch: {}  Train Accuracy: {}".format(i, accuracy))

        # test
        #switch for some specific layers/parts of the model 
        #that behave differently during training and inference (evaluating) time
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
        #add calculated value of test accuracy in this epoch to the end of "tests"
        tests.append(accuracy)
        print("Epoch: {}  Test Accuracy: {}".format(i, accuracy))

        #print("predictions:{}".format(predictions))


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
    plt.savefig("A1_plots.jpg")
    plt.show()


    # 
