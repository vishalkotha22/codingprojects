#!/usr/bin/env python
# coding: utf-8

# # TJHSST Computer Vision Club: Contest 1 - Data Augmentation
# The goal of this week's contest is to gain some exposure to PyTorch as well as the practice of data augmentation.

# ### Step 0. Imports

# In[41]:


import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import os


# ### Step 1. Downloading and Preparing Data
# PyTorch's torchvision.datasets module has data-downloading capability, which we will be using here.

# In[42]:


from torchvision import datasets, transforms
working_dir = os.getcwd()
normalize_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,)),
                                         ])
trainset = datasets.MNIST(working_dir, download=True, train=True, transform=normalize_transform)
testset = datasets.MNIST(working_dir, download=True, train=False, transform=normalize_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

###For the purposes of our contest, we want to reduce the data in the training set
trainset.targets = trainset.targets.narrow(0, 0, 5000)
trainset.data = trainset.data.narrow(0, 0, 5000)


# ### Step 2. Define our Neural Network
# This is just a simple fully connected network - by no means optimal, but good enough for what we're doing. While a CNN is usually 'better' at learning, convolutions are very computationally expensive (especially without a GPU).

# In[43]:


from torch import nn

input_size = 784
hidden_sizes = [128, 64]
output_size = 10
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))


# ### Step 3. Define Training and Testing Functions
# These are the main training and testing loops we'll be using. There's also a function to nicely print accuracies from the test function.

# In[61]:


def train(model, loader, optimizer, criterion, n_epochs=10):
    for e in range(n_epochs):
        running_loss = 0
        for images, labels in loader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {e} - Traning Loss: {running_loss/len(loader)}")
def test(model, loader):
    correct_nums = np.array([0 for x in range(10)])
    total_nums = np.array([0 for x in range(10)])
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            for i, v in enumerate(predicted):
                is_correct = (v.item() == labels[i].item())
                correct_nums[labels[i].item()] += is_correct
                total_nums[labels[i].item()] += 1
    return correct_nums / total_nums
def reportAccuracy(accuracies):
    print("Category\tAccuracy")
    for x in range(10):
        print("{}:\t\t{:.2f}%".format(x, 100*accuracies[x]))
    print("Average:\t{:.2f}%".format(100*np.mean(accuracies)))


# ### Step 4. Train our Model (or load pretrained model from file)
# If you don't download the state dict from Kaggle, this will train the network from scratch. But that's just a waste of time, so if the file 'data_aug_state_dict.pt' is available it will use that instead :)

# In[62]:


from torch import optim
saved_model_file = 'data_aug_state_dict.pt'
model_location = os.path.join(working_dir, saved_model_file)
if os.path.exists(model_location): #load the weights if the file exists
    model.load_state_dict(torch.load(model_location))
else: #otherwise, retrain the model (takes a while)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    criterion = nn.NLLLoss()
    train(model, trainloader, optimizer, criterion, n_epochs = 50)
    torch.save(model.state_dict(), model_location)
acc = test(model, testloader)
reportAccuracy(acc)


# ### Step 5. Write a function that uses OpenCV to augment data (YOU DO THIS!)
# The function `transformImage` is the only thing you need to modify. Just so you can tell what/how to use OpenCV functions, I have added an example for rotating each image 10 degrees to the left and right. The function should return a numpy array of all the transformations for each image. Also, an important thing when transforming the image is using the correct 'borderValue'. Because the MNIST images get normalized, this means 0, which would represent a black pixel, is now -1. As such, the border value should be -1.

# In[46]:


import cv2
def formatTensor(img):
    np_img = img.numpy().transpose(1, 2, 0)
    return np_img
def transformImage(img):
    orig_img = img.numpy()[0]
    #first, rearrange the tensor to something opencv can parse
    my_img = formatTensor(img)
    #next, add some transformations!
    #example of a slight rotation (taken from demo given on 10/14)
    center = (14, 14)
    angle = 10
    scale = 1
    borderArgs = {'borderMode': cv2.BORDER_CONSTANT, 'borderValue': -1}
    images = []
    images.append(orig_img)
    for i in range(30):
        rot_matrix = cv2.getRotationMatrix2D(center, i, scale)
        rot_matrix2 = cv2.getRotationMatrix2D(center, -i, scale)
        rot_img = cv2.warpAffine(my_img, rot_matrix, my_img.shape[:-1], **borderArgs)
        rot_img2 = cv2.warpAffine(my_img, rot_matrix2, my_img.shape[:-1], **borderArgs)
        images.append(rot_img)
        images.append(rot_img2)
    return np.array(images) #note that you also return the original image


# ### Step 5.5: Preview the output of our transform function
# Below, the first image is the original. The two that follow it are rotations.

# In[63]:


def displayTensor(img):
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.close()

my_img, _ = trainset[0]
for new_img in transformImage(my_img):
    displayTensor(new_img)


# ### Step 6. Augment the entire dataset and save it (may take a while)
# This is just a loop that applies the `transformImage` function to each image in the dataset. Depending on how many transformations you add, this could take a while to save. Note that this will create the directory 'MNIST_augment/train.pt'

# In[48]:


new_tensors = []
new_labels = []
for i, (image, label) in enumerate(trainset):
    augment_images = transformImage(image)
    reshaped_imgs = np.reshape(augment_images, (-1, 1, 28, 28))
    as_tensor = torch.from_numpy(reshaped_imgs)
    new_tensors.append(as_tensor)
    for l in range(as_tensor.shape[0]):
        new_labels.append(label)
new_images = torch.cat(new_tensors)
new_labels = torch.Tensor(new_labels)
#make a folder
save_folder = os.path.join(working_dir, 'MNIST_augment')
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
save_file = os.path.join(save_folder, 'train.pt')
torch.save((new_images, new_labels), save_file)


# ### Step 7: Write a Dataset class to read our augmented data
# PyTorch doesn't provide a class for reading data from a saved tensor, so we'll write that ourselves.

# In[49]:


from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, root):
        self.data, self.targets = torch.load(root)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        return img, target
augment_trainset = MyDataset(save_file)
augment_trainloader = DataLoader(augment_trainset, batch_size=64, shuffle=False)


# ### Step 8: Train on the augmented data
# Using the same optimizer and criterion as the first training loop, we'll do 50 epochs (iterations) over the augmented dataset. If you want, you can increase n_epochs (in fact, you probably should).

# In[64]:


optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
criterion = nn.NLLLoss()
train(model, augment_trainloader, optimizer, criterion, n_epochs = 50)
acc = test(model, testloader)
reportAccuracy(acc)


# ### Step 9: Save Results to CSV
# Now that our model has been retrained on the augmented dataset, let's save it to a CSV, which you can then upload to Kaggle.

# In[75]:


with open("results.csv", "w") as f:
    f.write("Id,Category\n")
    for i, (img, label) in enumerate(testset):
        with torch.no_grad():
            img = img.view(img.shape[0], -1)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            f.write(f"{i},{predicted.item()}\n")
print("complete")
f.close()