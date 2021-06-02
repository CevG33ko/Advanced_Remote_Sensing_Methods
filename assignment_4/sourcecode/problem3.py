import os
import time
import pandas as pd


from typing import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.metrics import f1_score

import logging
from functools import lru_cache

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

class MyNet(nn.Module):
    # define nn
    def __init__(self, input_size, num_classes):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size,10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(10, num_classes)
        self.relu3 = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu1(out)
        #out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.relu2(out)
        #out = self.sigmoid(out)
        out = self.fc3(out)
        #out = self.relu3(out)
        #out = self.sigmoid(out)
        #out = self.softmax(out)

        return out

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

# First generate text files to randomly split the original datat irisMy.csv into train and test dataset (80/20)
label_csv_path = os.path.join('./', 'dataset/one-hundred-plants-texture.csv')


df = pd.read_csv(label_csv_path)

train_size = 0.8 # includes training and validation


mask = np.random.rand(len(df)) < train_size
train = df[mask].reset_index(drop = True)
train.to_csv(os.path.join('./', 'dataset/plants_train_split.csv'),index = False)

val = df[~mask].reset_index(drop = True)
val.to_csv(os.path.join('./', 'dataset/plants_test_split.csv'),index = False)

# Fclass to provide input data for net and classes
class IrisDataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    self.dataset = dataset
    
    x_tmp = dataset[:, 0:64]
    y_tmp = dataset[:, 64]
    self.x_data = torch.tensor(x_tmp, dtype=torch.float32)
    self.y_data = torch.tensor(y_tmp, dtype=torch.int64)
 
  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    
    preds = self.x_data[idx]
    species = self.y_data[idx] - 1
    sample = (preds, species)
    
    return sample

def get_data(batch_size = 128, data_root='dataset', num_workers=0):
   
  val_size = 0.25
 
  dataset = np.loadtxt("dataset/plants_train_split.csv", usecols=range(0,65), delimiter=",", skiprows=1, dtype=np.float32)
    
  dataset_train, dataset_val = train_test_split(dataset, test_size = val_size, random_state=0)
   
  train_ds = IrisDataset(dataset_train)
  val_ds = IrisDataset(dataset_val)
 
  train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers = 0)
  val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers = 0) 
 
    
  return train_loader, val_loader

@dataclass
class SystemConfiguration:
    '''
    Describes the common system setting needed for reproducible training
    '''
    seed: int = 21  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = True  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)

@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 6400 #amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 8000  # number of times the whole dataset will be passed through the network
    learning_rate: float = 1  # determines the speed of network's weights update
    log_interval: int = 100  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    data_root: str = "./dataset"  # folder to save dataset (default: data)
    num_workers: int = 0  # number of concurrent processes used to prepare data
    device: str = 'cuda'  # device to use for training.
    
def setup_system(system_config: SystemConfiguration) -> None:
    torch.manual_seed(system_config.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic

def save_model(model, device, model_dir='models', model_file_name='plants.pt'):
    

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # make sure you transfer the model to cpu.
    if device == 'cuda':
        model.to('cpu')

    # save the state_dict
    torch.save(model.state_dict(), model_path)
    
    if device == 'cuda':
        model.to('cuda')
    
    return

def load_model(model, model_dir='models', model_file_name='plants.pt'):
    model_path = os.path.join(model_dir, model_file_name)

    # loading the model and getting model parameters by using load_state_dict
    model.load_state_dict(torch.load(model_path))
    
    return model

def train(
    train_config: TrainingConfiguration, model: nn.Module, optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader, epoch_idx: int
) -> None:
    
    # change model in training mode
    model.train()
    
    # to get batch loss
    batch_loss = np.array([])
    
    # to get batch accuracy
    batch_acc = np.array([])
        
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # clone target
        indx_target = target.clone()
        # send data to device (it is mandatory if GPU has to be used)
        data = data.to(train_config.device)
        # send target to device
        target = target.to(train_config.device)

        # reset parameters gradient to zero
        optimizer.zero_grad()
        
        # forward pass to the model
        output = model(data)
        
        # cross entropy loss
        loss = F.cross_entropy(output, target)
        
        # find gradients w.r.t training parameters
        loss.backward()
        
        # Update parameters using gradients
        optimizer.step()
        
        batch_loss = np.append(batch_loss, [loss.item()])
        
        # get probability score using softmax
        prob = F.softmax(output, dim=1)
            
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]  
                        
        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()
            
        # accuracy
        acc = float(correct) / float(len(data))
        
        batch_acc = np.append(batch_acc, [acc])

        #if batch_idx % train_config.log_interval == 0 and batch_idx > 0:              
        #    print(
        #        'Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
        #            epoch_idx, batch_idx * len(data), len(train_loader.dataset), loss.item(), acc
        #        )
        #    )
            
    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.mean()
    
 
    return epoch_loss, epoch_acc

def validate(
    train_config: TrainingConfiguration,
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
) -> float:
    # 
    model.eval()
    test_loss = 0
    count_corect_predictions = 0
    for data, target in test_loader:
        indx_target = target.clone()
        data = data.to(train_config.device)
        
        target = target.to(train_config.device)
        
        output = model(data)
        # add loss for each mini batch
        test_loss += F.cross_entropy(output, target).item()
        
        # get probability score using softmax
        prob = F.softmax(output, dim=1)
        
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1] 
        
        # add correct prediction count
        count_corect_predictions += pred.cpu().eq(indx_target).sum()

    # average over number of mini-batches
    test_loss = test_loss / len(test_loader)  
    
    # average over number of dataset
    accuracy = 100. * count_corect_predictions / len(test_loader.dataset)
    
    #print(
    #    'Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #        test_loss, count_corect_predictions, len(test_loader.dataset), accuracy
    #    )
    #)
    acc = accuracy/100.0
    
    return test_loss, acc

def main(model, optimizer, system_configuration=SystemConfiguration(), 
         training_configuration=TrainingConfiguration()):
    
    # system configuration
    setup_system(system_configuration)

    # batch size
    batch_size_to_set = training_configuration.batch_size
    # num_workers
    num_workers_to_set = training_configuration.num_workers
    # epochs
    epoch_num_to_set = training_configuration.epochs_count

    # if GPU is available use training config, 
    # else lower batch_size, num_workers and epochs count
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        #batch_size_to_set = 16
        num_workers_to_set = 0

    # data loader
    train_loader, test_loader = get_data(
        batch_size=batch_size_to_set,
        data_root=training_configuration.data_root,
        num_workers=num_workers_to_set
    )
    
    # Update training configuration
    training_configuration = TrainingConfiguration(
        device=device,
        batch_size=batch_size_to_set,
        num_workers=num_workers_to_set
    )
        
    # send model to device (GPU/CPU)
    model.to(training_configuration.device)

    best_loss = torch.tensor(np.inf)
    
    # epoch train/test loss
    epoch_train_loss = np.array([])
    epoch_test_loss = np.array([])
    
    # epoch train/test accuracy
    epoch_train_acc = np.array([])
    epoch_test_acc = np.array([])
    
    # trainig time measurement
    t_begin = time.time()
    for epoch in range(training_configuration.epochs_count):
        
        train_loss, train_acc = train(training_configuration, model, optimizer, train_loader, epoch)
        
        #print('train_acc=', train_acc)
        
        epoch_train_loss = np.append(epoch_train_loss, [train_loss])
        
        epoch_train_acc = np.append(epoch_train_acc, [train_acc])

        elapsed_time = time.time() - t_begin
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_loader)
        eta = speed_epoch * training_configuration.epochs_count - elapsed_time
        
        if epoch%train_config.log_interval == 0 or epoch == 1:
            print(
                "\nEpoch: {}, Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(epoch,
                elapsed_time, speed_epoch, speed_batch, eta
                )
            )
            print('Epoch: {} Loss: {:.6f} Acc: {:.4f}'.format(epoch, train_loss, train_acc))
  
        
        # Validate
        if epoch % training_configuration.test_interval == 0:
            current_loss, current_accuracy = validate(training_configuration, model, test_loader)
            
            epoch_test_loss = np.append(epoch_test_loss, [current_loss])
        
            epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])
            
            if epoch%train_config.log_interval == 0 or epoch == 1:
                print('Validation set: Average loss: {:.4f}, Accuracy: {:.1f}\n'.format(
                    current_loss, current_accuracy
                    )
                )
            
            if current_loss < best_loss:
                best_loss = current_loss
                #print('Model Improved. Saving the Model...\n')
                save_model(model, device=training_configuration.device)
        
                
        
    
    return model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc

input_size = 64
num_classes = 100

model = MyNet(input_size, num_classes)

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=train_config.learning_rate
)


model, train_loss, train_acc, test_loss, test_acc = main(model, optimizer)

# plot losses
plt.plot(train_loss, label = 'Training loss')
plt.plot(test_loss, label = 'Validation loss')
plt.legend(frameon=False)
plt.show()

# plot training accuracies
plt.plot(train_acc, label = 'Training accuracy')
plt.plot(test_acc, label = 'Validation accuracy')
plt.legend(frameon=False)
plt.show()

@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 6000 #amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 5000  # number of times the whole dataset will be passed through the network
    learning_rate: float = 1  # determines the speed of network's weights update
    log_interval: int = 100  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    data_root: str = "./dataset"  # folder to save dataset (default: data)
    num_workers: int = 0  # number of concurrent processes used to prepare data
    device: str = 'cuda'  # device to use for training.
input_size = 64
num_classes = 100

model = MyNet(input_size, num_classes)

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=train_config.learning_rate
)


model, train_loss, train_acc, test_loss, test_acc = main(model, optimizer)

# plot losses
plt.plot(train_loss, label = 'Training loss')
plt.plot(test_loss, label = 'Validation loss')
plt.legend(frameon=False)
plt.show()

# plot training accuracies
plt.plot(train_acc, label = 'Training accuracy')
plt.plot(test_acc, label = 'Validation accuracy')
plt.legend(frameon=False)
plt.show()

class MyNet(nn.Module):
    # define nn
    def __init__(self, input_size, num_classes):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size,11)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5, 4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, 2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(11, num_classes)
        self.relu4 = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu1(out)
        #out = self.sigmoid(out)
        #out = self.fc2(out)
        #out = self.relu2(out)
        #out = self.sigmoid(out)
        #out = self.fc3(out)
        #out = self.relu3(out)
        out = self.fc4(out)
        #out = self.relu4(out)
        #out = self.sigmoid(out)
        #out = self.softmax(out)

        return out

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 8000 #amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 10000  # number of times the whole dataset will be passed through the network
    learning_rate: float = 1  # determines the speed of network's weights update
    log_interval: int = 100  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    data_root: str = "./dataset"  # folder to save dataset (default: data)
    num_workers: int = 0  # number of concurrent processes used to prepare data
    device: str = 'cuda'  # device to use for training.
input_size = 64
num_classes = 100

model = MyNet(input_size, num_classes)

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=train_config.learning_rate
)

model, train_loss, train_acc, test_loss, test_acc = main(model, optimizer)

#classes -> 10 -> classes
# plot losses bsize 6.4k lr 0.2
plt.plot(train_loss, label = 'Training loss')
plt.plot(test_loss, label = 'Validation loss')
plt.legend(frameon=False)
plt.show()

# plot training accuracies
plt.plot(train_acc, label = 'Training accuracy')
plt.plot(test_acc, label = 'Validation accuracy')
plt.legend(frameon=False)
plt.show()

class MyNet(nn.Module):
    # define nn
    def __init__(self, input_size, num_classes):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size,10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5, 4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, 2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(10, num_classes)
        self.relu4 = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu1(out)
        #out = self.sigmoid(out)
        #out = self.fc2(out)
        #out = self.relu2(out)
        #out = self.sigmoid(out)
        #out = self.fc3(out)
        #out = self.relu3(out)
        out = self.fc4(out)
        #out = self.relu4(out)
        #out = self.sigmoid(out)
        #out = self.softmax(out)

        return out

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 8000 #amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 10000  # number of times the whole dataset will be passed through the network
    learning_rate: float = 1  # determines the speed of network's weights update
    log_interval: int = 100  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    data_root: str = "./dataset"  # folder to save dataset (default: data)
    num_workers: int = 0  # number of concurrent processes used to prepare data
    device: str = 'cuda'  # device to use for training.
input_size = 64
num_classes = 100

model = MyNet(input_size, num_classes)

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=train_config.learning_rate
)

model, train_loss, train_acc, test_loss, test_acc = main(model, optimizer)

#classes -> 10 -> classes
# plot losses bsize 8k lr 1
plt.plot(train_loss, label = 'Training loss')
plt.plot(test_loss, label = 'Validation loss')
plt.legend(frameon=False)
plt.show()

# plot training accuracies
plt.plot(train_acc, label = 'Training accuracy')
plt.plot(test_acc, label = 'Validation accuracy')
plt.legend(frameon=False)
plt.show()

@dataclass
class TrainingConfiguration:
    '''
    Describes configuration of the training process
    '''
    batch_size: int = 6400 #amount of data to pass through the network at each forward-backward iteration
    epochs_count: int = 8000  # number of times the whole dataset will be passed through the network
    learning_rate: float = 1  # determines the speed of network's weights update
    log_interval: int = 100  # how many batches to wait between logging training status
    test_interval: int = 1  # how many epochs to wait before another test. Set to 1 to get val loss at each epoch
    data_root: str = "./dataset"  # folder to save dataset (default: data)
    num_workers: int = 0  # number of concurrent processes used to prepare data
    device: str = 'cuda'  # device to use for training.
        
class MyNet(nn.Module):
    # define nn
    def __init__(self, input_size, num_classes):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size,4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5, 4)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2, 2)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(4, num_classes)
        self.relu4 = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu1(out)
        #out = self.sigmoid(out)
        #out = self.fc2(out)
        #out = self.relu2(out)
        #out = self.sigmoid(out)
        #out = self.fc3(out)
        #out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        #out = self.sigmoid(out)
        #out = self.softmax(out)

        return out

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
input_size = 64
num_classes = 100

model = MyNet(input_size, num_classes)

train_config = TrainingConfiguration()

# optimizer
optimizer = optim.SGD(
    model.parameters(),
    lr=train_config.learning_rate
)


model, train_loss, train_acc, test_loss, test_acc = main(model, optimizer)
# plot losses
plt.plot(train_loss, label = 'Training loss')
plt.plot(test_loss, label = 'Validation loss')
plt.legend(frameon=False)
plt.show()

# plot training accuracies
plt.plot(train_acc, label = 'Training accuracy')
plt.plot(test_acc, label = 'Validation accuracy')
plt.legend(frameon=False)
plt.show()

# plot losses
plt.plot(train_loss, label = 'Training loss')
plt.plot(test_loss, label = 'Validation loss')
plt.legend(frameon=False)
plt.show()

# plot training accuracies
plt.plot(train_acc, label = 'Training accuracy')
plt.plot(test_acc, label = 'Validation accuracy')
plt.legend(frameon=False)
plt.show()

def printStatistics(labels_val, preds_val, state):
    
        print('---------------------------------------------------------------------')
        kappa = cohen_kappa_score(labels_val,preds_val, labels=None, weights=None, sample_weight=None)   
        print('\n' + 'Cohen’s kappa (on '+state+' dataset): ' + str(round(kappa,4)))
        # confusion_matrix
        print('\n' + 'Confusion Matrix: ' +state+ '\n')
        
        cm=confusion_matrix(labels_val,preds_val)
        
        print(cm)
        # precision & recall
        pr=precision_score(labels_val,preds_val, average=None)
        re=recall_score(labels_val,preds_val, average=None)
        ff=f1_score(labels_val,preds_val, average=None)
        dt2=pd.DataFrame(data=[pr,re,ff],
                         index=['precision','recall','f1-score'])
        print(dt2.round(4))
        acc = accuracy_score(labels_val,preds_val)
        print(state + ' accuracy:', acc)
        report = classification_report(test_y.data, preds_val.data, digits=4)
        print(report)
        print('---------------------------------------------------------------------')

        # Using the best-performing model, perform prediction over the testing set,
# which should have been used during fine-tuning process

val_model = load_model(model, model_dir='models', model_file_name='plants.pt')
#model.to(training_configuration.device)
dataset = dataset = pd.read_csv("dataset/plants_test_split.csv")

Test_X = dataset.iloc[:,:-1]
test_y = dataset['Class']

Test_X = torch.tensor(Test_X.iloc[:,0:64].values).float()#.cpu()
test_y = torch.tensor(test_y.iloc[:].values) - 1

#model.load_state_dict(val_model)

# set model in evaluation mode
# Labels can be estimated

val_model.eval()
test_pred = model(Test_X)
test_pred = torch.exp(test_pred)

# seeks for the class with hightest prediction probability
top_p, top_class_test = test_pred.topk(1, dim = 1)
acc_test = accuracy_score(test_y,top_class_test)

# print statistics test data set
printStatistics(test_y,top_class_test,'Testing')


