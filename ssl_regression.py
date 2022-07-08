# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 23:07:57 2022

@author: Jimmy
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.layer1 = nn.Linear(1, 10)
    self.layer2 = nn.Linear(10, 1)
    self.act = nn.ReLU()

  def forward(self, x):
    y = self.layer1(x)
    y = self.act(y)  
    y = self.layer2(y)

    return y

def gt_function(x):
    y = np.exp(x) * np.power(x, 5) + 3 * x + np.random.rand(x.shape[0]) * 0.8
    return y

if __name__ == '__main__':
    
    # training data
    x = np.random.rand(100)
    y = gt_function(x)
    
    plt.scatter(x, y)
    plt.show()
    
    # convert numpy array to tensor in shape of input size
    x = torch.from_numpy(x.reshape(-1,1)).float()
    y = torch.from_numpy(y.reshape(-1,1)).float()
    
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()
    
    
    # train
    inputs = Variable(x)
    outputs = Variable(y)
    for i in range(500):
        prediction = net(inputs)
        loss = loss_func(prediction, outputs) 
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()       
    
        if i % 50 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            new_x, new_y = zip(*sorted(zip(x.data.numpy(), prediction.data.numpy())))
            plt.plot(new_x, new_y, 'r-', lw=2)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
            plt.title('First train / epoch: {}'.format(i))
            plt.show()
         
    # test data
    test = np.random.rand(500)
    test_gt = gt_function(test)
    test_gt = torch.from_numpy(test_gt.reshape(-1,1)).float()
    test_gt = Variable(test_gt)
    test = torch.from_numpy(test.reshape(-1,1)).float()
    
    # evaluation for first training
    test_pred = net(Variable(test))
    plt.cla()
    plt.scatter(test.data.numpy(), test_gt.data.numpy(), c='green')
    new_x, new_y = zip(*sorted(zip(test.data.numpy(), test_pred.data.numpy())))
    plt.plot(new_x, new_y, 'r-', lw=2)
    plt.text(0.5, 0, 'Loss=%.4f' % loss_func(test_pred, test_gt).data.numpy(), fontdict={'size': 10, 'color':  'red'})
    plt.title('First train / test result')
    plt.show()
           
    # generate pseudo-label from additional data
    z = np.random.rand(100)
    z = torch.from_numpy(z.reshape(-1,1)).float()
    z_pred = net(Variable(z))
    
    # train with additional pseudo-label
    inputs = Variable(torch.cat((x, z), 0))
    outputs = Variable(torch.cat((y, z_pred), 0))
    for i in range(500):
        prediction = net(inputs)
        loss = loss_func(prediction, outputs) 
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()   
       
        if i % 50 == 0:
            # plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            new_x, new_y = zip(*sorted(zip(x.data.numpy(), prediction[:x.size(0)].data.numpy())))
            plt.plot(new_x, new_y, 'r-', lw=2)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 10, 'color':  'red'})
            plt.title('Re-train / epoch: {}'.format(i))
            plt.show()
       
    # evaluation for re-training
    test_pred = net(Variable(test))
    plt.cla()
    plt.scatter(test.data.numpy(), test_gt.data.numpy(), c='green')
    new_x, new_y = zip(*sorted(zip(test.data.numpy(), test_pred.data.numpy())))
    plt.plot(new_x, new_y, 'r-', lw=2)
    plt.text(0.5, 0, 'Loss=%.4f' % loss_func(test_pred, test_gt).data.numpy(), fontdict={'size': 10, 'color':  'red'})
    plt.title('Re-train / test result')
    plt.show()
