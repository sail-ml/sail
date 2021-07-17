import sail 
import tensorflow as tf 
import torch
import numpy as np
import time

# parameters
epochs = 100
batch_size = 64
learning_rate = 1e-4

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# need to convert images to be float 32
x_train = x_train.astype(np.float32).reshape((-1, 1, 28, 28))
x_test = x_test.astype(np.float32).reshape((-1, 1, 28, 28))

# we need to normalize the data, otherwise sigmoid will overflow 
mean = np.mean(x_train)
std = np.std(x_train)

x_train = (x_train - mean)/std 
x_test = (x_test - mean)/std

# convert labels to int64
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

# define model 

class MnistModel(sail.modules.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = sail.modules.Conv2D(1, 16, (3, 3), (2, 2))
        self.linear2 = sail.modules.Conv2D(16, 32, (3, 3), (1, 1))
        self.relu1 = sail.modules.ReLU()
        self.relu2 = sail.modules.ReLU()

        self.z = z = 11 * 11 * 32
        self.linear = sail.modules.Linear(z, 10)

    def forward(self, x):
        y = self.linear1(x)
        y = self.relu1(y)
        y = self.linear2(y)
        y = self.relu2(y)
        y = sail.reshape(y, (y.shape[0], self.z))
        y = self.linear(y)
        return y

mnist = MnistModel()

def accuracy(logits, labels):
    logits = logits.numpy()
    am = np.argmax(logits, 1)
    y = am == labels 
    y = y.astype(np.int32)
    return np.mean(y)



# define optimizer
opt = sail.optimizers.SGD(learning_rate)
opt.track_module(mnist)

# define loss function
loss_fcn = sail.losses.SoftmaxCrossEntropy()

for i in range(epochs):

    start = 0
    end = batch_size
    step = batch_size
    total_loss = 0
    steps = 0

    ar = np.arange(len(x_train))
    np.random.shuffle(ar)
    x_train = x_train[ar]
    y_train = y_train[ar]

    u_time = 0
    while end < len(x_train):
        x_batch = x_train[start:end]
        y_batch = y_train[start:end]

        x_batch = sail.Tensor(x_batch)
        y_batch = sail.Tensor(y_batch)

        preds = mnist(x_batch)
       
        loss = loss_fcn(preds, y_batch)
        total_loss += loss.numpy()[0]
        loss.backward()

        opt.update()
        
        start += batch_size
        end += batch_size
        steps += 1

    pred_test = mnist(sail.Tensor(x_test))
    acc = accuracy(pred_test, y_test)

    print ("E: %s | L: %s | ACC: %s" % ((i + 1), total_loss / steps, acc))
