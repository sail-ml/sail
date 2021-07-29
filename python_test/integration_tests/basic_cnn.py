from test_case import *
import numpy as np
import sail
import time
import unittest, random
import tensorflow as tf 

class BasicCNN(UnitTest):

    # UnitTest._test_registry.append(AddTest)
    def test_base(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype(np.float32).reshape((-1, 1, 28, 28))
        x_test = x_test.astype(np.float32).reshape((-1, 1, 28, 28))

        mean = np.mean(x_train)
        std = np.std(x_train)

        x_train = (x_train - mean)/std 
        x_test = (x_test - mean)/std

        # convert labels to int64
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        ar = np.arange(len(x_train))
        np.random.shuffle(ar)

        x_train = x_train[ar]
        y_train = y_train[ar]
        
        ar = np.arange(len(x_test))
        np.random.shuffle(ar)

        x_test = x_test[ar]
        y_test = y_test[ar]

        x_train = x_train[:5000]
        y_train = y_train[:5000]
        x_test = x_test[:500]
        y_test = y_test[:500]

    # define model 

        class MnistModel(sail.modules.Module):

            def __init__(self):
                super().__init__()
                self.linear1 = sail.modules.Conv2D(1, 4, (5, 5), (1, 1))
                self.linear2 = sail.modules.Conv2D(4, 8, (5, 5), (2, 2))
                self.maxpool2d = sail.modules.MaxPool2D(2)
                self.relu1 = sail.modules.ReLU()
                self.relu2 = sail.modules.ReLU()

                self.z = z = 8 * 4 * 4
                self.linear = sail.modules.Linear(z, 10)

            def forward(self, x):
                y = self.linear1(x)
                y = self.relu1(y)
                y = self.maxpool2d(y)
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

        epochs = 10
        batch_size = 32
        learning_rate = 1e-4

        # define optimizer
        opt = sail.optimizers.SGD(learning_rate)
        opt.track_module(mnist)

        # define loss function
        loss_fcn = sail.losses.SoftmaxCrossEntropy()
        losses = []
        accs = []

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

            losses.append(total_loss/steps)
            pred_test = mnist(sail.Tensor(x_test))
            acc = accuracy(pred_test, y_test)
            accs.append(acc)


        self.assert_gt(losses[0], losses[-1])

        return