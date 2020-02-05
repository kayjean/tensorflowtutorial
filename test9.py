from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 讀入 MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
x_train = mnist.train.images
y_train = mnist.train.labels
x_test = mnist.test.images
y_test = mnist.test.labels

# 檢視結構
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print("---")

# 檢視一個觀測值
#print(x_train[1, :])
print(np.argmax(y_train[1, :])) # 第一張訓練圖片的真實答案