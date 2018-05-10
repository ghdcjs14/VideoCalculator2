# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import cv2
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
print(type(x_train))

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

sampled = 10000 # 고속화를 위한 표본추출
x_test = x_test[:sampled]
t_test = t_test[:sampled]

#print("caluculate accuracy (float64) ... ")
#print(network.accuracy(x_test, t_test))

print("predict (float64) ... ")
print(x_test[0][0][0][0])
#print(network.predict(x_train[0]))

original = cv2.imread('images/digits3.jpg', cv2.IMREAD_COLOR)
gray = cv2.imread('image_result/gray.jpg', cv2.IMREAD_COLOR)

print(original.shape)
print(gray.shape)

arr = np.array([original,gray])
print(arr.shape)

print(network.predict(arr))

# float16(반정밀도)로 형변환
#x_test = x_test.astype(np.float16)
#for param in network.params.values():
#    param[...] = param.astype(np.float16)

#print("caluculate accuracy (float16) ... ")
#print(network.accuracy(x_test, t_test))
