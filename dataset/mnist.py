# coding: utf-8
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')
import os.path
import gzip
import pickle
import os
import numpy as np
import cv2


url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',   # 학습 셋 이미지 - 55000개의 트레이닝 이미지, 5000개의 검증 이미지
    'train_label':'train-labels-idx1-ubyte.gz', # 이미지와 매칭되는 학습 셋 레이블
    'test_img':'t10k-images-idx3-ubyte.gz',     # 테스트 셋 이미지 - 10000개의 이미지
    'test_label':'t10k-labels-idx1-ubyte.gz'    # 이미지와 매칭되는 테스트 셋 레이블
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")
    
def download_mnist():
    for v in key_file.values():
       _download(v)
        
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")
    
    return labels

def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    
    print("Converting " + file_name + " to NumPy Array ...")    
    with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")
    
    return data

def _load_custom_img():
    train_dataset = np.array([])
    train_label = np.array([])
    test_dataset = np.array([])
    test_label = np.array([])

    # %: 나누기
    for i in range(0,49):
        if i%10 <= 7:
            fname = "{}.jpg".format("%_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset = np.append(train_dataset, cv2.imread(file_path,cv2.IMREAD_GRAYSCALE).flatten())
            train_label = np.append(train_label, 14)
        else :
            fname = "{}.jpg".format("%_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset = np.append(test_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label = np.append(test_label,14)

    # *: 곱하기 47
    for i in range(0,49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("*_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset = np.append(train_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label = np.append(train_label, 13)
        else :
            fname = "{}.jpg".format("*_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset = np.append(test_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label = np.append(test_label, 13)

    # +: 더하기 36
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("+_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset = np.append(train_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label = np.append(train_label, 11)
        else :
            fname = "{}.jpg".format("+_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset = np.append(test_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label = np.append(test_label, 11)

    # -: 빼기
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("-_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset = np.append(train_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label = np.append(train_label, 12)
        else :
            fname = "{}.jpg".format("-_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset = np.append(test_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label = np.append(test_label, 12)

    # 0 107
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("0_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset = np.append(train_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label = np.append(train_label, 0)
        else :
            fname = "{}.jpg".format("0_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset = np.append(test_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label = np.append(test_label, 0)

    # 1 85
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("1_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset = np.append(train_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label = np.append(train_label, 1)
        else :
            fname = "{}.jpg".format("1_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset = np.append(test_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label = np.append(test_label, 1)

    # 2 48
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("2_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset= np.append(train_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label= np.append(train_label, 2)
        else :
            fname = "{}.jpg".format("2_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset= np.append(test_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label= np.append(test_label, 2)

    # 3 37
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("3_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset= np.append(train_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label= np.append(train_label, 3)
        else :
            fname = "{}.jpg".format("3_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset= np.append(test_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label=np.append(test_label, 3)

    # 4 94
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("4_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset= np.append(train_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label= np.append(train_label, 4)
        else :
            fname = "{}.jpg".format("4_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset= np.append(test_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label= np.append(test_label,4)

    # 5 126
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("5_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset= np.append(train_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label= np.append(train_label,5)
        else :
            fname = "{}.jpg".format("5_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset= np.append(test_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label= np.append(test_label,5)

    # 6 26
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("6_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset= np.append(train_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label= np.append(train_label,6)
        else :
            fname = "{}.jpg".format("6_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset= np.append(test_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label= np.append(test_label,6)

    # 7 164
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("7_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset= np.append(train_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label= np.append(train_label,7)
        else :
            fname = "{}.jpg".format("7_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset= np.append(test_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label= np.append(test_label,7)

    # 8
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("8_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset= np.append(train_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label= np.append(train_label,8)
        else :
            fname = "{}.jpg".format("8_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset= np.append(test_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label= np.append(test_label,8)

    # 9
    for i in range(0, 49):
        if i % 10 <= 7:
            fname = "{}.jpg".format("9_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            train_dataset= np.append(train_dataset, cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            train_label= np.append(train_label,9)
        else :
            fname = "{}.jpg".format("9_{0:05d}".format(i))
            file_path = 'train_images/' + fname
            test_dataset= np.append(test_dataset,cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).flatten())
            test_label= np.append(test_label,9)


    return train_dataset.astype('uint8').reshape(-1,img_size), train_label, \
           test_dataset.astype('uint8').reshape(-1,img_size), test_label


def _convert_numpy():
    dataset = {}
    #dataset['train_img'] =  _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    #dataset['test_img'] = _load_img(key_file['test_img'])
    #dataset['test_label'] = _load_label(key_file['test_label'])
    #dataset2 = {}
    #print(dataset['train_label'])
    dataset['train_img'], dataset['train_label'], dataset['test_img'], dataset['test_label'] = _load_custom_img()
    #print(len(dataset2['train_label']))
    print(dataset['train_img'])
    #print(dataset2['train_label'])
    
    return dataset

def init_mnist():
    #download_mnist()
    dataset = _convert_numpy()
    #print(dataset['train_img'][0])
    #print(dataset['train_label'])
    #print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T
    

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNIST 데이터셋 읽기
    
    Parameters
    ----------
    normalize : 이미지의 픽셀 값을 0.0~1.0 사이의 값으로 정규화할지 정한다.
    one_hot_label : 
        one_hot_label이 True면、레이블을 원-핫(one-hot) 배열로 돌려준다.
        one-hot 배열은 예를 들어 [0,0,1,0,0,0,0,0,0,0]처럼 한 원소만 1인 배열이다.
    flatten : 입력 이미지를 1차원 배열로 만들지를 정한다. 
    
    Returns
    -------
    (훈련 이미지, 훈련 레이블), (시험 이미지, 시험 레이블)
    """
    #if not os.path.exists(save_file):
    init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            #print(type(dataset[key]))
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()
