import random
from PIL import Image
import os
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
import cv2
# np.set_printoptions(threshold=np.inf)


def decoder_image(img, mean, std):
    inputs_decoder = []
    for ss, m, s in zip(img, mean, std):
        ss = np.array(ss * s)
        ss = np.array(ss + m)
        ss = ss * 255
        inputs_decoder.append(ss)
    return np.stack(inputs_decoder)

def my_fft(img, threshold):    
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = decoder_image(img, mean, std)   
    print('fft_img_shape:', img.shape)    
    img = np.transpose(img, (1, 2, 0))
        
    H,W,C = img.shape
    
    if threshold == None:
        thresholds = [15, 25, 35, 45, 55, 65, 75, 85]
        index = np.random.randint(0, len(thresholds))
        threshold = thresholds[index]
    
    f = np.fft.fft2(img, axes=(0,1))
    fshift = np.fft.fftshift(f)

    crows, ccols =int(H/2), int(W/2)
    mask = np.zeros((H, W, C), dtype=np.uint8)
    mask[crows-threshold:crows+threshold, ccols-threshold:ccols+threshold] = 1 #求低频   
    fshift = fshift * mask               
    ishift = np.fft.ifftshift(fshift)
    i_img = np.fft.ifft2(ishift, axes=(0,1))
    i_img_L = np.abs(i_img)

    img_H_temp = (img - i_img_L)
    img_H_temp = img_H_temp*(255/np.max(img_H_temp))*3

    img_H_temp[img_H_temp>255] = 255
    img_H_temp[img_H_temp<0] = 0
    i_img_L[i_img_L>255] = 255
    i_img_L[i_img_L<0] = 0
    
    img_H_temp = img_H_temp.astype(np.uint8)
    img_H_temp = np.array(Image.fromarray(img_H_temp).resize((112, 112)))
    i_img_L = i_img_L.astype(np.uint8)
    i_img_L = np.array(Image.fromarray(i_img_L).resize((112, 112)))
    
    img_H_temp_ag = np.array(Image.fromarray(img_H_temp).transpose(Image.FLIP_LEFT_RIGHT))
    i_img_L_ag = np.array(Image.fromarray(i_img_L).transpose(Image.FLIP_LEFT_RIGHT))

    return img_H_temp, i_img_L, img_H_temp_ag, i_img_L_ag, threshold

def gen_gaussian_noise(image,SNR):
    """
    :param image: 原始信号
    :param SNR: 添加噪声的信噪比
    :return: 生成的噪声
    """
    assert len(image.shape) == 3
    H, W, C = image.shape
    noise=np.random.randn(H, W, 1) # *signal.shape 获取样本序列的尺寸
    noise = noise - np.mean(noise)
    image_power=1/(H*W*3)*np.sum(np.power(image,2))
    noise_variance=image_power/np.power(10,(SNR/10))
    print(noise_variance)
    noise=(np.sqrt(noise_variance)/np.std(noise))*noise
    return noise

def my_fft_trans(img1, threshold):
    #数据预处理
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img1 = decoder_image(img1, mean, std)    
    
    img1 = np.transpose(img1, (1, 2, 0))
    
    H,W,C = img1.shape
    crows, ccols = H//2, W//2
    
    f1= np.fft.fft2(img1, axes=(0,1))
    
    fig_abs_temp = np.abs(f1)
    fig_pha_temp = np.angle(f1)
    
    #高频
    noise_w_h_p = np.random.uniform(0.8, 1.2, (H,W,1))
    noise_b_h_p = np.random.uniform(-np.pi/6, np.pi/6, (H,W,1))
    fig_pha_ag = noise_w_h_p*fig_pha_temp + noise_b_h_p
    
    noise_w_h_a = np.random.uniform(0.5, 1.5, (H,W,1))
    noise_b_h_a = gen_gaussian_noise(fig_abs_temp, 30)
    fig_abs_ag = noise_w_h_a*fig_abs_temp + noise_b_h_a
    
    f_ag = fig_abs_ag*np.cos(fig_pha_ag) + fig_abs_ag*np.sin(fig_pha_ag)*1j
    
#     #低频
#     fig_l_abs = fig_abs_temp[crows-threshold:crows+threshold, ccols-threshold:ccols+threshold]
#     fig_l_pha = fig_pha_temp[crows-threshold:crows+threshold, ccols-threshold:ccols+threshold]
#     noise_w_l_p = np.random.uniform(0.8, 1.2, (threshold*2,threshold*2,1))
#     noise_b_l_p = gen_gaussian_noise(fig_l_pha, 30)
#     fig_l_pha_ag = noise_w_l_p*fig_l_pha + noise_b_l_p
    
#     noise_w_l_a = np.random.uniform(0.5, 1.5, (threshold*2,threshold*2,1))
#     noise_b_l_a = gen_gaussian_noise(fig_l_abs, 30)
#     fig_l_abs_ag = noise_w_l_a*fig_l_abs + noise_b_l_a
    
#     f_ag_l = fshift_l_abs_ag*np.cos(fshift_l_pha_ag) + fshift_l_abs_ag*np.sin(fshift_l_pha_ag)*1j
            
#     f_ag[crows-threshold:crows+threshold, ccols-threshold:ccols+threshold] = f_ag_l
            
    #反变换
    img_ag = np.fft.ifft2(f_ag, axes=(0,1))
    img_ag = np.abs(img_ag)
    img_ag = np.uint8(np.clip(img_ag, 0, 255))   
    
    img_ag = np.array(Image.fromarray(img_ag).transpose(Image.FLIP_LEFT_RIGHT))

    return img_ag
 
    
def MMD_Loss_func(num_source, sigmas=None):
    if sigmas is None:
        sigmas = [1, 5, 10]
    def loss(e_pred,d_ture):
        cost = 0.0
        for i in range(num_source):
            domain_i = e_pred[d_ture == i]
            for j in range(i+1,num_source):
                domain_j = e_pred[d_ture == j]
                single_res = mmd_two_distribution(domain_i,domain_j,sigmas=sigmas)
                cost += single_res
        return cost
    return loss

def mmd_two_distribution(source, target, sigmas):
    sigmas = torch.tensor(sigmas).cuda()
    xy = rbf_kernel(source, target, sigmas)
    xx = rbf_kernel(source, source, sigmas)
    yy = rbf_kernel(target, target, sigmas)
    return xx + yy - 2 * xy

def rbf_kernel(x, y, sigmas):
    sigmas = sigmas.reshape(sigmas.shape + (1,))
    beta = 1. / (2. * sigmas)
    dist = compute_pairwise_distances(x, y)
    dot = -torch.matmul(beta, torch.reshape(dist, (1, -1)))
    exp = torch.mean(torch.exp(dot))
    return exp

def compute_pairwise_distances(x, y):
    dist = torch.zeros(x.size(0),y.size(0)).cuda()
    for i in range(x.size(0)):
        dist[i,:] = torch.sum(torch.square(x[i].expand(y.shape) - y),dim=1)
    return dist        
    
    
def unfold_label(labels, classes):
    # can not be used when classes are not complete
    new_labels = []

    assert len(np.unique(labels)) == classes
    # minimum value of labels
    mini = np.min(labels)

    for index in range(len(labels)):
        dump = np.full(shape=[classes], fill_value=0).astype(np.int8)
        _class = int(labels[index]) - mini
        dump[_class] = 1
        new_labels.append(dump)

    return np.array(new_labels)


def shuffle_data(samples, labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num)) #打乱index
    print(type(samples))
    print(type(labels))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_samples, shuffled_labels


def shuffle_list(li):
    np.random.shuffle(li)
    return li


def shuffle_list_with_ind(li):
    shuffle_index = np.random.permutation(np.arange(len(li)))
    li = li[shuffle_index]
    return li, shuffle_index


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def crossentropyloss():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn


def mseloss():
    loss_fn = torch.nn.MSELoss()
    return loss_fn


def sgd(parameters, lr, weight_decay=0.00005, momentum=0.9):
    opt = optim.SGD(params=parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    return opt


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def fix_python_seed(seed):
    print('seed-----------python', seed)
    random.seed(seed)
    np.random.seed(seed)


def fix_torch_seed(seed):
    print('seed-----------torch', seed)
    torch.manual_seed(seed) #为cpu设置随机种子
    torch.cuda.manual_seed_all(seed) #为gpu设置随机种子


def fix_all_seed(seed):
    print('seed-----------all device', seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_accuracy(predictions, labels):
    if np.ndim(labels) == 2:
        y_true = np.argmax(labels, axis=-1)
    else:
        y_true = labels
    accuracy = accuracy_score(y_true=y_true, y_pred=np.argmax(predictions, axis=-1))
    return accuracy
