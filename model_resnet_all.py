import os

dirpath = os.pardir
import sys

sys.path.append(dirpath)
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.optim import lr_scheduler

import resnet_vanilla_updata
from common.data_reader import BatchImageGenerator #change
from common.utils import *
import cv2

class ModelAggregate:
    def __init__(self, flags):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            print("使用cpu........")


        self.setup(flags)
        self.setup_path(flags)
        self.configure(flags)
        #归一化时使用的数据
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def setup(self, flags):
        
        model = resnet_vanilla_updata.resnet18(pretrained=True, num_classes=flags.num_classes, num_domains=flags.num_domains, flags=flags)
    
        if torch.cuda.is_available():
#             model = model.cuda()
            if torch.cuda.device_count()>1:
                model = torch.nn.DataParallel(model)
            self.network = model.cuda()
        else:
            print("使用cpu........")

            self.network = model.cpu()

        print(self.network)
        print('flags:', flags)
        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)

#         self.load_state_dict(flags, self.network)

    def setup_path(self, flags):

        root_folder = flags.data_root
        train_data = [
                     'art_painting_train.hdf5',
                      'cartoon_train.hdf5',
                        'photo_train.hdf5',
                        'sketch_train.hdf5'
                     ]

        val_data = [
                    'art_painting_val.hdf5',
                    'cartoon_val.hdf5',
                      'photo_val.hdf5',
                      'sketch_val.hdf5'
                   ]

        test_data = [
                    'art_painting_test.hdf5',
                     'cartoon_test.hdf5',
                      'photo_test.hdf5',
                      'sketch_test.hdf5'
                    ]

        self.train_paths = []
        for data in train_data:
            path = os.path.join(root_folder, data)
            self.train_paths.append(path)

        self.val_paths = []
        for data in val_data:
            path = os.path.join(root_folder, data)
            self.val_paths.append(path)

        unseen_index = flags.unseen_index

        self.unseen_data_path = os.path.join(root_folder, test_data[unseen_index])
        self.train_paths.remove(self.train_paths[unseen_index])
        self.val_paths.remove(self.val_paths[unseen_index])

        if not os.path.exists(flags.logs):
            os.makedirs(flags.logs)

        flags_log = os.path.join(flags.logs, 'path_log.txt')
        write_log(str(self.train_paths), flags_log)
        write_log(str(self.val_paths), flags_log)
        write_log(str(self.unseen_data_path), flags_log)

        self.batImageGenTrains = BatchImageGenerator(flags=flags, file_path=self.train_paths, stage='train',
                                               b_unfold_label=False)

        self.batImageGenVals = BatchImageGenerator(flags=flags, file_path=self.val_paths, stage='val',
                                             b_unfold_label=False)

        self.batImageGenTest = BatchImageGenerator(flags=flags, file_path=[self.unseen_data_path], stage='test',
                                                   b_unfold_label=False)

    def load_state_dict(self, flags, nn):

        if flags.state_dict:

            try:
                tmp = torch.load(flags.state_dict)
                if 'state' in tmp.keys():
                    pretrained_dict = tmp['state']
                else:
                    pretrained_dict = tmp
            except:
                pretrained_dict = model_zoo.load_url(flags.state_dict)

            model_dict = nn.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}

            print('model dict keys:', len(model_dict.keys()), 'pretrained keys:', len(pretrained_dict.keys()))
            print('model dict keys:', model_dict.keys(), 'pretrained keys:', pretrained_dict.keys())
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            nn.load_state_dict(model_dict)

    def configure(self, flags):

        if torch.cuda.is_available():
            base_params = list(map(id, self.network.module.fc.parameters()))
            logits_params = filter(lambda p: id(p) not in base_params, self.network.module.parameters())
            params = [
              {"params": logits_params, "lr": flags.lr[0]},
              {"params": self.network.module.fc.parameters(), "lr": flags.lr[1]},
            ]
            self.opt_network = torch.optim.SGD(params, weight_decay=flags.weight_decay, momentum=flags.momentum)

        self.scheduler = lr_scheduler.StepLR(optimizer=self.opt_network, step_size=int(flags.step_size), gamma=0.1)

        self.loss_fn_CE = torch.nn.CrossEntropyLoss()
        self.loss_fn_MSE = torch.nn.MSELoss()
        self.loss_fn_BCE = torch.nn.BCELoss()
        
        
    def train(self, flags):
        self.network.train()
        
        self.best_accuracy_val = -1
        self.best_accuracy_test = -1

        for ite in range(flags.loops_train):

            # get the inputs and labels from the data reader
            total_loss = 0.0
            loss_C_all = 0.0
            flag = 0 #是否拼接的标志
            for index in range(flags.num_domains):
                images_train, labels_train, H, L = self.batImageGenTrains.get_images_labels_batch(index)

                inputs, labels, H, L = torch.from_numpy(
                    np.array(images_train, dtype=np.float32)), torch.from_numpy(
                    np.array(labels_train, dtype=np.float32)),torch.from_numpy(
                    np.array(H, dtype=np.float32)),torch.from_numpy(
                    np.array(L, dtype=np.float32))
                
                # wrap the inputs and labels in Variable
                if torch.cuda.is_available():
                    inputs, labels, H, L = Variable(inputs, requires_grad=False).cuda(), \
                                 Variable(labels, requires_grad=False).long().cuda(), \
                                    Variable(H, requires_grad=False).cuda(), \
                                    Variable(L, requires_grad=False).cuda()
                else:
                    inputs, labels, H, L = Variable(inputs, requires_grad=False).cpu(), \
                                 Variable(labels, requires_grad=False).long().cpu(), \
                                    Variable(H, requires_grad=False).cpu(), \
                                    Variable(L, requires_grad=False).cpu()
                print('labels ',labels)
                
                # forward with the adapted parameters
                outputs_lc, outputs_hc, outputs_L, outputs_H = self.network(x=inputs, types='L')
                                    
                print('outputs_H.shape ',outputs_H.shape)
                print('H.shape ',H.shape)
                print('outputs_lc',outputs_lc.shape)
                
                #高频频特征学习
                loss_H_mse = self.loss_fn_MSE(outputs_H, H) #仅使用交叉熵
#                 loss_H_bce = self.loss_fn_BCE(outputs_H, H) #outputs需要先经过sigmoid
#                 loss_H = loss_H_mse + loss_H_bce
                
                #高频频特征学习
                loss_L_mse = self.loss_fn_MSE(outputs_L, L) #仅使用交叉熵
#                 loss_L_bce = self.loss_fn_BCE(outputs_L, L) #outputs需要先经过sigmoid
#                 loss_L = loss_L_mse + loss_L_bce
                
                loss_C = self.loss_fn_CE(outputs_lc, labels) + self.loss_fn_CE(outputs_hc, labels)
                
                total_loss = loss_C + total_loss + loss_H_mse + loss_L_mse
                
                if flag == 0:
                    data, image_labels = inputs, labels
                    flag = 1
                else:
                    data, image_labels = \
                    torch.cat((data,inputs),0),\
                    torch.cat((image_labels,labels),0)
    
                #进行图像绘制
                if ite % 100 == 0:# and ite is not 0:
                    recon_image_H = outputs_H[0].detach().clone()
                    src_image_H = H[0].detach().clone()
                    recon_image_L = outputs_L[0].detach().clone()
                    src_image_L = L[0].detach().clone()
                    src_image = inputs[0].detach().clone()
                    
                    recon_image_H = np.array(recon_image_H.cpu())*255 #逆预处理：0-1 -> 0-255
                    src_image_H = np.array(src_image_H.cpu())*255 #逆预处理：0-1 -> 0-255
                    recon_image_L = np.array(recon_image_L.cpu())*255 #逆预处理：0-1 -> 0-255
                    src_image_L = np.array(src_image_L.cpu())*255 #逆预处理：0-1 -> 0-255
                    
                    src_image = np.array(src_image.cpu())#.permute(1,2,0),dtype=np.float32)[:,:,(2,1,0)]
                    
                    image_src = []
                    for ss, m, s in zip(src_image, self.mean, self.std):
                        
                        #源输入图像进行逆预处理
                        ss = np.array(ss * s)
                        ss = np.array(ss + m)
                        ss = ss * 255
                        
                        image_src.append(ss)
    
                    image_src = np.stack(image_src) #stack默认axis=0

                    image_src = np.array(image_src.transpose((1,2,0)),dtype=np.float32)[:,:,(2,1,0)]
                    recon_H = np.array(recon_image_H.transpose((1,2,0)),dtype=np.float32)[:,:,(2,1,0)]
                    image_H = np.array(src_image_H.transpose((1,2,0)),dtype=np.float32)[:,:,(2,1,0)]
                    recon_L = np.array(recon_image_L.transpose((1,2,0)),dtype=np.float32)[:,:,(2,1,0)]
                    image_L = np.array(src_image_L.transpose((1,2,0)),dtype=np.float32)[:,:,(2,1,0)]
                    
                    cv2.imwrite(flags.image_path + os.sep + "L_epoch_{}_{}.png".format(ite,index),recon_L)
                    cv2.imwrite(flags.image_path + os.sep + "src_L_epoch_{}_{}.png".format(ite,index),image_L)
                    cv2.imwrite(flags.image_path + os.sep + "H_epoch_{}_{}.png".format(ite,index),recon_H)
                    cv2.imwrite(flags.image_path + os.sep + "src_H_epoch_{}_{}.png".format(ite,index),image_H)
                    cv2.imwrite(flags.image_path + os.sep + "src_epoch_{}_{}.png".format(ite,index),image_src)
            
            # init the grad to zeros first
            self.opt_network.zero_grad()
            # backward your network
            total_loss.backward()
#             # optimize the parameters
#             self.opt_network.step()
    
            shuffle_index = torch.randperm(len(image_labels))
            data, image_labels = data[shuffle_index], image_labels[shuffle_index]
            
            outputs_hc, _ = self.network(x=data, types='H')
            loss_C_h = self.loss_fn_CE(outputs_hc, image_labels)
            
#             # init the grad to zeros first
#             self.opt_network.zero_grad()
            # backward your network
            loss_C_h.backward()
            # optimize the parameters
            self.opt_network.step()

            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(
                "total_loss:" + str(total_loss.item()) + "  loss_C:" +str(loss_C_h.item()),
                flags_log)    

            self.scheduler.step(epoch=ite)
            
            if ite < 500 or ite % 500 == 0:
                print(
                    'ite:', ite, 'total loss:', total_loss.cpu().item(),
                    'lr:', self.opt_network.param_groups[0]['lr'])
                
            if ite % flags.test_every == 0 and ite is not 0: 
                self.test_workflow(self.batImageGenVals, flags, ite)

    def test_workflow(self, batImageGenVals, flags, ite):

        accuracies = []
        for d_index in range(flags.num_domains):
            accuracy_val = self.test(batImageGenTest=batImageGenVals, flags=flags, ite=ite,
                                     log_dir=flags.logs, log_prefix='val_index_{}'.format(d_index), d_index=d_index)

            accuracies.append(accuracy_val)
                        
        mean_acc = np.mean(accuracies)
        
        f = open(os.path.join(flags.logs, '{}.txt'.format("val_accuracy")), mode='a')
        f.write('accuracy:{}\n'.format(mean_acc))
        f.close()
        
        if mean_acc > self.best_accuracy_val or ite > 1000:
            if mean_acc > self.best_accuracy_val:
                self.best_accuracy_val = mean_acc

            acc_test = self.test(batImageGenTest=self.batImageGenTest, flags=flags, ite=ite,
                                 log_dir=flags.logs, log_prefix='dg_test')

            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write(
                'ite:{}, best val accuracy:{}, test accuracy:{}\n'.format(ite, self.best_accuracy_val,
                                                                          acc_test))
            f.close()

            if not os.path.exists(flags.model_path):
                os.makedirs(flags.model_path)
                
            if acc_test > self.best_accuracy_test:
                self.best_accuracy_test = acc_test
                outfile = os.path.join(flags.model_path, 'best_model.tar')
                torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)

    def test(self, flags, ite, log_prefix, log_dir='logs/', batImageGenTest=None, d_index=0):

        # switch on the network test mode
        self.network.eval()

        if batImageGenTest is None:
            batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', b_unfold_label=True)

        images_test = batImageGenTest.images[d_index]
        labels_test = batImageGenTest.labels[d_index]

        threshold = 50
        if len(images_test) > threshold:

            n_slices_test = int(len(images_test) / threshold)
            indices_test = []
            for per_slice in range(n_slices_test - 1):
                indices_test.append(int(len(images_test) * (per_slice + 1) / n_slices_test))
            test_image_splits = np.split(images_test, indices_or_sections=indices_test)

            # Verify the splits are correct
            test_image_splits_2_whole = np.concatenate(test_image_splits)
            assert np.all(images_test == test_image_splits_2_whole)

            # split the test data into splits and test them one by one
            test_image_preds = []
            index = 0
            for test_image_split in test_image_splits:
                index = index + 1
                if torch.cuda.is_available():
                    images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cuda()
                else:
                    images_test = Variable(torch.from_numpy(np.array(test_image_split, dtype=np.float32))).cpu()
                tuples = self.network(images_test, 'H')
                
                predictions = tuples[1]['Predictions']
                predictions = predictions.cpu().data.numpy()
                test_image_preds.append(predictions)
                
            # concatenate the test predictions first
            predictions = np.concatenate(test_image_preds)
        else:
            if torch.cuda.is_available():
                images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32))).cuda()
            else:
                images_test = Variable(torch.from_numpy(np.array(images_test, dtype=np.float32))).cpu()

            tuples = self.network(images_test, 'H')

            predictions = tuples[1]['Predictions']
            predictions = predictions.cpu().data.numpy()

        accuracy = compute_accuracy(predictions=predictions, labels=labels_test)
        print('----------accuracy test----------:', accuracy)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        f = open(os.path.join(log_dir, '{}.txt'.format(log_prefix)), mode='a')
        f.write('ite:{}, accuracy:{}\n'.format(ite, accuracy))
        f.close()

        # switch on the network train mode
        self.network.train()

        return accuracy
