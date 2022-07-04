import argparse

from model_resnet_all import ModelAggregate   #change


def main():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--test_every", type=int, default=100,
                                  help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=8,
                                  help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=7,
                                  help="")
    train_arg_parser.add_argument("--num_domains", type=int, default=3,
                                  help="")
    train_arg_parser.add_argument("--step_size", type=int, default=1001,
                                  help="")
    train_arg_parser.add_argument("--loops_train", type=int, default=5001,
                                  help="")
    train_arg_parser.add_argument("--unseen_index", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--lr", type=float, default=[0.001, 0.01], #特征提取器，低频，高频，分类器
                                  help='')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.0001,
                                  help='')
    train_arg_parser.add_argument("--momentum", type=float, default=0.9,
                                  help='')
    train_arg_parser.add_argument("--logs", type=str, default='',
                                  help='日志目录')
    train_arg_parser.add_argument("--model_path", type=str, default='', 
                                  help='保存模型地址的目录')
    train_arg_parser.add_argument("--state_dict", type=str, default='',
                                  help='起始模型的地址')
    train_arg_parser.add_argument("--data_root", type=str, default='/data2/wangjingye/DG/datasets/PACS_DataSet/Train_val_splits_and_h5py_files_pre-read',
                                  help='数据集的目录')
    train_arg_parser.add_argument("--image_path",type=str,default = "decoder_112_H_L",
                                  help='保存decoder生成的图片地址')
    train_arg_parser.add_argument("--threshold",type=int, default=25,
                                  help='保存decoder生成的图片地址')
    
    args = train_arg_parser.parse_args()
    
    index = [3,1,2,0]
    styles = ['art','cartoon','photo','sketch',]
    for x in range(10):
        for i in index:
            args.unseen_index = i
            args.logs = 'logs_PoolFPN/{}/attack_{}'.format(str(x),styles[i])
            args.model_path = 'logs_PoolFPN/{}/attack_{}_model'.format(str(x),styles[i])
            model_obj = ModelAggregate(flags=args)
            model_obj.train(flags=args)


if __name__ == "__main__":
    main()
