import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from losses.mi_loss import *
from dataset import InferDataset, MyDataset, mixup_data, mixup_criterion
from utils import plt_tensorboard,remove_prefix,create_table
from models.MI_Net import MI_Net
import warnings
from csv_example import write_csv
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import random
random.seed(2024)
warnings.simplefilter("ignore", UserWarning)
import argparse
parser = argparse.ArgumentParser()
from RetinaFace.models.retinaface import RetinaFace
from RetinaFace.data import cfg_mnet
from RetinaFace.Retain_Face import Retain_Face

# general arguments
parser.add_argument("--gpu_num", default="0", type=str, help="gpu number")
                    
# arguments for train or test
parser.add_argument('--model', default="resnet34", type=str, help="choose backbone model")
parser.add_argument("--epoch", default=20, type=int, help="epoch of training")
parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay of training")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate of training")
parser.add_argument("--test_bs", default=32, type=int, help="batch size of training")
parser.add_argument("--num_workers", default=0, type=int, help="num workers")

# arguments for loss
parser.add_argument("--lil_loss", default=True, type=bool, help="if local information loss")
parser.add_argument("--gil_loss", default=True, type=bool, help="if global information loss")
parser.add_argument('--temperature', type=float, default=1.5, help="the temperature used in knowledge distillation")
parser.add_argument('--mi_calculator', default="kl", type=str, help="mutual information calculation method")
parser.add_argument('--scales', default=[1,2,10], type=list, help="multiple losses weights")
parser.add_argument('--balance_loss_method', default='auto', type=str, help="balance multiple losses method")

# model parameters
parser.add_argument("--num_LIBs", default=4, type=int, help="the number of Local Information Block")
parser.add_argument("--resume_model",
                    default="output/train_deepfake_real_kaggle_with_extract/model_best.pth", # output/train_celeb_df_v2/model_epoch_9.pth
                    type=str,
                    help="Path of resume model")

# arguments for test
parser.add_argument("--test", default=True, type=bool,help="Test or not")

# dataset
parser.add_argument("--size", default=224, type=int, help="Specify the size of the input image, applied to width and height")
parser.add_argument('--dataset', default="deepfake", type=str, help="dataset txt path")
# 'Face2Face','Deepfakes','FaceSwap','NeuralTextures', Celeb-DF-v2, DFDC-Preview, DFDC, FF++_c23, DeeperForensics-1.0, cifar-10-batches-py
parser.add_argument("--mixup", default=True, type=bool, help="mix up or not")
parser.add_argument("--alpha", default=0.5, type=float, help="mix up alpha")
parser.add_argument("--data_path", default='./datasets/fake/fake_video_extract', type=str, help="path to inference file")
parser.add_argument("--extract_face", default=False, type=bool, help="whether to extract face from img")

args = parser.parse_args()

class inference_model():
    def __init__(self,args):
        self.infer_results = []
        self.infer_prob = []
        self.best_AUC =  0
        self.start_epoch = 1
        self.plt_tb = plt_tensorboard(args)
        self.args = args
        self.net = MI_Net(args=args, model=self.args.model, num_regions=self.args.num_LIBs, num_classes=2)
        self.device_ids = list(map(int, args.gpu_num.split(',')))
        self.dataset = InferDataset(args)
        self.test_dataset = MyDataset(args, self.dataset.data['test'], self.dataset.labels['test'], size=args.size , test=True)
        self.device = torch.device("cuda", self.device_ids[0])
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, batch_size=args.test_bs, num_workers=args.num_workers)              
        self.loss_function = loss_functions(method='mi',
                                            mi_calculator=self.args.mi_calculator, temperature=self.args.temperature,
                                            bml_method=self.args.balance_loss_method, scales=self.args.scales,
                                            lil_loss=self.args.lil_loss,
                                            gil_loss=self.args.gil_loss,
                                            device=self.device)
        if len(self.device_ids) > 1:  # 单机多卡
            self.net = nn.DataParallel(self.net, device_ids=self.device_ids)
        self.net = self.net.cuda(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.net.parameters(), 'lr': self.args.lr, 'weight_decay': args.weight_decay, 'betas': (0.9, 0.999)},
            {'params': self.loss_function.balance_loss.parameters(), 'weight_decay': args.weight_decay}
            ])
        max_iters = self.args.epoch * len(self.test_loader)
        # lr_scheduler aims to update 'optimizer.param_groups[n]['lr']' in optimizer
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 0.05 ** (iter / max_iters))
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150)
        if self.args.resume_model:
            self.load_model(path=self.args.resume_model)
        # load faceDetect model
        cfg_detect = cfg_mnet
        detect_model = RetinaFace(cfg=cfg_detect, phase = 'test').to(self.device)
        self.detect_model = self.load_detect_model(detect_model,"./RetinaFace/weights/mobilenet0.25_Final.pth", False)
        self.detect_model.eval()
        
        self.retainFace = Retain_Face()
    def load_model(self,path):
        loaded_params = torch.load(f"{path}", map_location = torch.device(self.device))               
        state_dict = loaded_params['state_dict']
        self.best_AUC = loaded_params['best_AUC']
        self.loss_function = loaded_params['loss_function']
        try:
            self.net.load_state_dict(state_dict)
            self.optimizer.load_state_dict(loaded_params['optimizer'])
            self.scheduler.load_state_dict(loaded_params['scheduler'])
        except:
            state_dict = remove_prefix(state_dict, 'module.')
            self.net.load_state_dict(state_dict)
        self.start_epoch = loaded_params['epoch'] + 1
    
    def load_detect_model(self,model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
    
    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def test(self, net, epoch, val=False):
        epoch = epoch if epoch != 0 else 'best'
        info = '['+f"-"*15 + f"Starting Testing at Epoch {epoch}" + "-"*15+']'
        print(info)
        net.eval()
        self.plt_tb.reset_metrics()
        with torch.no_grad():
            loader = self.test_loader
            pbar = tqdm(total=len(loader))
            for i, (data,y) in enumerate(loader):
                data = data.cuda(self.device)
                # 加载检测模型获取检测结果
                
                with torch.no_grad():
                    loc, conf, landms = self.detect_model(data)
                # 根据检测结果处理
                data = self.retainFace.generateBooxImgBatchWihImg(data, loc, conf, landms)
                out = net(data)
                outputs = out['p_y_given_z']
                _, pred = outputs.topk(1, 1, True, True)
                pred = pred.t()
                self.infer_results.append(list(pred.view(-1).cpu().numpy()))

                outputs = outputs.cpu().detach()
                prob = softmax(np.array(outputs.data.tolist()), axis=1)[:,1]
                self.infer_prob.append(prob)
                
                pbar.update(1)
            pbar.close()    
        test_path = self.test_dataset.data
        pred_list = [y for x in self.infer_results for y in x]
        label_list = self.test_dataset.label
        if label_list[0] != 'unk':
            _infer_prob = np.concatenate(self.infer_prob)
            sum_of_diff = sum(abs(np.array(pred_list) - np.array(label_list)))
            test_acc = 1 - sum_of_diff/len(pred_list)
            test_auc = roc_auc_score(label_list, _infer_prob)
            print(f'Tested ACC is: {test_acc}')
            print(f'Test AUC is: {test_auc}')
        lines = []
        for idx in range(len(test_path)):
            line = []
            line.append(test_path[idx])
            if label_list[idx] != 'unk':
                line.append('label=' + str(label_list[idx]))
            line.append(str(pred_list[idx]))
            lines.append(line)
        postfix = args.data_path.split('/')[-1]
        write_csv(f'./results18_{postfix}.csv', lines)
        print('Finished Inference')

if __name__ == "__main__":
    if args.extract_face:
        import multiprocessing
        multiprocessing.set_start_method('spawn')
    model = inference_model(args) 
    print(args)
    print('Starting Testing')
    model.test(model.net, 0, val=False)
