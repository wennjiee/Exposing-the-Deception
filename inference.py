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
from tkinter import _flatten
import random
random.seed(2024)
warnings.simplefilter("ignore", UserWarning)
import argparse
parser = argparse.ArgumentParser()

# general arguments
parser.add_argument("--name", default="race", type=str, help="Specify name of the model")
parser.add_argument("--gpu_num", default="0", type=str, help="gpu number")
                    
# arguments for train
parser.add_argument('--model', default="resnet34", type=str, help="choose backbone model")
parser.add_argument("--epoch", default=20, type=int, help="epoch of training")
parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay of training")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate of training")
parser.add_argument("--bs", default=32, type=int, help="batch size of training")
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
                    default="output/train_celeb_df_v2/model_best.pth", # output/train_celeb_df_v2/model_epoch_9.pth
                    type=str,
                    help="Path of resume model")

# arguments for test
parser.add_argument("--test", default=True, type=bool,help="Test or not")

# save models
parser.add_argument("--save_model", default=True, type=bool, help="whether save models or not")
parser.add_argument("--save_path", default="output", type=str, help="Path of test file, work when test is true")

# dataset
parser.add_argument("--size", default=224, type=int, help="Specify the size of the input image, applied to width and height")
parser.add_argument('--dataset', default="race", type=str, help="dataset txt path")
# 'Face2Face','Deepfakes','FaceSwap','NeuralTextures', Celeb-DF-v2, DFDC-Preview, DFDC, FF++_c23, DeeperForensics-1.0, cifar-10-batches-py
parser.add_argument("--mixup", default=True, type=bool, help="mix up or not")
parser.add_argument("--alpha", default=0.5, type=float, help="mix up alpha")

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
        self.test_dataset = MyDataset(args.dataset, self.dataset.data['test'], self.dataset.labels['test'], size=args.size , test=True)
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
            line.append('pred=' + str(pred_list[idx]))
            lines.append(line)
        write_csv(f'./results18.csv', lines)
        
        print('Finished Inference')

if __name__ == "__main__":

    model = inference_model(args)
    
    print('Starting Testing')
    print('resume_model:', args.resume_model)
    print('datasets:', args.dataset)
    model.test(model.net, 0, val=False)
