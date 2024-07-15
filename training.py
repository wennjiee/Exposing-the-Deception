import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import sys
import logging
import numpy as np
from tqdm import tqdm
from losses.mi_loss import *
from parameters import parse_args
from dataset import ReadDataset, MyDataset, mixup_data, mixup_criterion
from utils import plt_tensorboard,remove_prefix,create_table
from models.MI_Net import MI_Net
from datetime import datetime
import warnings
from csv_example import write_csv
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
import random
random.seed(2024)
warnings.simplefilter("ignore", UserWarning)
class train_and_test_model():
    def __init__(self,args):
        self.test_loss = []
        self.best_AUC =  0
        self.start_epoch = 1
        self.plt_tb = plt_tensorboard(args)
        self.args = args
        self.net = MI_Net(args=args, model=self.args.model, num_regions=self.args.num_LIBs, num_classes=2)
        self.device_ids = list(map(int, args.gpu_num.split(',')))
        self.dataset = ReadDataset(args)
        self.train_dataset = MyDataset(args, self.dataset.data['train'],self.dataset.labels['train'], size=args.size )
        self.val_dataset = MyDataset(args, self.dataset.data['val'], self.dataset.labels['val'], size=args.size , test=True)
        self.test_dataset = MyDataset(args, self.dataset.data['test'], self.dataset.labels['test'], size=args.size , test=True)
        self.best_epoch = -1
        self.device = torch.device("cuda", self.device_ids[0])
        # must be shuffled due to calculation of auc 
        try:
            self.train_loader = DataLoader(self.train_dataset, shuffle=True, batch_size=args.bs, num_workers=args.num_workers)
                                           
        except:
            print("train_dataset is null")
        try:
            self.val_loader = DataLoader(self.val_dataset, shuffle=False, batch_size=args.test_bs, num_workers=args.num_workers)
                                      
        except:
            print("val_dataset is null")
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
        max_iters = self.args.epoch * len(self.train_loader)
        # lr_scheduler aims to update 'optimizer.param_groups[n]['lr']' in optimizer
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 0.05 ** (iter / max_iters))
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150)
        
        if self.args.resume_model:
            self.load_model(path=self.args.resume_model)
        # self.update_lr()
    
    def load_model(self,path):
        logging.info(f'Resuming training from {path}')
        print(f'Resuming training from {path}')
        loaded_params = torch.load(f"{path}", map_location=torch.device(self.device))
                                   
        state_dict = loaded_params['state_dict']
        self.best_AUC = loaded_params['best_AUC']
        self.loss_function = loaded_params['loss_function']
        # self.loss_function.load_state_dict(loaded_params['loss_function'])
        try:
            self.net.load_state_dict(state_dict)
            self.optimizer.load_state_dict(loaded_params['optimizer'])
            self.scheduler.load_state_dict(loaded_params['scheduler'])
        except:
            state_dict = remove_prefix(state_dict, 'module.')
            self.net.load_state_dict(state_dict)
        self.start_epoch = loaded_params['epoch'] + 1
        logging.warning(f"Loaded parameters from saved model: current epoch is"f" {self.start_epoch}")

    def save_model(self,epoch,metric,best=False):
        if self.args.save_model:
            logging.info(f"Saving model to {self.args.save_path}/{self.args.name}.")
            saved_dict = {'state_dict': self.net.state_dict(),
                          'epoch': epoch,
                          'scheduler': self.scheduler.state_dict(),
                          'optimizer': self.optimizer.state_dict(),
                          'best_AUC': self.best_AUC,
                          'loss_function': self.loss_function,
                          'cur_metric': metric}
            if best:
                model_name=f"{self.args.save_path}/train_{self.args.name}/model_best.pth"
                logging.info(f"Saving best model_at_epoch_{epoch}")
            else:
                model_name = f"{self.args.save_path}/train_{self.args.name}/model_epoch_{epoch}.pth"
            torch.save(saved_dict, model_name)
                        
    def update_lr(self):
        if len(self.test_loss)>=5:
            test_loss=np.array(self.test_loss[-5:])
            loss_drop=test_loss[:4]-test_loss[1:5]
            if min(loss_drop)<0:
                self.args.lr=self.args.lr/2
                self.test_loss=[]
                logging.info(f"Update lr to {self.args.lr}")
        self.optimizer = torch.optim.Adam([
            {'params': self.net.parameters(), 'lr': self.args.lr, 'weight_decay': args.weight_decay, 'betas': (0.9, 0.999)},
            {'params': self.loss_function.balance_loss.parameters(), 'weight_decay': args.weight_decay}
        ])
        # lr_scheduler更新optimizer的optimizer.param_groups[n]['lr']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=150)

    def train(self):
        # epbar = tqdm(total=self.args.epoch)
        # self.test(self.net, self.start_epoch-1)
        for epoch in range(self.start_epoch, self.args.epoch + 1):
            info = '['+f"-"*15 + f"Starting Training at Epoch {epoch}" + "-"*15+']'
            logging.info(info)
            print(info)
            # self.update_lr()
            self.net.train()
            avg_loss = []
            avg_ce_loss = []
            avg_global_mi_loss = []
            avg_local_loss = []
            self.plt_tb.reset_metrics()
            # loader_pbar = tqdm(loader, position=1)
            total_iterators = len(self.train_loader)
            pbar = tqdm(total = total_iterators)
            for i, (data,y) in enumerate(self.train_loader):
                
                data = data.cuda(self.device)
                y = y.type(torch.int64).cuda(self.device)

                if self.args.mixup:
                    data, y_a, y_b, lam = mixup_data(data, y, self.args.alpha) # mixed_x, y_a, y_b, lam
                    out = self.net(data)
                    losses = mixup_criterion(self.loss_function.criterion, out, y_a, y_b, lam)
                else:
                    out = self.net(data)
                    losses = self.loss_function.criterion(out, y)
                
                loss = self.loss_function.balance_mult_loss(losses)

                if torch.isnan(loss).any():
                    logging.info("loss is NAN, so stop training...")
                    sys.exit()
                
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                avg_loss.append(loss.item())
                avg_ce_loss.append(losses[0].item())
                if self.args.gil_loss:
                    avg_global_mi_loss.append(losses[1].item())
                if self.args.lil_loss:
                    avg_local_loss.append(losses[-1].item())

                self.plt_tb.accumulate_metrics(out['p_y_given_z'], y, loss)

                if i % 10 == 0:
                    log_info=f"Training total loss: {np.mean(avg_loss)}, CE loss: {np.mean(avg_ce_loss)}, "
                    if self.args.gil_loss:
                        log_info+=f"global MI loss: {np.mean(avg_global_mi_loss)},"
                    if self.args.lil_loss:
                        log_info += f"local MI loss: {np.mean(avg_local_loss)}"
                    logging.info(log_info)
                    # maybe something strange occur
                    # self.plt_tb.write_tensorboard((epoch-1)*total_iterators + i, tb_writer=self.plt_tb.tb_writer, tb_prefix=f'Test_False')
                # self.plt_tb.write_tensorboard(epoch, tb_writer=self.plt_tb.tb_writer, tb_prefix=f'Metrics in Training')  
                pbar.update(1)
            
            pbar.close()
            self.plt_tb.write_tensorboard(epoch, tb_writer=self.plt_tb.tb_writer, tb_prefix=f'Metrics in Training')                                                
            logging.info(f"Finish Epoch: {epoch} Training Average loss: {np.mean(avg_loss)}")
            
            self.test(self.net, epoch, val=False)

    def test(self, net, epoch, val=False):
        epoch = epoch if epoch != 0 else 'best'
        info = '['+f"-"*15 + f"Starting Testing at Epoch {epoch}" + "-"*15+']'
        print(info)
        net.eval()
        self.plt_tb.reset_metrics()
        avg_total_loss = []
        avg_ce_loss = []
        avg_global_mi_loss = []
        avg_local_loss = []
        info = 'Starting Testing At Best Model'
        logging.warning(info)
        infer_results = []
        infer_prob = []
        with torch.no_grad():
            if val:
                loader = self.val_loader
            else:
                loader = self.test_loader
            pbar = tqdm(total=len(loader))
            for i, (data,y) in enumerate(loader):
                data = data.cuda(self.device)
                y = y.type(torch.int64).cuda(self.device)
                out = net(data)
                losses = self.loss_function.criterion(out, y)
                loss = self.loss_function.balance_mult_loss(losses)

                avg_total_loss.append(loss.item())
                avg_ce_loss.append(losses[0].item())
                
                if self.args.gil_loss:
                    avg_global_mi_loss.append(losses[1].item())
                if self.args.lil_loss:
                    avg_local_loss.append(losses[-1].item())
                
                self.plt_tb.accumulate_metrics(out['p_y_given_z'], y, loss)

                outputs = out['p_y_given_z']
                _, pred = outputs.topk(1, 1, True, True)
                pred = pred.t()
                infer_results.append(list(pred.view(-1).cpu().numpy()))
                outputs = outputs.cpu().detach()
                prob = softmax(np.array(outputs.data.tolist()), axis=1)[:,1]
                infer_prob.append(prob)
                
                pbar.update(1)

            pbar.close()    
        self.test_loss.append(np.mean(avg_total_loss))
        
        # it's not necessary to plot during test
        if not args.test:
            # metric = self.plt_tb.report_test_metrics(), metric could be fluctuating for a random batch at each epoch
            metric, acc_socre = self.plt_tb.report_metrics(epoch, tb_writer=self.plt_tb.tb_writer, tb_prefix=f'Metrics in Testing')
            # report it in testing
            # metric, acc_socre = self.plt_tb.report_for_cifar(epoch, tb_writer=self.plt_tb.tb_writer, tb_prefix=f'Test_True')
        else:
            metric, acc_socre = self.plt_tb.report_test_metrics()
            test_path = self.test_dataset.data
            label_list = self.test_dataset.label
            pred_list = [y for x in infer_results for y in x]
            # pred_list = [y for x in self.plt_tb.metrics[0].pred_results for y in x]
            sum_of_diff = sum(abs(np.array(pred_list) - np.array(label_list)))
            test_acc = 1 - sum_of_diff/len(pred_list)
            test_auc = roc_auc_score(label_list, np.concatenate(infer_prob))
            print(f'Tested AUC is: {test_auc}, ACC is: {test_acc}')
            # 实验发现 对 infer_prob(为1的概率)设置阈值为0.5时 基于ROC曲线计算的ACC结果与作者代码中直接取output最大值的索引为分类标签一致
            tmp = np.concatenate(infer_prob)
            ip = []
            for i in range(len(tmp)):
                a = 1 if tmp[i]>=0.5 else 0
                ip.append(a)
            diff = sum(abs(np.array(ip) - np.array(label_list)))
            print(f'ACC is: {1 - diff/len(ip)}')

            lines = []
            for idx in range(len(test_path)):
                line = []
                line.append(test_path[idx])
                line.append('label=' + str(label_list[idx]))
                line.append('pred=' + str(pred_list[idx]))
                lines.append(line)
            write_csv(f'./logs/test_{args.dataset}_reults.csv', lines)

            # lines = []
            # for idx in range(len(test_path)):
            #     line = []
            #     line.append(test_path[idx])
            #     line.append('label=' + str(label_list[idx]))
            #     line.append('pred=' + str(pred_list[idx]))
            #     lines.append(line)
            # write_csv(f'./logs/test_{args.dataset}_label_reults.csv', lines)
            
            # lines = [['img_name', 'pred']]
            # _infer_prob = np.concatenate(infer_prob)
            # for idx in range(len(test_path)):
            #     line = []
            #     line.append(test_path[idx].split('/')[-1])
            #     line.append(str(_infer_prob[idx]))
            #     lines.append(line)
            # write_csv(f'./logs/test_{args.dataset}_prob_reults.csv', lines)
        log_info = f"Tested AUC: {metric}, ACC: {acc_socre}, Training CE loss: {np.mean(avg_ce_loss)}, "
        if self.args.gil_loss:
            log_info += f"global MI loss: {np.mean(avg_global_mi_loss)},"
        if self.args.lil_loss:
            log_info += f"local MI loss: {np.mean(avg_local_loss)}"
        logging.info(log_info)

        if not args.test and 'cifar' not in args.dataset:
            if self.best_AUC<=metric:
                self.best_AUC=metric
                self.save_model(epoch,metric,best=True)
            self.save_model(epoch,metric,best=False)
            print('Saved Model!')
            return metric


def unNormalize(tensor,mean,std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    return tensor

if __name__ == "__main__":
    args = parse_args()
    print(args)
    if args.extract_face:
        import multiprocessing
        multiprocessing.set_start_method('spawn')
    mode = 'train' if not args.test else 'test'
    if args.name and mode == 'train':
        format_current = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        os.makedirs(f"output/train_{args.name}", exist_ok=True)
    logging.basicConfig(filename=f"./logs/{mode}_{args.name}.log",
                        filemode="a",
                        format='[%(asctime)s]%(levelname)s:%(message)s',
                        datefmt='%Y.%m.%d %I:%M:%S %p',
                        level=logging.INFO, )
    logging.warning(create_table(args))

    train_model = train_and_test_model(args)

    if mode == 'train':
        train_model.train()
        print('Finished Training And Starting Testing Best Model')
        args.test = True
        train_model.load_model(os.path.join(args.save_path, f'train_{args.name}', 'model_best.pth'))
        train_model.test(train_model.net, 0, val=False)
    else:
        print('resume_model:', args.resume_model)
        print('datasets:', args.dataset)
        train_model.test(train_model.net, 0, val=False)
