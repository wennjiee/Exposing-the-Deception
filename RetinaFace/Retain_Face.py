from __future__ import print_function
import os
import argparse
from RetinaFace.generate_bbox_single_img import load_model
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from RetinaFace.data import cfg_mnet, cfg_re50
from RetinaFace.layers.functions.prior_box import PriorBox
from RetinaFace.utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from RetinaFace.models.retinaface import RetinaFace
from RetinaFace.utils.box_utils import decode, decode_landm
from RetinaFace.utils.timer import Timer

class Retain_Face():
    

    def __init__(self):
        # 参数初始化
        parser = argparse.ArgumentParser(description='Retinaface')

        parser.add_argument('-m', '--trained_model', default='./RetinaFace/weights/mobilenet0.25_Final.pth',
                            type=str, help='Trained state_dict file path to open')
        parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
        parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
        parser.add_argument('--save_folder', default='./kaggleData/waitan2024_deepfake_challenge/phase1/bboxTxt/', type=str, help='Dir to save txt results')
        parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
        parser.add_argument('--dataset_folder', default='./kaggleData/waitan2024_deepfake_challenge/phase1/valset/', type=str, help='dataset path')
        parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
        parser.add_argument('--top_k', default=5000, type=int, help='top_k')
        parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
        parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
        parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
        parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
        self.args = parser.parse_args()

        # torch.set_grad_enabled(False)
        # 模型初始化
        self.cfg = cfg_mnet
        self._t = {'forward_pass': Timer(), 'misc': Timer()}
        # net and model
        net = RetinaFace(cfg=self.cfg, phase = 'test')
        net = load_model(net, self.args.trained_model, self.args.cpu)
        net.eval()
        # print('Finished loading model!')
        cudnn.benchmark = True
        self.device = torch.device("cpu" if self.args.cpu else "cuda")
        self.net = net.to(self.device)

    
    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # print('Missing keys:{}'.format(len(missing_keys)))
        # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        # print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        # print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}
    
    def generateBooxImg(self, img_raw):

        img = np.float32(img_raw)
        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        # 根据输入图像尺寸确定是否需要resize处置
        if self.args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        self._t['forward_pass'].tic()
        # 模型推理生成人脸检测框
        loc, conf, landms = self.net(img)  # forward pass
        self._t['forward_pass'].toc()
        self._t['misc'].tic()
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        
        max_matrix = np.zeros(5, dtype=np.int32)
        temp = 0
        self._t['misc'].toc()

        for b in dets:
                if b[4] < self.args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                if(float(text) > temp):
                    temp = float(text)
                    max_matrix[0] = int(b[0])
                    max_matrix[1] = int(b[1])
                    max_matrix[2] = int(b[2])
                    max_matrix[3] = int(b[3])
        left = max_matrix[1] - 30 if max_matrix[1] >= 30 else 0
        right = max_matrix[0] - 30 if max_matrix[0] >= 30 else 0
        img_crop = img_raw[left:max_matrix[3] + 30, right:max_matrix[2] + 30]
        return img_crop
    
if __name__ == '__main__':
    img_raw = cv2.imread("img/1.jpg", cv2.IMREAD_COLOR)
    face = Retain_Face()
    crop_img = face.generateBooxImg(img_raw)
    print(type(crop_img))
    print(crop_img.dtype)
    print(crop_img)