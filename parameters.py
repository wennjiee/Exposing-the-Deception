import argparse
def parse_args():

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("--name", default="all_df_adv_fas_withex", type=str, help="Specify name of the model")
    parser.add_argument("--gpu_num", default="0", type=str, help="gpu number")
                        

    # arguments for train
    parser.add_argument('--model', default="resnet34", type=str, help="choose backbone model")
    parser.add_argument("--epoch", default=15, type=int, help="epoch of training")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay of training")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate of training")
    parser.add_argument("--bs", default=24, type=int, help="batch size of training") # 128 when without ex, 48 with ex
    parser.add_argument("--test_bs", default=24, type=int, help="batch size of training")
    parser.add_argument("--num_workers", default=8, type=int, help="num workers")

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
                        default="output/train_all_df_adv_fas_withex/model_epoch_10.pth", # output/train_deepfake_real_kaggle_with_extract/model_best. output/train_race/model_best.pth
                        type=str,
                        help="Path of resume model")

    # arguments for test
    parser.add_argument("--test", default=False, type=bool,help="Test or not")

    # save models
    parser.add_argument("--save_model", default=True, type=bool, help="whether save models or not")
    parser.add_argument("--save_path", default="output", type=str, help="Path of test file, work when test is true")
    
    # dataset
    parser.add_argument("--size", default=224, type=int, help="Specify the size of the input image, applied to width and height")
    parser.add_argument('--dataset', default="all", type=str, help="dataset txt path")
    # 'Face2Face','Deepfakes','FaceSwap','NeuralTextures', Celeb-DF-v2, DFDC-Preview, DFDC, FF++_c23, DeeperForensics-1.0, cifar-10-batches-py
    parser.add_argument("--mixup", default=True, type=bool, help="mix up or not")
    parser.add_argument("--alpha", default=0.5, type=float, help="mix up alpha")
    parser.add_argument("--extract_face", default=True, type=bool, help="whether to extract face from img")
    return parser.parse_args()

