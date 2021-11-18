
import os
import glob
import torch
import imgaug
import numpy as np

# --------------- Setting Random Seeds ------------------ #
def set_deterministic_training():
    os.environ['PYTHONHASHSEED']=str(42)
    os.environ["PL_GLOBAL_SEED"] = str(42) 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':16:8'         
    # Added above due to torch.set_deterministic(True) 
    # Ref: https://github.com/pytorch/pytorch/issues/47672#issuecomment-725404192

    np.random.seed(42)
    imgaug.seed(42)         
    # Although, imgaug seed and torch seed are set but internally when torch will be using multi threads and 
    # We might not be having control over which thread will call imgaug augmenter with which img sequence.
    # e.g. For exp-1, img-1, img-2, img-3 will be provided by thread-1, thread-3, thread-2 respectively.
    #      In exp-2, img-1, img-2, img-3 might be provided by thread-3, thread-2, thread-1 respectively.
    # And imgaug will provide augmentations to these img in same sequence.
    # E.g. In exp-1, img-1, img-2, img-3 are provided to imgaug module in sequence 1, 3, 2, then
    #      img-1 will face augmentation-1, img-3 will face augmentation-2 and img-2 --> augmentation-3
    #      In exp-2, img-1, img-2, img-3 are provided to imgaug module in sequence 3, 2, 1, then
    #      img-3 will face augmentation-1, img-2 will face augmentation-2 and img-1 --> augmentation-3
    # So complete control over randomness is not achieved due to irregularities/randomness between imgaug and pytorch dataloader

    torch.set_deterministic(True)       # This will set deterministic behaviour for cuda operations
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)      # sets random seed for cuda for all gpus

    torch.backends.cudnn.benchmark = False      # Cuda will not try to find the best possible algorithm implementations, performance might degrade due to this being set to False 
    torch.backends.cudnn.deterministic = True     # cuda will use only deterministic implementations

    # pytorch reproducibility
    # Ref: https://stackoverflow.com/q/56354461         
    # Ref: https://learnopencv.com/ensuring-training-reproducibility-in-pytorch/
    return True

is_deterministic = False
# is_deterministic = set_deterministic_training()
# Disabling deterministic training due to only non-deterministic operations
# available for the backpropagation of nn.functional.interpolate method.
# Ref. : https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
if not is_deterministic:
    print("WARNING: Deterministic training is disabled.")
# -------------------------------------------------------- #


# ----------------------- Dataset ------------------------ #
# train_csv_path  = "./dataset/ann_files/voc_train_ann.csv"
train_csv_path  = "./dataset/ann_files/voc_test_ann.csv"
print("\n\nWARNING: Test set is being used as training set for debugging purpose\n\n")
test_csv_path   = "./dataset/ann_files/voc_test_ann.csv"
label_txt       = "./dataset/voc_labels.txt"
label_dict      = {}
label_dict[0]   = 'background'                      # background class at index 0
with open(label_txt, 'r') as f: 
    for i, line in enumerate(f.readlines(), 1):
        label_dict[i] = line.replace('\n', '')

num_classes         = len(label_dict)
normalization_type  = 'imagenet'        # ['imagenet', 'div_255']

with open(train_csv_path, 'r') as f:
    train_len = len(f.readlines())
    
with open(test_csv_path, 'r') as f:
    test_len = len(f.readlines())
# ------------------------------------------------------- #


# ------------------- Training Routine ------------------ #
use_amp 				= False				 # AMP will give reduced memory footprint and reduced computation time for GPU having Tensor Cores 
backbone                = 'resnet18'         # ['resnet50', 'resnet18']
# backbone                = 'resnet50'         # ['resnet50', 'resnet18']
backbone_ckpt           = './pretrained_ckpt/resnet18-5c106cde.pth'
# backbone_ckpt           = './pretrained_ckpt/resnet50-19c8e357.pth'
freeze_backend          = [False, False, False, False]      
# To freeze features upto P2, P3, P4, P5 level in backbone, 4 boolean values are provided.  
fpn_features            = 256
use_cntr_sampling       = True
use_det_head_group_norm = True
centerness_on_regression= True
central_sampling        = True
if central_sampling:
    central_sampling_radius = 1.5
# We would sample gt bboxes for each point such that the bbox center lies within the 
# 1.5 times the stride than we will consider that bbox as label for further computation.
regression_loss_type    = 'giou'                # ['giou', 'iou']    

use_gradient_checkpointing  = True
# True will checkpoint provided tensors and discard rest of the tensors.
# This will help in reducing the memory footprint.
# Ref. : https://spell.ml/blog/gradient-checkpointing-pytorch-YGypLBAAACEAefHs

batch_size              = 24
epochs                  = 30
overfit_epochs          = 1000      # Will be used only in case of overfit training
# input_size              = [800, 1024]            # [H x W]
input_size              = [400, 512]            # [H x W]
multi_scale_training    = False
if multi_scale_training:
    multiscale_step         = 10
    multiscale_input_sizes  = [[800, 1333], [400, 666]]
else:
    multiscale_step         = 1
    multiscale_input_sizes  = [input_size]
strides                 = [8, 16, 32, 64, 128]      # Strides for P3, P4, P5, P6, P7 feature-map
limit_range             = [[0, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]    
# If max(l, r, t, b) falls in the given range for a given feature map then it is considered
# as positive or else it will be negative. So bbox regression wont happen for such points.
# e.g. For P3 feature map range is [0, 64]. So all points for which max(l, r, t, b) is within
#      this range, only those will be considered for regression loss calculation.
 
l2_weight_decay         = 0.00005
weight_init_method      = 'msra'        # ['msra', 'xavier_normal']       
# 'msra' also known as variance scaling initializer and Kaiming He (normal dist) initialization 
# He initialization works better for layers with ReLu activation.
# Xavier initialization works better for layers with sigmoid activation.
# Ref: https://stats.stackexchange.com/a/319849

exp_path = "./summaries/"
os.makedirs(exp_path, exist_ok=True)

train_steps = int(np.ceil(train_len / batch_size))
test_steps  = int(np.ceil(test_len / batch_size))
loss_logging_frequency  = (train_steps // 100) if 0 < (train_steps // 100) < 100 else 1 if (train_steps // 100) == 0 else 100           # At every 100 steps or num_training_steps/3 steps training loss will be printed and summary will be saved.
# ------------------------------------------------------- #


# --------------- Learning Rate --------------- #
warm_up         = True
warm_up_eps     = 2
init_lr         = 0.001
lr_scheduler    = 'cosine_annealing'    # ['exp_decay', 'cosine_annealing']
lr_exp_decay    = 0.94                  # Only for 'burn_in_decay'. Set this such that at the end of training (= after "epochs" number of iterations), the lr will be of scale 1e-6 or 1e-7.
steps_per_epoch = train_steps
burn_in_steps   = steps_per_epoch * warm_up_eps
# --------------------------------------------- #                    
