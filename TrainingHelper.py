
import os
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import config
from LRHelper import LRHelper
from DatasetHelper import get_train_loader, get_test_loader

from models.FCOS import FCOS
from models.FCOSLoss import FCOSLoss

from utils.Logger import Logger
from utils.CheckpointHandler import CheckpointHandler
from utils.SummaryHelper import SummaryHelper
from utils.misc import init_training, np_cpu, LossAverager

cuda = torch.device('cuda:0')
cpu = torch.device("cpu:0")


class TrainingHelper:
    def __init__(self):
        self.log, self.exp_path = init_training()
        self.lr_helper = LRHelper()

        ckpt_folder = self.exp_path + "/ckpt/"
        os.makedirs(ckpt_folder, exist_ok=True)
        
        ckpt_path = ckpt_folder + "fcos_resnet50.pth"
        self.ckpt_handler = CheckpointHandler(ckpt_path, max_to_keep=3)


    def compute_loss_and_map(self, predictions, batch_bboxes, model, fcos_loss_fn):
        # cls_probs_pred, cnt_logits_pred, reg_values_pred = prediction
        # cls_probs_pred  : [[B x 81 x H x W], [B x 81 x H x W], ...]
        # cnt_logits_pred : [[B x 1 x H x W], [B x 1 x H x W], ...]
        # reg_values_pred : [[B x 4 x H x W], [B x 4 x H x W], ...]
        
        # batch_bbox : [B x M x 5] --> [x1, y1, x2, y2, cls_id]
        
        loss_reg = torch.tensor(0, dtype=torch.float32, device=cuda, requires_grad=False)
        for layer in model.modules():
            if isinstance(layer,torch.nn.Conv2d):
                for p in layer.named_parameters():
                    if 'weight' in p[0]:
                        loss_reg += torch.sum((torch.square(p[1]) / 2))   

        loss_reg *= config.l2_weight_decay

        loss_loc, loss_cls, loss_cnt = fcos_loss_fn(predictions, batch_bboxes)  

        loss_total = loss_loc + loss_cls + loss_cnt + loss_reg

        # sm_outputs = F.softmax(logits.detach(), dim=-1)
        # accuracy = (torch.argmax(sm_outputs, dim=1) == labels).sum() * 100 / labels.size(0)

        return loss_total, loss_loc, loss_cls, loss_cnt, loss_reg


    def get_model(self):
        model = FCOS(backbone_model=config.backbone, freeze_backend=config.freeze_backend, \
                    fpn_features=config.fpn_features, num_classes=config.num_classes, \
                    use_det_head_group_norm=config.use_det_head_group_norm, \
                    centerness_on_regression=config.centerness_on_regression, \
                    use_gradient_checkpointing=config.use_gradient_checkpointing, \
                    weight_init_method=config.weight_init_method).to(cuda, non_blocking=True)
    
        return model


    def define_loss(self):
        fcos_loss = FCOSLoss(limit_range=config.limit_range, central_sampling=config.central_sampling, \
                        central_sampling_radius=config.central_sampling_radius, strides=config.strides, \
                        regression_loss_type=config.regression_loss_type, num_classes=config.num_classes)
        
        return fcos_loss
    

    def train(self, resume=False, resume_ckpt=None, pretrained_ckpt=None):
        model = self.get_model()
        
        model_stats = summary(model, (1, 3, config.input_size[0], config.input_size[1]))
        
        for line in str(model_stats).split('\n'):
            self.log(line)
        
        fcos_loss_fn = self.define_loss()
        
        opt = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.0)
        # Setting lr equal to 0.0 here so that it wont work as per this line.
        # But we will explicitly set lr for each weights dynamically, at every step.
        # Same is case for weight_decay, We will calculate L2_regularization_loss on our own separately.
        # L2 loss will only be calculated for conv-weights only.
        
        if config.use_amp:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        if resume:
            checkpoint = torch.load(resume_ckpt)
            model.load_state_dict(checkpoint['model'])
            opt.load_state_dict(checkpoint['optimizer'])
            if config.use_amp:
                scaler.load_state_dict(checkpoint['scalar'])
            resume_g_step = checkpoint['global_step']
            resume_eps = checkpoint['epoch']
            self.log("Resuming training from {} epochs.".format(resume_eps))
        elif pretrained_ckpt is not None and config.model_type == 'resnet18':
            self.log("Using pre-trained checkpoint from :".format(pretrained_ckpt))
            checkpoint = torch.load(pretrained_ckpt)
            
            filtered_checkpoint = {}
            self.log("\nFollowing variables will be restored:")
            for var_name, var_value in checkpoint.items():
                if var_name == 'fc.weight' or var_name == 'fc.bias':        
                    # As these layers change arent there in pretrained model definition
                    continue
                if config.backbone == 'resnet':
                    new_var_name = 'resnet_feat.' + var_name                
                    # why this prefix? This comes as the model that we created contains a variable resnet_feat 
                    # which is sequential group of layers containing resnet layers. So all the layers and parameters 
                    # within it are prefixed with resnet_feat and for restoring resnet pretrained weights 
                    # we need to update the statedict according to the model architectural definition.
                    self.log(f"{new_var_name} : {list(var_value.size())}")
                    filtered_checkpoint[new_var_name] = var_value
                else:
                    raise NotImplementedError('Pretrained model restoration is not implemented for ', config.backbone)

            self.log("\n\nFollowing variables will be initialized:")
            remaining_vars = model.load_state_dict(filtered_checkpoint, strict=False)
            for var_name in remaining_vars.missing_keys:
                self.log(var_name)
            
            resume_g_step = 0
            resume_eps = 0
        else:
            resume_g_step = 0
            resume_eps = 0

        train_writer = SummaryHelper(self.exp_path + "/train/")
        test_writer = SummaryHelper(self.exp_path + "/test/")

        input_x = torch.randn((1,3, config.input_size[0], config.input_size[1])).to(cuda, non_blocking=True)
        train_writer.add_graph(model, input_x)

        # Dataloader for train and test dataset
        train_loader = get_train_loader()
        test_loader = get_test_loader()


        g_step = max(0, resume_g_step)
        for eps in range(resume_eps, config.epochs):
            # I hope you noticed one particular statement in the code, to which I assigned a comment “What is this?!?” — model.train().
            # In PyTorch, models have a train() method which, somewhat disappointingly, does NOT perform a training step. 
            # Its only purpose is to set the model to training mode. Why is this important? Some models may use mechanisms like Dropout, 
            # for instance, which have distinct behaviors in training and evaluation phases.
            # Ref: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e
            model.train()

            train_iter = iter(train_loader)                     # This is creating issues sometimes. Check required.
            
            self.log("Epoch: {} Started".format(eps+1))
                
            for batch_num in tqdm(range(config.train_steps)):
                # start = time.time()
                batch = next(train_iter)

                opt.zero_grad()                             # Zeroing out gradients before backprop
                                                            # We cab avoid to zero out if we want accumulate gradients for 
                                                            # Multiple forward pass and single backward pass.
                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    predictions = model(batch['image'].to(cuda, non_blocking=True))
                    # cls_probs, cnt_logits, reg_values = predictions
                    # cls_probs : [[B x 81 x H x W], [B x 81 x H x W], ....]
                    # cnt_logits: [[B x 1 x H x W], [B x 1 x H x W], ....]
                    # reg_values: [[B x 4 x H x W], [B x 4 x H x W], ....]

                    loss_total, loss_clsf, loss_cntness, loss_regression, loss_regularizer = \
                        self.compute_loss_and_map(predictions, batch['bbox'].to(cuda, non_blocking=True), model, fcos_loss_fn)
                        
                if config.use_amp:
                    scaler.scale(loss_total).backward()		# Used when AMP is applied.
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss_total.backward()

                lr = self.lr_helper.step(g_step, opt)
                opt.step()
                # delta = (time.time() - start) * 1000        # in milliseconds
                # print("\nTime: {:.2f} ms".format(delta))

                if (batch_num+1) % config.loss_logging_frequency == 0:
                    self.log(f"Epoch: {eps+1}/{config.epochs}, Batch No.: {batch_num+1}/{config.train_steps}, " + \
                            f"Total Loss: {np_cpu(loss_total):.4f}, Loss Cls: {np_cpu(loss_clsf):.4f}, " + \
                            f"Loss Cntness: {np_cpu(loss_cntness):.4f}, Loss Regression: {np_cpu(loss_regression):.4f}, " + \
                            f"Loss Reg: {np_cpu(loss_regularizer):.4f}, Accuracy: {0}")
                    
                    train_writer.add_summary({'total_loss' : np_cpu(loss_total),
                                            'loss_cls' : np_cpu(loss_clsf),
                                            'loss_cntness' : np_cpu(loss_cntness),
                                            'loss_regression' : np_cpu(loss_regression),
                                            'loss_reg' : np_cpu(loss_regularizer), 
                                            # 'accuracy' : np_cpu(accuracy),
                                            'lr' : lr}, g_step)
                
                g_step += 1
                
            
            model.eval()            # Putting model in eval mode so that batch normalization and dropout will work in inference mode.

            test_iter = iter(test_loader)
            test_losses = LossAverager(num_elements=6)

            with torch.no_grad():   # Disabling the gradient calculations will reduce the calculation overhead.

                for batch_num in tqdm(range(config.test_steps)):
                    batch = next(test_iter)
                    predictions = model(batch['image'].to(cuda, non_blocking=True))
                    
                    loss_total, loss_clsf, loss_cntness, loss_regression, loss_regularizer = \
                        self.compute_loss_and_map(predictions, batch['bbox'].to(cuda, non_blocking=True), model, fcos_loss_fn)
                    test_losses([np_cpu(loss_total), np_cpu(loss_clsf), np_cpu(loss_cntness), \
                        np_cpu(loss_regression), np_cpu(loss_regularizer), 0])
                    
                self.log(f"Epoch: {eps+1}/{config.epochs} Completed, Test Total Loss: {test_losses.avg[0]:.4f}, " + \
                        f"Loss Cls: {test_losses.avg[1]:.4f}, Loss Cntness: {test_losses.avg[2]:.4f}, " + \
                        f"Loss Regression: {test_losses.avg[3]:.4f}, Loss Reg: {test_losses.avg[4]:.4f}, " + \
                        f"Accuracy: {test_losses.avg[5]:.2f}")
                
                test_writer.add_summary({'total_loss' : test_losses.avg[0], 
                                        'loss_cls' : test_losses.avg[1], 
                                        'loss_cntness' : test_losses.avg[2], 
                                        'loss_regression' : test_losses.avg[3], 
                                        'loss_reg' : test_losses.avg[4], 
                                        # 'accuracy' : test_losses.avg[3],
                                        }, g_step)

            checkpoint = {
                'epoch': eps + 1,
                'global_step': g_step,
                'test_loss': test_losses.avg[0],
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            if config.use_amp:
                checkpoint['scalar'] = scaler.state_dict()

            # Above code taken from : https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
            self.ckpt_handler.save(checkpoint)
            self.log("Epoch {} completed. Checkpoint saved.".format(eps+1))

        print("Training Completed.")
        train_writer.close()
        test_writer.close()


    def overfit_train(self, pretrained_ckpt=None):
        model = self.get_model()
        
        model_stats = summary(model, (1, 3, config.input_size[0], config.input_size[1]))
        
        for line in str(model_stats).split('\n'):
            self.log(line)
        
        fcos_loss_fn = self.define_loss()
        
        opt = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=0.0)
        # Setting lr equal to 0.0 here so that it wont work as per this line.
        # But we will explicitly set lr for each weights dynamically, at every step.
        # Same is case for weight_decay, We will calculate L2_regularization_loss on our own separately.
        # L2 loss will only be calculated for conv-weights only.
        
        if config.use_amp:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        if pretrained_ckpt is not None and config.model_type == 'resnet18':
            self.log("Using pre-trained checkpoint from :".format(pretrained_ckpt))
            checkpoint = torch.load(pretrained_ckpt)
            
            filtered_checkpoint = {}
            self.log("\nFollowing variables will be restored:")
            for var_name, var_value in checkpoint.items():
                if var_name == 'fc.weight' or var_name == 'fc.bias':        
                    # As these layers change arent there in pretrained model definition
                    continue
                if config.backbone == 'resnet':
                    new_var_name = 'resnet_feat.' + var_name                
                    # why this prefix? This comes as the model that we created contains a variable resnet_feat 
                    # which is sequential group of layers containing resnet layers. So all the layers and parameters 
                    # within it are prefixed with resnet_feat and for restoring resnet pretrained weights 
                    # we need to update the statedict according to the model architectural definition.
                    self.log(f"{new_var_name} : {list(var_value.size())}")
                    filtered_checkpoint[new_var_name] = var_value
                else:
                    raise NotImplementedError('Pretrained model restoration is not implemented for ', config.backbone)

            self.log("\n\nFollowing variables will be initialized:")
            remaining_vars = model.load_state_dict(filtered_checkpoint, strict=False)
            for var_name in remaining_vars.missing_keys:
                self.log(var_name)
            
        train_writer = SummaryHelper(self.exp_path + "/train/")

        input_x = torch.randn((1,3, config.input_size[0], config.input_size[1])).to(cuda, non_blocking=True)
        train_writer.add_graph(model, input_x)

        # Dataloader for train and test dataset
        train_loader = get_train_loader()
        train_iter = iter(train_loader)
        batch = next(train_iter)
        
        img_tensor = batch['image'][0:1].to(cuda, non_blocking=True)
        
        import cv2
        from utils.transforms.normalize import Normalize
        orig_img = Normalize(config.converter_normalization_type).denor(img_tensor)
        orig_img = np.transpose(orig_img[0], [1, 2, 0])
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(orig_img, "./sample_imgs/overfit_img")
        exit()

        single_bb_label = batch['bbox'].to(cuda, non_blocking=True)

        for step in range(config.overfit_epochs):
            model.train()

            opt.zero_grad()                             # Zeroing out gradients before backprop
                                                        # We cab avoid to zero out if we want accumulate gradients for 
                                                        # Multiple forward pass and single backward pass.
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                predictions = model(img_tensor)
                # cls_probs, cnt_logits, reg_values = predictions
                # cls_probs : [[B x 81 x H x W], [B x 81 x H x W], ....]
                # cnt_logits: [[B x 1 x H x W], [B x 1 x H x W], ....]
                # reg_values: [[B x 4 x H x W], [B x 4 x H x W], ....]

                loss_total, loss_clsf, loss_cntness, loss_regression, loss_regularizer = \
                    self.compute_loss_and_map(predictions, single_bb_label, model, fcos_loss_fn)
                        
                if config.use_amp:
                    scaler.scale(loss_total).backward()		# Used when AMP is applied.
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss_total.backward()

                lr = self.lr_helper.step(step, opt)
                opt.step()

                if (batch_num+1) % config.loss_logging_frequency == 0:
                    self.log(f"Epoch: {step+1}/{config.epochs}, Batch No.: {batch_num+1}/{config.train_steps}, " + \
                            f"Total Loss: {np_cpu(loss_total):.4f}, Loss Cls: {np_cpu(loss_clsf):.4f}, " + \
                            f"Loss Cntness: {np_cpu(loss_cntness):.4f}, Loss Regression: {np_cpu(loss_regression):.4f}, " + \
                            f"Loss Reg: {np_cpu(loss_regularizer):.4f}, Accuracy: {0}")
                    
                    train_writer.add_summary({'total_loss' : np_cpu(loss_total),
                                            'loss_cls' : np_cpu(loss_clsf),
                                            'loss_cntness' : np_cpu(loss_cntness),
                                            'loss_regression' : np_cpu(loss_regression),
                                            'loss_reg' : np_cpu(loss_regularizer), 
                                            # 'accuracy' : np_cpu(accuracy),
                                            'lr' : lr}, step)
                
            

            checkpoint = {
                'epoch': step + 1,
                'global_step': step,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
            }
            if config.use_amp:
                checkpoint['scalar'] = scaler.state_dict()

            # Above code taken from : https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
            self.ckpt_handler.save(checkpoint)
            self.log("Epoch {} completed. Checkpoint saved.".format(eps+1))

        print("Training Completed.")
        train_writer.close()
