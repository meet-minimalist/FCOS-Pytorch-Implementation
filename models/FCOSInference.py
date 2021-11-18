
import os
import sys
from typing_extensions import final
sys.path.append("../")
# TODO : Remove this append line
import numpy as np
import torch
import torch.nn as nn
from models.FCOS import FCOS
from models.PostProcessor import PostProcessor
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from torchvision import transforms
from utils.transforms.to_tensor import ToTensorOwn
from utils.transforms.normalize import Normalize
from utils.transforms.center_crop import CenterCrop

cuda = torch.device('cuda:0')
cpu = torch.device("cpu:0")


class FCOSInference(nn.Module):
    def __init__(self, backbone_model='resnet50', freeze_backend=[False, False, False, False], \
                        fpn_features=256, num_classes=81, use_det_head_group_norm=True, \
                        centerness_on_regression=True, use_gradient_checkpointing=False, \
                        strides=[8, 16, 32, 64, 128], use_cuda=False, \
                        add_centerness_in_cls_prob=True, max_detection_boxes_num=1000, \
                        cls_score_threshold=0.05, nms_iou_threshold=0.60):

        super(FCOSInference, self).__init__()

        self.strides = strides
        self.max_detection_boxes_num = max_detection_boxes_num
        self.model = FCOS(backbone_model, freeze_backend, fpn_features, num_classes, \
            use_det_head_group_norm, centerness_on_regression, use_gradient_checkpointing)
        self.post_process = PostProcessor(use_cuda, add_centerness_in_cls_prob, \
                max_detection_boxes_num, cls_score_threshold, nms_iou_threshold, num_classes)
        
        if use_cuda:
            self.model = self.model.to(cuda, non_blocking=True)
            self.post_process = self.post_process.to(cuda, non_blocking=True)
            
    
    def forward(self, preprocesed_image):
        # image : [B x 3 x img_h x img_w]

        cls_probs, cnt_logits, reg_values = self.model(preprocesed_image)
        # cls_probs, cnt_logit, reg_values each will have a list of features having shape as below.
        # cls_probs : [[B x 81 x H x W], [B x 81 x H x W], ....]
        # cnt_logits: [[B x 1 x H x W], [B x 1 x H x W], ....]
        # reg_values: [[B x 4 x H x W], [B x 4 x H x W], ....]
        

        # def print_data(x):
        #     x = x.detach().numpy()
        #     print(f"Min: {np.min(x):.4f}, Max: {np.max(x):.4f}, Mean: {np.mean(x):.4f}, Std: {np.std(x):.4f}")

        
        # for a, b, c in zip(cls_probs, cnt_logits, reg_values):
        #     print_data(a)
        #     print_data(b)
        #     print_data(c)
        #     print("="*30)
        # exit()

        predictions = self.post_process([cls_probs, cnt_logits, reg_values], self.strides)
        # predictions : List of [N x 6] tensor for each element in batch
        #             : [x1, y1, x2, y2, cls_prob, cls_id]

        B = preprocesed_image.shape[0]
        num_bboxes = torch.zeros(size=[B])
        for i, res_img in enumerate(preprocesed_image):
            img_h, img_w = res_img.shape[1:]    

            predictions[i][:, 0] = torch.clip(predictions[i][:, 0], 0, img_w)
            predictions[i][:, 1] = torch.clip(predictions[i][:, 1], 0, img_h)
            predictions[i][:, 2] = torch.clip(predictions[i][:, 2], 0, img_w)
            predictions[i][:, 3] = torch.clip(predictions[i][:, 3], 0, img_h)
            num_bboxes[i] = len(predictions[i])

        final_prediction = torch.zeros(size=[B, self.max_detection_boxes_num, 6], dtype=torch.float32)

        for i, pred in enumerate(predictions):
            final_prediction[i, :len(pred)] = pred
            
        return final_prediction, num_bboxes


if __name__ == "__main__":
    import cv2
    import config_converter as config

    complete_model = FCOSInference(backbone_model=config.converter_backbone, freeze_backend=[False, False, False, False], \
            fpn_features=config.converter_fpn_features, num_classes=config.converter_num_classes, \
            use_det_head_group_norm=config.converter_use_det_head_group_norm, \
            centerness_on_regression=config.converter_centerness_on_regression, \
            use_gradient_checkpointing=False, strides=config.converter_strides, use_cuda=False, \
            add_centerness_in_cls_prob=config.add_centerness_in_cls_prob, \
            max_detection_boxes_num=config.max_detection_boxes_num, \
            cls_score_threshold=config.cls_score_threshold, \
            nms_iou_threshold=config.nms_iou_threshold)

    ckpt_path = "../summaries/2021_07_26_00_01_29/ckpt/fcos_resnet50_eps_26_test_loss_2.5426.pth"

    ckpt = torch.load(ckpt_path)['model']
    complete_model.model.load_state_dict(ckpt, strict=True)      # Restore FCOS architecture part only
    complete_model.model.eval()        # TODO : Skipping this intentionally
    complete_model.eval()

    # Image loading and preprocessing
    img_path = "../sample_imgs/000026.jpg"
    # img_path = "../sample_imgs/000012.jpg"
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    transforms = transforms.Compose([
                    CenterCrop(),
                    ToTensorOwn(),             # Custom ToTensor transform, converts to CHW from HWC only
                    Normalize(config.converter_normalization_type),
                ])
    empty_bb = BoundingBoxesOnImage([BoundingBox(0, 0, 100, 100, label=0)], \
                    shape=(*config.input_size, 3))
    
    sample = {'image' : img, 'bbox' : empty_bb}
    preprocessed_tensor = transforms([sample, config.input_size])
    resized_img = preprocessed_tensor['image']
    resized_img = torch.unsqueeze(resized_img, dim=0)

    # Model Inference
    final_predictions, num_bboxes = complete_model(resized_img)
    final_predictions = final_predictions.detach().numpy()
    num_bboxes = num_bboxes.detach().numpy()
    resized_img = resized_img.detach().numpy()

    for pred, num_bb, img in zip(final_predictions, num_bboxes, resized_img):
        pred = pred[:int(num_bb)]            
        # Rest are padded zeros and not useful as we padded the predictions to make a batch of output

        img[0:1, :, :] = img[0:1, :, :] * 0.229 + 0.485 
        img[1:2, :, :] = img[1:2, :, :] * 0.224 + 0.456
        img[2:3, :, :] = img[2:3, :, :] * 0.225 + 0.406
        img = np.uint8(np.transpose(img, (1, 2, 0)) * 255)
        
        for bb in pred:
            x1, y1, x2, y2 = [int(c) for c in bb[:4]]
            cls_prob, cls_id = bb[4:] 
            cls_name = config.converter_label_dict[int(cls_id)]
            print(f"X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}, Cls_id: {int(cls_id)}, Cls_name: {cls_name}, Cls_prob: {cls_prob:.4f}")
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        op_path = os.path.splitext(img_path)[0] + "_res.jpg"
        cv2.imwrite(op_path, img)
