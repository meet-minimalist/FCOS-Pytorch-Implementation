import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.transforms.to_tensor import ToTensorOwn
from utils.transforms.normalize import Normalize
from utils.transforms.center_crop import CenterCrop
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import config_converter as config

class Inference:
    def __init__(self):
        self.transforms = transforms.Compose([
                CenterCrop(),
                ToTensorOwn(),             # Custom ToTensor transform, converts to CHW from HWC only
                Normalize(config.converter_normalization_type),
            ])
        self.empty_bb = BoundingBoxesOnImage([BoundingBox(0, 0, 100, 100, label=0)], \
                        shape=(*config.input_size, 3))

    def run(self, img_path, model_path, onnx_model, torchscript_model):
        # Image loading and preprocessing
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sample = {'image' : img, 'bbox' : self.empty_bb}
        preprocessed_tensor = self.transforms([sample, config.input_size])
        resized_img = preprocessed_tensor['image']
        resized_img = torch.unsqueeze(resized_img, dim=0)

        # Model Inference
        if onnx_model:
            import onnxruntime
            session = onnxruntime.InferenceSession(model_path, None)
            input_name = session.get_inputs()[0].name

            final_predictions, num_bboxes = session.run(None, {input_name: resized_img.numpy()})
        elif torchscript_model:
            model = torch.jit.load(model_path)
            model.eval()
            final_predictions, num_bboxes = model(resized_img)
            final_predictions = final_predictions.detach().numpy()
            num_bboxes = num_bboxes.detach().numpy()
        else:
            from models.FCOSInference import FCOSInference
            complete_model = FCOSInference(backbone_model=config.converter_backbone, freeze_backend=[False, False, False, False], \
                        fpn_features=config.converter_fpn_features, num_classes=config.converter_num_classes, \
                        use_det_head_group_norm=config.converter_use_det_head_group_norm, \
                        centerness_on_regression=config.converter_centerness_on_regression, \
                        use_gradient_checkpointing=False, use_cuda=False, \
                        add_centerness_in_cls_prob=config.add_centerness_in_cls_prob, \
                        max_detection_boxes_num=config.max_detection_boxes_num, \
                        cls_score_threshold=config.cls_score_threshold, \
                        nms_iou_threshold=config.nms_iou_threshold)
            ckpt = torch.load(model_path, map_location='cpu')['model']
            complete_model.model.load_state_dict(ckpt, strict=True)      # Restore FCOS architecture part only
            # fcos_model.eval()        # TODO : Skipping this intentionally
            complete_model.eval()
            final_predictions, num_bboxes = complete_model(resized_img)
            final_predictions = final_predictions.detach().numpy()
            num_bboxes = num_bboxes.detach().numpy()
            
        resized_img = resized_img.detach().numpy()

        print("Total detections : ", num_bboxes)

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
            print("Result saved at: ", op_path)
