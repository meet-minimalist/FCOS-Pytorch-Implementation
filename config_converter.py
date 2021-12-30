import os

# --------------- Model Converter --------------- #
model_output_folder                 = os.path.dirname(os.path.abspath(__file__)) + "/exported_model/"
input_size                          = [400, 512]            # [H x W]

converter_label_txt                 = os.path.dirname(os.path.abspath(__file__)) + "/dataset/voc_labels.txt"
converter_label_dict                = {}
converter_label_dict[0]             = 'background'                      # background class at index 0
with open(converter_label_txt, 'r') as f: 
    for i, line in enumerate(f.readlines(), 1):
        converter_label_dict[i] = line.replace('\n', '')
converter_num_classes               = len(converter_label_dict)     # 21

converter_backbone                  = 'resnet18'         # ['resnet50', 'resnet18']
# To freeze features upto P2, P3, P4, P5 level in backbone, 4 boolean values are provided.  
converter_normalization_type        = 'imagenet'        # ['imagenet', 'div_255']
converter_strides                   = [8, 16, 32, 64, 128]
converter_fpn_features              = 256
converter_use_cntr_sampling         = True
converter_use_det_head_group_norm   = True
converter_centerness_on_regression  = True
add_centerness_in_cls_prob          = True
max_detection_boxes_num             = 1000
cls_score_threshold                 = 0.4
nms_iou_threshold                   = 0.3
# ----------------------------------------------- #                    


