
import os
import onnx
import torch
from onnxsim import simplify
from torchinfo import summary

import config_converter as config
from models.FCOSInference import FCOSInference

cuda = torch.device('cuda')
cpu = torch.device("cpu")

class ModelConverter:
    def __init__(self, use_cuda=False):
        os.makedirs(config.model_output_folder, exist_ok=True)
        self.use_cuda = use_cuda
        self.device = cuda if self.use_cuda else cpu

    def get_model(self):
        with torch.no_grad():
            complete_model = FCOSInference(backbone_model=config.converter_backbone, freeze_backend=[False, False, False, False], \
                        fpn_features=config.converter_fpn_features, num_classes=config.converter_num_classes, \
                        use_det_head_group_norm=config.converter_use_det_head_group_norm, \
                        centerness_on_regression=config.converter_centerness_on_regression, \
                        use_gradient_checkpointing=False, use_cuda=self.use_cuda, \
                        add_centerness_in_cls_prob=config.add_centerness_in_cls_prob, \
                        max_detection_boxes_num=config.max_detection_boxes_num, \
                        cls_score_threshold=config.cls_score_threshold, \
                        nms_iou_threshold=config.nms_iou_threshold)
        
        return complete_model


    def convert_model(self, ckpt_path, op_folder='./exported_model/'):
        complete_model = self.get_model()
        fcos_model = complete_model.model

        model_stats = summary(fcos_model, (1, 3, config.input_size[0], config.input_size[1]), device=self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)['model']
        fcos_model.load_state_dict(ckpt, strict=True)      # Restore FCOS architecture part only
        # fcos_model.eval()        # TODO : Skipping this intentionally
        complete_model.eval()

        dummy_input = torch.randn((1, 3, config.input_size[0], config.input_size[1]), device=self.device, requires_grad=False)

        traced_model = torch.jit.trace(complete_model, dummy_input)
        torchscript_path = op_folder + "/" + os.path.splitext(os.path.basename(ckpt_path))[0] + ".pt"
        traced_model.save(torchscript_path)

        onnx_path = op_folder + "/" + os.path.splitext(os.path.basename(ckpt_path))[0] + ".onnx"
        torch.onnx.export(complete_model, dummy_input, onnx_path, opset_version=12, 
                          input_names=['input'], output_names=['output_bb', 'output_num_bb'],
                          dynamic_axes={'input': {0: 'B'}, 'output_bb': {0: 'B', 1: 'N'}, 'output_num_bb': {0: 'B'}})


        onnx_model = onnx.load(onnx_path)

        onnx_model_sim, status = simplify(onnx_model, input_shapes={'input':dummy_input.shape}, dynamic_input_shape=True)
        print("ONNX Simplification Status: ", status)

        onnx_model_sim_path = os.path.splitext(onnx_path)[0] + "_simplified.onnx"
        onnx.save(onnx_model_sim, onnx_model_sim_path)



if __name__ == "__main__":
    
    converter = ModelConverter(use_cuda=False)
    
    ckpt_path = "./summaries/2021_07_26_00_01_29/ckpt/fcos_resnet50_eps_26_test_loss_2.5426.pth"
    op_folder = config.model_output_folder
    converter.convert_model(ckpt_path, op_folder)