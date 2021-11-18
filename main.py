import os

import onnxruntime
import config
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FCOS Helper.')
    parser.add_argument('--mode', action='store', required=True, type=str, help='train : Training mode, ' + \
            'overfit_train : Overfit Train on only single image, convert : Convert to onnx and torchscript model, ' + \
            'or infer : Inference Mode.')
    parser.add_argument('--ckpt_path', required=False, type=str, help='When train mode is used, this checkpoint is used to resume training. ' + \
        'When convert mode is used, this checkpoint is used for conversion. When inference mode is used, this checkpoint is used for inference.')
    parser.add_argument('--use_pretrain_ckpt', required=False, action='store_true', help='True/False. Use pretrain checkpoint ' + \
        'as a starting weights for backbone network. Cant be used when resuming the training. Only used for train mode.')
    parser.add_argument('--img_path', required=False, type=str, help='Path of image for infer mode.')


    args = parser.parse_args()

    if args.mode not in ['train', 'overfit_train', 'convert', 'infer']:
        raise ValueError("Please provide mode from : ['train', 'overfit_train', 'convert', 'infer']")

    if args.mode == 'train':
        # python main.py --mode train --ckpt_path ./summaries/2021_07_26_00_01_29/ckpt/fcos_resnet50_eps_25_test_loss_2.5426.pth
        # python main.py --mode train --use_pretrain_ckpt

        from TrainingHelper import TrainingHelper
        trainer = TrainingHelper()
        if args.ckpt_path:
            trainer.train(resume=True, resume_ckpt=args.ckpt_path)
        elif args.use_pretrain_ckpt:
            trainer.train(pretrained_ckpt=config.backbone_ckpt)
        else:
            trainer.train()

    if args.mode == 'overfit_train':
        # python main.py --mode overfit_train --use_pretrain_ckpt
        from TrainingHelper import TrainingHelper
        trainer = TrainingHelper()
        if args.use_pretrain_ckpt:
            trainer.overfit_train(pretrained_ckpt=config.backbone_ckpt)
        else:
            trainer.overfit_train()

    if args.mode == 'convert':
        # python main.py --mode convert --ckpt_path ./summaries/2021_07_26_00_01_29/ckpt/fcos_resnet50_eps_26_test_loss_2.5426.pth

        from ModelConverter import ModelConverter
        converter = ModelConverter()
        if not args.ckpt_path:
            raise ValueError('ckpt_path should be provided.')
        converter.convert_model(args.ckpt_path)

    if args.mode == 'infer':
        # python main.py --mode infer --ckpt_path ./exported_model/fcos_resnet50_eps_26_test_loss_2.5426.onnx --img_path ./sample_imgs/000026.jpg

        if not args.ckpt_path:
            raise ValueError('ckpt_path should be the exported ONNX/Torchscript model.')
        if not args.img_path or not os.path.isfile(args.img_path):
            raise ValueError('Please provide valid image path.')

        if os.path.splitext(args.ckpt_path)[1] == '.onnx':
            onnx_model = True
        else:
            onnx_model = False

        from InferModel import Inference

        infer_model = Inference()
        infer_model.run(args.img_path, args.ckpt_path, onnx_model)