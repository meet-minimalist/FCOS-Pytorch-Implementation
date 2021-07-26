from TrainingHelper import TrainingHelper

if __name__ == "__main__":
    trainer = TrainingHelper()

    trainer.train()
    # restore_ckpt = "./summaries/2021_05_22_17_12_01/ckpt/model_eps_2_test_loss_1.4546.pth"
    # trainer.train(resume=True, resume_ckpt=restore_ckpt)
    # resnet_ckpt = "./pretrained_ckpt/resnet18-5c106cde.pth"
    # trainer.train(pretrained_ckpt=resnet_ckpt)

