
import os
import glob
import torch
import shutil
import datetime

import config
from utils.Logger import Logger

cpu = torch.device("cpu:0")

def init_training(exp_path=config.exp_path):
    start_time = datetime.datetime.now()
    exp_name = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    cur_exp_path = exp_path + "/" + exp_name
    os.makedirs(cur_exp_path, exist_ok=True)
    
    logger_path = cur_exp_path + "/log.txt"
    log = Logger(logger_path)

    # =========== Take a backup of training files ============= #
    current_files = glob.glob("*")
    for i in range(len(current_files)):
        if os.path.isfile(current_files[i]):
            shutil.copy2(current_files[i], cur_exp_path)
    
    shutil.copytree("./models/", cur_exp_path + "/models")
    shutil.copytree("./utils/", cur_exp_path + "/utils")
    # ================================================ #
    
    log("Experiment Start time: {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S")))
    log("Experiment files are saved at: {}".format(cur_exp_path))
    log("Training initialization completed.")
    return log, cur_exp_path


def np_cpu(tensor):
    return float(tensor.detach().to(cpu).numpy())


class LossAverager:
    def __init__(self, num_elements):
        self.num_elements = num_elements
        self.reset()

    def reset(self):
        self.val    = [0 for _ in range(self.num_elements)]
        self.count  = [0 for _ in range(self.num_elements)]
        self.sum    = [0 for _ in range(self.num_elements)]
        self.avg    = [0 for _ in range(self.num_elements)]


    def __call__(self, val_list):
        assert len(val_list) == self.num_elements

        for i, val in enumerate(val_list):
            self.val[i]     = val
            self.sum[i]     += self.val[i]
            self.count[i]   += 1
            self.avg[i]     = self.sum[i] / self.count[i]
