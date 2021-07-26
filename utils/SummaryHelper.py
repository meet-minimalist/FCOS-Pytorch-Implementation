
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class SummaryHelper:
    def __init__(self, summary_path):
        self.summary_writer = SummaryWriter(summary_path)

    def add_graph(self, model, ip_tensor):
        self.summary_writer.add_graph(model, ip_tensor)

    def add_summary(self, summary_dict, gstep):
        for key, value in summary_dict.items():

            if isinstance(value, float) or isinstance(value, int):
                # For scalar values,
                self.summary_writer.add_scalar(key, value, gstep)
            elif isinstance(value, np.ndarray):
                # For images,
                self.summary_writer.add_image(key, value, gstep, dataformats='CHW')
            else:
                print("Summary Input not identified", type(value))

            self.summary_writer.flush()

    def close(self):
        self.summary_writer.close()

    
        