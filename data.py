import torch
import os
import skvideo
import numpy as np


def trim(video, T):
    start = np.random.randint(0, video.shape[1] - T)
    end = start + T
    return video[:, start:end, :, :]


class SkVideoFolder(torch.utils.data.Dataset):
    def __init__(self, data_root, T):
        self.data_root = data_root
        self.T = T
        self.videos = [os.path.join(self.data_root, i) for i in os.listdir(self.data_root) if i.endswith('.mp4')]

    def __getitem__(self, index):
        video = skvideo.io.vread(self.videos[index])
        video = video.transpose(3, 0, 1, 2) / 255.0
        video = trim(video, self.T)
        return video

    def __len__(self):
        return len(self.videos)
