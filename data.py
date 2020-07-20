import torch
import os
import skvideo


class SkVideoFolder(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.videos = [os.path.join(self.data_root, i) for i in os.listdir(self.data_root) if i.endswith('.mp4')]

    def __getitem__(self, index):
        video = skvideo.io.vread(self.videos[index])
        video = video.transpose(3, 0, 1, 2) / 255.0
        return video

    def __len__(self):
        return len(self.videos)
