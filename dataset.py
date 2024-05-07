import os
import decord
import numpy as np
import random
import json
import torchvision
import torchvision.transforms as T
import torch

from glob import glob
from PIL import Image
from itertools import islice
from pathlib import Path
import pandas as pd

decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange, repeat


    
def normalize_input(
        item, 
        width,
        height,
        mean=[0.5, 0.5, 0.5], # Imagenet [0.485, 0.456, 0.406]
        std=[0.5, 0.5, 0.5], # Imagenet [0.229, 0.224, 0.225]
        use_simple_norm=False,
    ):  
        item = T.CenterCrop(size = (height, width))(item)
        if item.dtype == torch.uint8 and not use_simple_norm:
            item = rearrange(item, 'f c h w -> f h w c')
            item = item.float() / 255.0
            mean = torch.tensor(mean)
            std = torch.tensor(std)
            out = rearrange((item - mean) / std, 'f h w c -> f c h w')            
            return out
        else:
            item = rearrange(item, 'f c h w -> f h w c')
            return  rearrange(item / 127.5 - 1.0, 'f h w c -> f c h w')
            
def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

### Change to uniform sampler from mplugowl 
def get_video_frames(vr, start_idx, max_range, sample_rate=1, max_frames=24):
    frame_indices = get_index(max_range, max_frames)
    return frame_indices

def process_video(root, vid_path, w, h, get_frame_batch, max_range_data):
    vid_location = os.path.join(root, vid_path)
    vr = decord.VideoReader(vid_location)
    try:
        max_range = max_range_data[vid_path]
    except:
        raise Exception(f'max range data does not have {vid_path}')
    video = get_frame_batch(vr, max_range = max_range)

    return video, vr

class VideoJsonlDataset(Dataset):
    def __init__(
            self,
            tokenizer = None,
            width: int = 256,
            height: int = 256,
            n_sample_frames: int = 8,
            sample_start_idx: int = 1,
            frame_step: int = 1,
            path="",
            root="",
            max_range_file="",
            vid_data_key: str = "video",
            time_aligned_caption: bool = False,
            max_scenes: int = 1,
            preprocessed: bool = False,
            **kwargs
    ):
        self.vid_types = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        
        self.root = root
        self.max_range_file = max_range_file
        with open(self.max_range_file, 'r') as f:
            self.max_range_data = json.load(f)

        self.vid_data_key = vid_data_key
        
        with open(path, 'r') as f:
            self.train_data = list(f)

        print(f"Time aligned caption: {time_aligned_caption}")
        self.time_aligned_caption = time_aligned_caption
        self.max_scenes = max_scenes
        self.n_sample_frames = n_sample_frames  ## sample frames per video segment

        self.width = width
        self.height = height

        print(f"sample frames: {self.n_sample_frames}")

        self.sample_start_idx = sample_start_idx
        self.frame_step = frame_step
            
    def get_frame_range(self, vr, max_range):
        return get_video_frames(
            vr, 
            self.sample_start_idx, 
            max_range,
            self.frame_step, 
            self.n_sample_frames 
        )
    
    def get_frame_batch(self, vr, max_range, resize=None):
        frame_range = self.get_frame_range(vr, max_range)
        frames = vr.get_batch(frame_range)
        video = rearrange(frames, "f h w c -> f c h w")
        if resize is not None: video = resize(video)
        return video


    def set_n_scenes(self, n):
        self.scenes = n

    def train_data_batch(self, index):

        vid_data = eval(self.train_data[index])
        video_segments = vid_data['video_segments']
        captions = vid_data['captions']
        if len(video_segments) > 1:
            n_scenes = random.randint(2, self.max_scenes)
            video_segments = video_segments[:n_scenes]
            captions = captions[:n_scenes]
        print(f"number of segments: {len(video_segments)}") 

        all_vids = []
        for vid_path in video_segments:
            video, _ = process_video(
                self.root,
                vid_path,
                self.width, 
                self.height, 
                self.get_frame_batch,
                self.max_range_data
            )
            all_vids.append(video)

        prompt_ids = []
        if self.time_aligned_caption:
            for caption in captions:
                prompt_ids.append(get_prompt_ids(caption, self.tokenizer))
        else:
            strg = captions[0]
            for caption in captions[1:]:
                strg = strg + " Then, " + caption
            prompt_ids.append(get_prompt_ids(strg, self.tokenizer))

        return all_vids, prompt_ids

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        
        videos, prompt_ids = self.train_data_batch(index)
        combine_vids = []
        for vid in videos:
            vid_pix_val = normalize_input(vid, self.width, self.height) # T x C x H x W
            combine_vids.append(vid_pix_val)
        combine_vids = torch.cat(combine_vids, dim = 0)
        example = {'video_pixel_values': combine_vids, 'prompt_ids': prompt_ids}
        return example