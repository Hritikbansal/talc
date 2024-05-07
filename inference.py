import os
import cv2
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from einops import repeat
from transformers import  CLIPTextModel, CLIPTokenizer
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, DDIMScheduler, TextToVideoSDPipeline

parser = argparse.ArgumentParser()

parser.add_argument('--captions', nargs='+', help='prompts to generate', required=True)
parser.add_argument('--outfile', type = str, help = "name of the output video file")
parser.add_argument('--model-name-path', type = str, help = "model name or path")
parser.add_argument('--frames-per-scene', type = int, help = "number of frames per scene", default = 16)
parser.add_argument('--guidance_scale', type = int, help = "guidance scale", default = 12)
parser.add_argument('--steps', type = int, help = "inference steps", default = 100)
parser.add_argument('--width', type = int, help = "width", default = 256)
parser.add_argument('--height', type = int, help = "height", default = 256)
parser.add_argument('--talc', action = "store_true")
parser.add_argument('--merge', action = "store_true")

args = parser.parse_args()

device = "cuda"

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids


def main():

    pretrained_model_path = args.model_name_path
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder.to(device, dtype=torch.float16)
    text_encoder.eval()

    if "damo" in pretrained_model_path:
        pipeline = TextToVideoSDPipeline.from_pretrained(pretrained_model_path, torch_dtype=torch.float16, variant='fp16').to("cuda")
    else:
        pipeline = TextToVideoSDPipeline.from_pretrained(pretrained_model_path).to("cuda", dtype=torch.float16)
    diffusion_scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler = diffusion_scheduler

    prompt = args.captions
    max_scenes = 4

    if len(prompt) > max_scenes:
        prompt = prompt[:max_scenes]

    num_frames = args.frames_per_scene * len(prompt)
    length = len(prompt)
    outfile = args.outfile

    if not args.merge:
        if args.talc:
            with torch.no_grad():
                all_eh = []
                all_neh = []
                for j in range(length):
                    prompt_ids = get_prompt_ids(prompt[j], tokenizer).to(device)
                    text_emb = text_encoder(prompt_ids)[0][0]
                    text_emb = repeat(text_emb, 'b c -> a b c', a = args.frames_per_scene)
                    all_eh.append(text_emb.detach().clone())
                    neg_prompt_ids = get_prompt_ids([""], tokenizer).to(device)
                    neg_text_emb = text_encoder(neg_prompt_ids)[0][0]
                    neg_text_emb = repeat(neg_text_emb, 'b c -> a b c', a = args.frames_per_scene)
                    all_neh.append(neg_text_emb.detach().clone())
                prompt_embeds = torch.cat(all_eh, dim = 0)
                negative_prompt_embeds = torch.cat(all_neh, dim = 0)
                video_frames = pipeline(
                        width=args.width,
                        height=args.height,
                        num_frames=num_frames,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        time_aligned_caption=True,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds                
                    ).frames
        else:
            strg = prompt[0]
            for p in prompt[1:]:
                strg = strg + " Then, " + p
            with torch.no_grad():
                video_frames = pipeline(
                    strg,
                    width=args.width,
                    height=args.height,
                    num_frames=num_frames,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                ).frames
        export_to_video(video_frames, outfile, int(args.frames_per_scene/2))
    else:
        all_videos = []
        with torch.no_grad():
            for p in prompt:
                video_frames = pipeline(
                    p,
                    width=args.width,
                    height=args.height,
                    num_frames=args.frames_per_scene,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                ).frames
                all_videos.append(video_frames)
            merged_video = np.concatenate(all_videos, axis = 0)
        export_to_video(merged_video, outfile, int(args.frames_per_scene/2))

    print('Done')

if __name__ == "__main__":
    main()
