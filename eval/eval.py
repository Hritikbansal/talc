import os
import re
import io
import cv2
import csv
import json
import openai
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
import base64
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--vidpath', type = str, help = 'data file', required=True)
parser.add_argument('--captions', nargs='+', help='prompts to generate', required=True)

args = parser.parse_args()

PROMPT_TWO = """You are a capable video evaluator. You will be shown a text script with two-scene descriptions where the events/actions . Video generating AI models receive this text script as input and asked to generate relevant videos. You will be provided with eight video frames from the generated video. Your task is to answer the following questions for the generated video. 
1. Entity Consistency: Throughout the video, are entities consistent? (e.g., clothes do not change without a change described in the text script)
2. Background Consistency: Throughout the video, is the background consistent? (e.g., the room does not change described in the text script)
3. Text Adherence: Does the video adhere to the script? (e.g., are events/actions described in the script shown in the video accurately and in the correct temporal order)

Respond with NO, PARTIALLY, and YES for each category at the end. Do not provide any additional explanations.

Two-scene descriptions: 

Scene 1: {scene1}
Scene 2: {scene2}"""

PROMPT_THREE = """You are a capable video evaluator. You will be shown a text script with three-scene descriptions where the events/actions . Video generating AI models receive this text script as input and asked to generate relevant videos. You will be provided with twelve video frames from the generated video. Your task is to answer the following questions for the generated video. 
1. Entity Consistency: Throughout the video, are entities consistent? (e.g., clothes do not change without a change described in the text script)
2. Background Consistency: Throughout the video, is the background consistent? (e.g., the room does not change described in the text script)
3. Text Adherence: Does the video adhere to the script? (e.g., are events/actions described in the script shown in the video accurately and in the correct temporal order)

Respond with NO, PARTIALLY, and YES for each category at the end. Do not provide any additional explanations.

three-scene descriptions: 

Scene 1: {scene1}
Scene 2: {scene2}
Scene 3: {scene3}"""

PROMPT_FOUR = """You are a capable video evaluator. You will be shown a text script with four-scene descriptions where the events/actions . Video generating AI models receive this text script as input and asked to generate relevant videos. You will be provided with sixteen video frames from the generated video. Your task is to answer the following questions for the generated video. 
1. Entity Consistency: Throughout the video, are entities consistent? (e.g., clothes do not change without a change described in the text script)
2. Background Consistency: Throughout the video, is the background consistent? (e.g., the room does not change described in the text script)
3. Text Adherence: Does the video adhere to the script? (e.g., are events/actions described in the script shown in the video accurately and in the correct temporal order)

Respond with NO, PARTIALLY, and YES for each category at the end. Do not provide any additional explanations.

Four-scene descriptions: 

Scene 1: {scene1}
Scene 2: {scene2}
Scene 3: {scene3}
Scene 4: {scene4}"""

def image_to_base64(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='png')
    imgByteArr = imgByteArr.getvalue()
    return base64.b64encode(imgByteArr).decode('utf-8')


def get_payload(scenes, images, num_frames):

    if num_frames == 8:
        text = PROMPT_TWO.format(scene1 = scenes[0], scene2 = scenes[1])

    elif num_frames == 12:
        text = PROMPT_THREE.format(scene1 = scenes[0], scene2 = scenes[1], scene3 = scenes[2])

    elif num_frames == 16:
        text = PROMPT_FOUR.format(scene1 = scenes[0], scene2 = scenes[1], scene3 = scenes[2], scene4 = scenes[3])

    else:
        raise Exception(f'Number of frames: {num_frames} which is not supported -- {scenes}')

    payload = {
        "messages": [
            {
            "role": "user",
            "content": [{
                "type": "text",
                "text": text}]
            }
            ]}
    for image in images:
        payload['messages'][0]['content'].append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image}'}})

    return payload

def video_to_frames(vidpath, num_frames):
    cap = cv2.VideoCapture(vidpath)
    frames = []
    while 1:
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img)
            im_pil.save('test.png')
            frames.append(im_pil)
        else:
            break
    diff = len(frames) // num_frames
    frames = frames[1::diff][:num_frames]
    print(f'length of frames: {len(frames)}')
    frames = [image_to_base64(frame) for frame in frames]
    return frames
        
def get_scores(output):
    output = output.split("\n")
    if len(output) == 3:
        entity, background, adherence = -1, -1, -1
        if 'Entity Consistency: ' in output[0]:
            entity = 1 if 'YES' in output[0] else 0.5 if 'PARTIALLY' in output[0] else 0
        if 'Background Consistency: ' in output[1]:
            background = 1 if 'YES' in output[1] else 0.5 if 'PARTIALLY' in output[1] else 0
        if 'Text Adherence: ' in output[2]:
            adherence = 1 if 'YES' in output[2] else 0.5 if 'PARTIALLY' in output[2] else 0
        if entity >= 0 and background >= 0 and adherence >= 0:
            return [entity, background, adherence]
    return None


def main():

    vidpath = args.vidpath
    prompts = args.captions
    num_frames = 4 * len(prompts)

    images = video_to_frames(vidpath, num_frames)
    
    messages = get_payload(prompts, images, num_frames)['messages']
    response = openai.ChatCompletion.create(model = "gpt-4-vision-preview", messages = messages, max_tokens=300)
    output = response['choices'][0]['message']['content']
    print(output)

if __name__ == "__main__":
    main()