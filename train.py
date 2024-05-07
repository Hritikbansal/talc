import argparse
import datetime
import logging
import inspect
import math
import os
import random
import gc
import copy
import numpy as np

from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import cv2
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms as T
import diffusers
import transformers
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers.models.unet_3d_condition import UNet3DConditionModel
from diffusers.models import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler, DDPMScheduler, TextToVideoSDPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import AttnProcessor2_0, Attention
from diffusers.models.attention import BasicTransformerBlock

from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPEncoder
from dataset import VideoJsonlDataset
from einops import rearrange, repeat

# os.environ["WANDB_DISABLED"] = "true"
already_printed_trainables = False

logger = get_logger(__name__, log_level="INFO")

def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

def accelerate_set_verbose(accelerator):
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

def create_output_folders(output_dir, config):
    out_dir = os.path.join(output_dir)
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/samples", exist_ok=True)
    OmegaConf.save(config, os.path.join(out_dir, 'config.yaml'))

    return out_dir

def load_primary_models(pretrained_model_path):
    noise_scheduler = DDPMScheduler.from_config(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    if "damo" in pretrained_model_path:
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", variant = "fp16")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae", variant = "fp16")
        unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", variant = "fp16")
    else:
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
        unet = UNet3DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    return noise_scheduler, tokenizer, text_encoder, vae, unet

def unet_and_text_g_c(unet, text_encoder, unet_enable, text_enable):
    unet._set_gradient_checkpointing(value=unet_enable)
    text_encoder._set_gradient_checkpointing(CLIPEncoder, value=text_enable)

def freeze_models(models_to_freeze):
    for model in models_to_freeze:
        if model is not None: model.requires_grad_(False) 
            
def is_attn(name):
   return ('attn1' or 'attn2' == name.split('.')[-1])

def set_processors(attentions):
    for attn in attentions: attn.set_processor(AttnProcessor2_0()) 

def set_torch_2_attn(unet):
    optim_count = 0
    
    for name, module in unet.named_modules():
        if is_attn(name):
            if isinstance(module, torch.nn.ModuleList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0: 
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")

def handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet): 
    try:
        is_torch_2 = hasattr(F, 'scaled_dot_product_attention')
        enable_torch_2 = is_torch_2 and enable_torch_2_attn
        
        if enable_xformers_memory_efficient_attention and not enable_torch_2:
            if is_xformers_available():
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        if enable_torch_2:
            set_torch_2_attn(unet)
            
    except:
        print("Could not enable memory efficient attention for xformers or Torch 2.0.")

def param_optim(model, condition, extra_params=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model, 
        "condition": condition, 
        'extra_params': extra_params,
    }
    

def create_optim_params(name='param', params=None, lr=5e-6, extra_params=None):
    params = {
        "name": name, 
        "params": params, 
        "lr": lr
    }
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v
    
    return params

def get_optimizer(use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        return bnb.optim.AdamW8bit
    else:
        return torch.optim.AdamW

def is_mixed_precision(accelerator):
    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    return weight_dtype

def cast_to_gpu_and_type(model_list, accelerator, weight_dtype):
    for model in model_list:
        if model is not None: model.to(accelerator.device, dtype=weight_dtype)

def handle_cache_latents(
        should_cache, 
        output_dir, 
        train_dataloader, 
        train_batch_size, 
        vae, 
        cached_latent_dir=None,
        shuffle=False
    ):

    # Cache latents by storing them in VRAM. 
    # Speeds up training and saves memory by not encoding during the train loop.
    if not should_cache: return None
    vae.to('cuda', dtype=torch.float16)
    vae.enable_slicing()
    
    cached_latent_dir = (
        os.path.abspath(cached_latent_dir) if cached_latent_dir is not None else None 
        )

    if cached_latent_dir is None:
        cache_save_dir = f"{output_dir}/cached_latents"
        os.makedirs(cache_save_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(train_dataloader, desc="Caching Latents.")):

            save_name = f"cached_{i}"
            full_out_path =  f"{cache_save_dir}/{save_name}.pt"

            pixel_values = batch['pixel_values'].to('cuda', dtype=torch.float16)
            batch['pixel_values'] = tensor_to_vae_latent(pixel_values, vae)
            for k, v in batch.items(): batch[k] = v[0]
        
            torch.save(batch, full_out_path)
            del pixel_values
            del batch

            # We do this to avoid fragmentation from casting latents between devices.
            torch.cuda.empty_cache()
    else:
        cache_save_dir = cached_latent_dir
        

    return torch.utils.data.DataLoader(
        CachedDataset(cache_dir=cache_save_dir), 
        batch_size=train_batch_size, 
        shuffle=shuffle,
        num_workers=0
    ) 


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]
    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
    latents = latents * 0.18215
    return latents

def sample_noise(latents, noise_strength, use_offset_noise=False):
    b ,c, f, *_ = latents.shape
    noise_latents = torch.randn_like(latents, device=latents.device)
    offset_noise = None

    if use_offset_noise:
        offset_noise = torch.randn(b, c, f, 1, 1, device=latents.device)
        noise_latents = noise_latents + noise_strength * offset_noise

    return noise_latents

def enforce_zero_terminal_snr(betas):
    """
    Corrects noise in diffusion schedulers.
    From: Common Diffusion Noise Schedules and Sample Steps are Flawed
    https://arxiv.org/pdf/2305.08891.pdf
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1 - betas
    alphas_bar = alphas.cumprod(0)
    alphas_bar_sqrt = alphas_bar.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (
        alphas_bar_sqrt_0 - alphas_bar_sqrt_T
    )

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas


def save_pipe(
        path, 
        epoch,
        accelerator, 
        unet, 
        text_encoder, 
        vae, 
        output_dir,
        is_checkpoint=False,
        save_pretrained_model=True
    ):

    if is_checkpoint:
        save_path = os.path.join(output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = output_dir

   # Copy the model without creating a reference to it. This allows keeping the state of our lora training if enabled.
    unet_save = copy.deepcopy(unet)
    unet_out = copy.deepcopy(accelerator.unwrap_model(unet_save, keep_fp32_wrapper=False)).cpu()

    pipeline = TextToVideoSDPipeline.from_pretrained(
        path,
        unet=unet_out,
        text_encoder=text_encoder,
        vae=vae,
    ).to(torch_dtype=torch.float32)
    
    if save_pretrained_model:
        pipeline.save_pretrained(save_path)

    logger.info(f"Saved model at {save_path} on epoch {epoch}")
    
    del pipeline
    del unet_out
    torch.cuda.empty_cache()
    gc.collect()


def replace_prompt(prompt, token, wlist):
    for w in wlist:
        if w in prompt: return prompt.replace(w, token)
    return prompt 

def get_optim_params(unet, lr):
    params = []
    for n, p in unet.named_parameters():
        if p.requires_grad:
            params.append({"name": n, "params": p, "lr": lr})
    return params

def main(
    pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    extra_train_data: list = [],
    dataset_types: Tuple[str] = ('json'),
    shuffle: bool = True,
    validation_steps: int = 100,
    trainable_modules: str = None, # Eg: ("attn1", "attn2")
    extra_unet_params = None,
    extra_text_encoder_params = None,
    train_batch_size: int = 1,
    num_train_epochs: int = 1,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    text_encoder_gradient_checkpointing: bool = False,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    resume_step: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    enable_torch_2_attn: bool = False,
    seed: Optional[int] = None,
    train_text_encoder: bool = False,
    use_offset_noise: bool = False,
    rescale_schedule: bool = False,
    offset_noise_strength: float = 0.1,
    extend_dataset: bool = False,
    cache_latents: bool = False,
    cached_latent_dir = None,
    save_pretrained_model: bool = True,
    logger_type: str = 'wandb',
    conditioning_dropout_prob: float = 0.05,
    image: bool = False,
    prediction_type: str = "epsilon",
    **kwargs
):

    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir
    )

    # Make one log on every process with the configuration for debugging.
    create_logging(logging, logger, accelerator)

    # Initialize accelerate, transformers, and diffusers warnings
    accelerate_set_verbose(accelerator)

    # If passed along, set the training seed now.
    # if seed is not None:
    #     set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
       output_dir = create_output_folders(output_dir, config)

    # Load scheduler, tokenizer and models.
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_primary_models(pretrained_model_path)
    noise_scheduler.prediction_type = prediction_type
    # Freeze any necessary models
    freeze_models([vae, text_encoder])

    tp = 0
    if not image:     
        for name, param in unet.named_parameters():
            if trainable_modules == "temporal":
                if ("temp_convs" in name) or ("temp_attentions" in name):
                    param.requires_grad = True
                    tp += param.numel()
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True
                tp += param.numel()
    else:
        for name, param in unet.named_parameters():
            if ("temp_convs." in name) or ("temp_attentions." in name) or ("transformer_in." in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
                tp += param.numel()                

    print(f"Trainable params: {tp}")

    # Enable xformers if available
    handle_memory_attention(enable_xformers_memory_efficient_attention, enable_torch_2_attn, unet)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = get_optimizer(use_8bit_adam)

    optim_params = get_optim_params(unet, learning_rate)

    # Create Optimizer
    optimizer = optimizer_cls(
        optim_params,
        # unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    time_aligned_caption = kwargs['time_aligned_caption']
    
    # train dataset 
    train_dataset = VideoJsonlDataset(tokenizer = tokenizer, width = train_data['width'], height = train_data['height'], n_sample_frames = train_data['n_sample_frames'], path = kwargs['train_path'], root = kwargs['root'], max_range_file = kwargs['max_range_file'], max_scenes = kwargs['max_scenes'], time_aligned_caption = kwargs['time_aligned_caption'])

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=shuffle
    )

    # val dataset 
    val_dataset = VideoJsonlDataset(tokenizer = tokenizer, width = train_data['width'], height = train_data['height'], n_sample_frames = train_data['n_sample_frames'], path = kwargs['val_path'], root = kwargs['root'], max_range_file = kwargs['max_range_file'], max_scenes = kwargs['max_scenes'], time_aligned_caption = kwargs['time_aligned_caption'])
    # val_dataset = torch.utils.data.Subset(val_dataset, range(10))

    # DataLoaders creation:
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=train_batch_size,
        shuffle=False
    )

    print(f'length of val dataloader: {len(val_dataloader)}')

     # Latents caching
    cached_data_loader = handle_cache_latents(
        cache_latents, 
        output_dir,
        train_dataloader, 
        train_batch_size, 
        vae,
        cached_latent_dir
    ) 

    if cached_data_loader is not None: 
        train_dataloader = cached_data_loader

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps / accelerator.num_processes)

    max_train_steps = int(num_train_epochs * num_update_steps_per_epoch)
    
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(0.2 * max_train_steps),
    )
    
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, 
        optimizer, 
        train_dataloader, 
        lr_scheduler, 
    )

    # Use Gradient Checkpointing if enabled.
    print(f'gradient checkpoint: {gradient_checkpointing}')
    unet.module._set_gradient_checkpointing(value=gradient_checkpointing)
    # unet._set_gradient_checkpointing(value=gradient_checkpointing)
    
    # Enable VAE slicing to save memory.
    vae.enable_slicing()

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = is_mixed_precision(accelerator)

    # Move text encoders, and VAE to GPU
    models_to_cast = [text_encoder, vae]
    cast_to_gpu_and_type(models_to_cast, accelerator, weight_dtype)

    # Fix noise schedules to predcit light and dark areas if available.
    if not use_offset_noise and rescale_schedule:
        noise_scheduler.betas = enforce_zero_terminal_snr(noise_scheduler.betas)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def finetune_unet(batch, train_encoder=False, val=0):
        nonlocal use_offset_noise
        nonlocal rescale_schedule
        
        # Check if we are training the text encoder
        text_trainable = train_text_encoder
        
        # Unfreeze UNET Layers
        if global_step == 0: 
            already_printed_trainables = False
            unet.train()

        # Convert videos to latent space
        video_pixel_values = batch['video_pixel_values'].to(accelerator.device)
        latents = tensor_to_vae_latent(video_pixel_values.half(), vae)

        if not val:
            prob = np.random.binomial(1, conditioning_dropout_prob)
        else:
            prob = 0

        # Get video length
        video_length = latents.shape[2]
        
        # Sample noise that we'll add to the latents
        use_offset_noise = use_offset_noise and not rescale_schedule
        noise = sample_noise(latents, offset_noise_strength, use_offset_noise)
        bsz = latents.shape[0]

        # Sample a random timestep for each video
        if not val:
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
        else:
            timesteps = torch.tensor([val] * bsz).to(latents.device).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                   
        # *Potentially* Fixes gradient checkpointing training.
        # See: https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
        if kwargs.get('eval_train', False):
            unet.eval()
            text_encoder.eval()

        prompt_ids = batch['prompt_ids']
        if time_aligned_caption:
            # print('inside time aligned')
            all_eh = []
            for pid in prompt_ids:
                text_emb = text_encoder(pid[0].to(accelerator.device))[0][0]
                text_emb = repeat(text_emb, 'b c -> a b c', a = train_data['n_sample_frames'])
                all_eh.append(text_emb.detach().clone())
            encoder_hidden_states = torch.cat(all_eh, dim = 0)
        else:
            token_ids = batch['prompt_ids'][0].to(accelerator.device)
            encoder_hidden_states = text_encoder(token_ids[0])[0]     
        # print(f'encoder hidden state: {encoder_hidden_states.shape}')       
        
        # Get the target for loss depending on the prediction type
        if noise_scheduler.prediction_type == "epsilon":
            target = noise

        elif noise_scheduler.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)

        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
        
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, time_aligned_caption=time_aligned_caption).sample

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss, latents

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    loss, latents = finetune_unet(batch, train_encoder=train_text_encoder)
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                try:
                    accelerator.backward(loss)
                    if max_grad_norm > 0:
                        if accelerator.sync_gradients:
                            params_to_clip = list(filter(lambda x: x.requires_grad, unet.parameters()))
                            accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                            # params_to_clip = list(unet.parameters())
                            # accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                except Exception as e:
                    print(f"An error has occured during backpropogation! {e}") 
                    continue

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]})
                train_loss = 0.0
            
                if global_step % validation_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        print('perform validation!')
                        with torch.no_grad():
                            with accelerator.autocast():
                                unet.eval()
                                unet.module._set_gradient_checkpointing(value=False)
                                # unet._set_gradient_checkpointing(value=False)
                                eval_losses = {1: 0, 10: 0, 100: 0, 500: 0, 800: 0, 950: 0}
                                total = 0
                                for _, batch in tqdm(enumerate(val_dataloader)):
                                    # print("validation")
                                    for k in eval_losses.keys():
                                        loss, latents = finetune_unet(batch, train_encoder=train_text_encoder, val=k)
                                        eval_losses[k] += loss.item()
                                        total += 1
                                with open('eval_losses.txt', 'a') as f:
                                    f.write(str(eval_losses) + "\n")
                                for k in eval_losses.keys():
                                    # print('log wandb')
                                    accelerator.log({f"eval_loss_{k}": eval_losses[k] / total})
                            unet.module._set_gradient_checkpointing(value=gradient_checkpointing)
                            # unet._set_gradient_checkpointing(value=gradient_checkpointing)
                            unet.train()
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            accelerator.log({"training_loss": loss.detach().item()}, step=step)
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            save_pipe(
                    pretrained_model_path, 
                    epoch, 
                    accelerator, 
                    unet, 
                    text_encoder, 
                    vae, 
                    output_dir, 
                    is_checkpoint=True,
                    save_pretrained_model=save_pretrained_model
            )     
            print("Pipe saved!")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_pipe(
                pretrained_model_path, 
                epoch, 
                accelerator, 
                unet, 
                text_encoder, 
                vae, 
                output_dir, 
                is_checkpoint=False,
                save_pretrained_model=save_pretrained_model
        )     
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/my_config.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))