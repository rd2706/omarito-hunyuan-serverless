#!/usr/bin/env python3
"""
RunPod Serverless Handler for Omarito HunyuanVideo LoRA
"""

import runpod
import torch
import json
import base64
import io
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
import os

# Global variables for model caching
pipe = None
lora_loaded = False

def load_models():
    """Load HunyuanVideo model and Omarito LoRA"""
    global pipe, lora_loaded
    
    if pipe is None:
        print("🔄 Loading HunyuanVideo model...")
        # Import here to avoid loading during container build
        from diffusers import HunyuanVideoPipeline
        
        # Load base model
        pipe = HunyuanVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ HunyuanVideo loaded")
    
    if not lora_loaded:
        print("🔄 Loading Omarito LoRA...")
        # Download LoRA from your HuggingFace repo
        lora_path = hf_hub_download(
            repo_id="Rd2706/omarito-hunyuan-video-lora",
            filename="omarito_lora.safetensors"
        )
        
        # Load LoRA weights
        pipe.load_lora_weights(lora_path)
        lora_loaded = True
        print("✅ Omarito LoRA loaded")

def generate_video(job):
    """Main inference function"""
    try:
        # Load models if needed
        load_models()
        
        # Get job input
        job_input = job["input"]
        prompt = job_input.get("prompt", "sks_omarito walking, cinematic lighting")
        negative_prompt = job_input.get("negative_prompt", "blurry, low quality")
        num_inference_steps = job_input.get("steps", 30)
        guidance_scale = job_input.get("cfg", 7.0)
        height = job_input.get("height", 512)
        width = job_input.get("width", 768)
        num_frames = job_input.get("frames", 49)
        lora_strength = job_input.get("lora_strength", 0.8)
        reference_image = job_input.get("reference_image", None)  # Optional IP-Adapter
        
        print(f"🎬 Generating video: '{prompt}'")
        print(f"📐 Resolution: {width}x{height}, Frames: {num_frames}")
        
        # Set LoRA strength
        pipe.set_adapters(["default"], adapter_weights=[lora_strength])
        
        # Generate video
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=torch.Generator().manual_seed(42)
        )
        
        # Get video tensor
        video_tensor = result.frames[0]  # Shape: [frames, height, width, channels]
        
        # Convert to video file (MP4)
        video_path = "/tmp/generated_video.mp4"
        frames_to_video(video_tensor, video_path)
        
        # Encode video to base64 for return
        with open(video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        
        print("✅ Video generation complete")
        
        return {
            "video_base64": video_b64,
            "prompt": prompt,
            "settings": {
                "steps": num_inference_steps,
                "cfg": guidance_scale,
                "resolution": f"{width}x{height}",
                "frames": num_frames,
                "lora_strength": lora_strength
            }
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "error": str(e)
        }

def frames_to_video(frames_tensor, output_path, fps=8):
    """Convert tensor frames to MP4 video"""
    import cv2
    
    # Convert tensor to numpy
    frames = frames_tensor.cpu().numpy()
    
    # Normalize to 0-255
    frames = (frames * 255).astype(np.uint8)
    
    height, width = frames.shape[1:3]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()

# RunPod handler
runpod.serverless.start({"handler": generate_video})