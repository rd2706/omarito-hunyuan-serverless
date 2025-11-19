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
        
        # Import with torch 2.5.1 compatibility
        import torch
        import os
        import gc
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🧹 GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        try:
            from diffusers import HunyuanVideoPipeline
            
            # Load with PyTorch 2.5.1 compatible settings + memory optimization
            print("📦 Loading model with memory optimization...")
            pipe = HunyuanVideoPipeline.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                torch_dtype=torch.bfloat16,  # Use bfloat16 for better compatibility
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # Memory optimizations
                variant="fp16",  # Use smaller variant if available
                use_safetensors=True
            )
            
            # Enable memory efficient attention
            pipe.enable_attention_slicing(1)
            pipe.enable_model_cpu_offload()
            
            print("✅ HunyuanVideo loaded with memory optimization")
            
        except Exception as e:
            print(f"❌ Error loading HunyuanVideo: {e}")
            # Fallback with different precision
            try:
                print("🔄 Trying fallback with float16...")
                pipe = HunyuanVideoPipeline.from_pretrained(
                    "hunyuanvideo-community/HunyuanVideo",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                pipe.enable_attention_slicing(1)
                pipe.enable_model_cpu_offload()
                pipe = pipe.to("cuda")
                print("✅ HunyuanVideo loaded (float16 fallback)")
            except Exception as e2:
                print(f"❌ All fallbacks failed: {e2}")
                raise e2
    
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
        
        # Set LoRA strength - use correct adapter name
        try:
            pipe.set_adapters(["default_0"], adapter_weights=[lora_strength])
        except Exception as e:
            print(f"⚠️ Adapter error, trying fallback: {e}")
            # Fallback: try default or skip adapter setting
            try:
                pipe.set_adapters(["default"], adapter_weights=[lora_strength])
            except:
                print("⚠️ Using pipeline without adapter strength setting")
        
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
        
        # Get video tensor - HunyuanVideo returns different format
        if hasattr(result, 'frames'):
            video_tensor = result.frames
        elif hasattr(result, 'videos'):
            video_tensor = result.videos
        else:
            # Fallback: try to get the video from result directly
            video_tensor = result
        
        # Save to local VPS videos directory
        videos_dir = "/home/reda/dev/runpod/videos"
        os.makedirs(videos_dir, exist_ok=True)
        
        # Generate unique filename
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"omarito_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
        video_path = f"{videos_dir}/{video_filename}"
        
        # Convert to video file (MP4)
        frames_to_video(video_tensor, video_path)
        
        # Clear GPU memory after generation
        del result
        del video_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Get file size
        file_size_mb = os.path.getsize(video_path) / 1024 / 1024
        
        print(f"✅ Video saved to VPS: {video_path} ({file_size_mb:.1f} MB)")
        
        return {
            "video_path": video_path,
            "video_filename": video_filename,
            "file_size_mb": round(file_size_mb, 1),
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
    import torch
    
    # Handle different output formats from HunyuanVideo
    if isinstance(frames_tensor, list):
        # If it's a list, take the first element
        frames_tensor = frames_tensor[0]
    
    # Ensure it's a tensor and move to CPU
    if isinstance(frames_tensor, torch.Tensor):
        frames = frames_tensor.cpu().numpy()
    else:
        frames = np.array(frames_tensor)
    
    # Handle different tensor shapes
    if len(frames.shape) == 5:  # [batch, frames, channels, height, width]
        frames = frames[0]  # Take first batch
        frames = frames.transpose(0, 2, 3, 1)  # [frames, height, width, channels]
    elif len(frames.shape) == 4:  # [frames, channels, height, width]
        frames = frames.transpose(0, 2, 3, 1)  # [frames, height, width, channels]
    
    # Normalize to 0-255
    if frames.max() <= 1.0:
        frames = (frames * 255).astype(np.uint8)
    else:
        frames = frames.astype(np.uint8)
    
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