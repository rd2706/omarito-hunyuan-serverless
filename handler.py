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
            from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
            
            # Load transformer with bfloat16 (research-backed solution)
            print("📦 Loading transformer with bfloat16...")
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                subfolder="transformer", 
                torch_dtype=torch.bfloat16
            )
            
            # Load pipeline with fp16 and the bfloat16 transformer
            print("📦 Loading pipeline with proper precision settings...")
            pipe = HunyuanVideoPipeline.from_pretrained(
                "hunyuanvideo-community/HunyuanVideo",
                transformer=transformer,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            
            # CRITICAL: Enable VAE tiling (main fix for black output)
            pipe.vae.enable_tiling()
            print("✅ VAE tiling enabled (fixes black video issue)")
            
            # Enable memory optimizations
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
        print("⚠️ DEBUGGING: Skipping LoRA for black video troubleshooting")
        # Skip LoRA loading to test base model
        # lora_path = hf_hub_download(
        #     repo_id="Rd2706/omarito-hunyuan-video-lora",
        #     filename="omarito_lora.safetensors"
        # )
        # pipe.load_lora_weights(lora_path)
        lora_loaded = True
        print("⚠️ Base model only (no LoRA)")

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
        
        print(f"🎬 Generating video (BASE MODEL ONLY): '{prompt}'")
        print(f"📐 Resolution: {width}x{height}, Frames: {num_frames}")
        
        # Skip LoRA adapter settings for debugging
        print("⚠️ DEBUGGING: No LoRA adapters will be used")
        
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
        
        # Get video tensor - HunyuanVideo returns HunyuanVideoPipelineOutput
        print(f"🔍 Debug: result type: {type(result)}")
        print(f"🔍 Debug: result attributes: {dir(result)}")
        
        if hasattr(result, 'frames') and result.frames is not None:
            video_tensor = result.frames
            print(f"🔍 Debug: Using result.frames, type: {type(video_tensor)}")
        elif hasattr(result, 'videos') and result.videos is not None:
            video_tensor = result.videos
            print(f"🔍 Debug: Using result.videos, type: {type(video_tensor)}")
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            video_tensor = result[0]  # Take first element if it's a tuple/list
            print(f"🔍 Debug: Using result[0], type: {type(video_tensor)}")
        else:
            print(f"🔍 Debug: Using result directly, type: {type(result)}")
            video_tensor = result
            
        print(f"🔍 Debug: Final video_tensor type: {type(video_tensor)}")
        if hasattr(video_tensor, 'shape'):
            print(f"🔍 Debug: video_tensor shape: {video_tensor.shape}")
        
        # Save to worker's temporary directory
        videos_dir = "/tmp"
        
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
        
        # Read video file and encode as base64 for download
        with open(video_path, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode('utf-8')
        
        print(f"✅ Video generated: {video_filename} ({file_size_mb:.1f} MB)")
        
        return {
            "video_filename": video_filename,
            "video_base64": video_base64,
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
    """Convert HunyuanVideo tensor frames to MP4 video"""
    import cv2
    import torch
    
    print(f"🔍 Debug: frames_tensor type: {type(frames_tensor)}")
    
    # Handle HunyuanVideo output format: (batch_size, num_frames, channels, height, width)
    if isinstance(frames_tensor, list):
        frames_tensor = frames_tensor[0]  # Take first batch from list
    
    # Ensure it's a tensor and move to CPU
    if isinstance(frames_tensor, torch.Tensor):
        frames = frames_tensor.cpu().numpy()
    else:
        frames = np.array(frames_tensor)
    
    print(f"🔍 Debug: Initial frames shape: {frames.shape}")
    print(f"🔍 Debug: frames dtype: {frames.dtype}, min/max: {frames.min():.3f}/{frames.max():.3f}")
    
    # HunyuanVideo format: (batch_size, num_frames, channels=3, height, width)
    if len(frames.shape) == 5:  # [batch, num_frames, channels, height, width]
        frames = frames[0]  # Take first batch: [num_frames, channels, height, width]
        print(f"🔍 Debug: After batch selection: {frames.shape}")
    
    # Check if already in correct format: [num_frames, height, width, channels]
    if len(frames.shape) == 4:
        if frames.shape[3] == 3:  # [num_frames, height, width, 3] - already correct!
            print(f"🔍 Debug: Tensor already in correct format: {frames.shape}")
        elif frames.shape[1] == 3:  # [num_frames, 3, height, width] - needs transpose
            frames = frames.transpose(0, 2, 3, 1)  # -> [num_frames, height, width, 3]
            print(f"🔍 Debug: After transpose: {frames.shape}")
        else:
            raise ValueError(f"Unexpected frame tensor shape: {frames.shape}")
    else:
        raise ValueError(f"Expected 4D tensor, got shape: {frames.shape}")
    
    # Validate we have correct shape for OpenCV
    if len(frames.shape) != 4 or frames.shape[3] != 3:
        raise ValueError(f"Expected [num_frames, height, width, 3], got {frames.shape}")
    
    # Normalize to 0-255 range for OpenCV
    if frames.dtype in [np.float32, np.float64]:
        if frames.max() <= 1.0:
            frames = (frames * 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    elif frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    
    print(f"🔍 Debug: Final frames shape: {frames.shape}, dtype: {frames.dtype}")
    
    height, width = frames.shape[1:3]
    num_frames = frames.shape[0]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    
    # Write frames
    for i, frame in enumerate(frames):
        # Validate frame shape before OpenCV conversion
        if frame.shape != (height, width, 3):
            print(f"⚠️ Warning: Frame {i} has invalid shape {frame.shape}")
            continue
            
        # Convert RGB to BGR for OpenCV (HunyuanVideo outputs RGB)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"✅ Video saved: {output_path} ({num_frames} frames, {width}x{height})")

# RunPod handler
runpod.serverless.start({"handler": generate_video})