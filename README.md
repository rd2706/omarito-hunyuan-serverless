# Omarito HunyuanVideo LoRA - RunPod Serverless

Serverless deployment of the Omarito character LoRA for HunyuanVideo generation on RunPod.

## Features

- **Character**: Omarito LoRA trained for 1500 steps
- **Base Model**: HunyuanVideo (13B parameters)
- **Trigger Word**: `sks_omarito`
- **Output**: MP4 videos up to 61 frames (5 seconds)

## API Usage

### Input Parameters

```json
{
  "prompt": "sks_omarito walking in garden, cinematic lighting, 4k",
  "negative_prompt": "blurry, low quality, distorted",
  "steps": 30,
  "cfg": 7.0,
  "height": 512,
  "width": 768,
  "frames": 49,
  "lora_strength": 0.8
}
```

### Response

```json
{
  "video_base64": "base64_encoded_mp4_video",
  "prompt": "sks_omarito walking in garden, cinematic lighting, 4k",
  "settings": {
    "steps": 30,
    "cfg": 7.0,
    "resolution": "768x512",
    "frames": 49,
    "lora_strength": 0.8
  }
}
```

## Example Prompts

- `sks_omarito smiling at camera, natural lighting`
- `sks_omarito dancing, happy expression, professional lighting`
- `sks_omarito walking in garden, cinematic lighting, 4k`
- `sks_omarito turning head, close-up portrait, soft lighting`

## Deployment

1. Push this repo to GitHub
2. Create RunPod Serverless endpoint
3. Set GitHub repo URL
4. Deploy and test

## Cost Estimate

- **Cold start**: ~60 seconds (model loading)
- **Inference**: ~30-60 seconds per video
- **Cost**: ~$0.20-0.50 per video (depending on settings)

## Model Details

- **LoRA Source**: https://huggingface.co/Rd2706/omarito-hunyuan-video-lora
- **Training Steps**: 1500
- **Base Model**: HunyuanVideo
- **Model Size**: 315MB LoRA + 25GB base model