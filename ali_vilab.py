import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler
from moviepy.editor import ImageSequenceClip
import numpy as np
import os

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the AnimateDiff-Lightning model
model_id = "ByteDance/AnimateDiff-Lightning"
pipe = AnimateDiffPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

def generate_video(prompt, output_path="output_video.mp4", num_frames=16, num_inference_steps=25):
    print(f"Generating video for prompt: '{prompt}'")
    
    # Generate the video frames
    output = pipe(prompt, num_inference_steps=num_inference_steps, num_frames=num_frames)

    # Process the frames
    video_frames = []
    for frame in output.frames:
        # Convert to numpy array and ensure it's in uint8 format
        frame = np.array(frame)
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)
        video_frames.append(frame)

    # Create and save the video
    clip = ImageSequenceClip(video_frames, fps=8)
    clip.write_videofile(output_path, codec="libx264")
    
    print(f"Video saved at: {os.path.abspath(output_path)}")

# Example usage
prompt = "A majestic eagle soaring over a mountain range during sunrise."
generate_video(prompt)