import torch
from diffusers import DiffusionPipeline
from moviepy.editor import ImageSequenceClip, concatenate_videoclips
import numpy as np
import os
import gc
from PIL import Image

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def generate_video_chunk(pipe, prompt, negative_prompt, num_frames, start_frame=0):
    output = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        num_frames=num_frames,
        height=384,
        width=384,
        guidance_scale=7.5,
    )
    
    frames = process_frame(output.frames[0])
    return frames

def process_frame(frame):
    frame = np.array(frame)
    if frame.ndim == 4:
        frames = [f for f in frame]
    else:
        frames = [frame]
    
    processed_frames = []
    for f in frames:
        if f.max() <= 1.0:
            f = (f * 255).astype(np.uint8)
        else:
            f = f.astype(np.uint8)
        processed_frames.append(f)
    
    return processed_frames

try:
    if torch.cuda.is_available():
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pipe = pipe.to(device)

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xformers attention enabled.")
    except Exception as e:
        print(f"xformers not available, falling back to attention slicing. Error: {e}")
        pipe.enable_attention_slicing()

    prompt = "a woman is drinking coffee in a cafe, high quality, detailed, cinematic lighting"
    negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy"

    # Generate video in chunks
    chunk_size = 24  # 1 second chunks
    total_frames = 120  # 5 seconds total
    video_chunks = []

    for start_frame in range(0, total_frames, chunk_size):
        print(f"Generating chunk starting at frame {start_frame}")
        chunk_frames = generate_video_chunk(pipe, prompt, negative_prompt, chunk_size, start_frame)
        video_chunks.append(ImageSequenceClip(chunk_frames, fps=24))
        
        # Clear CUDA cache after each chunk
        torch.cuda.empty_cache()
        gc.collect()

    # Combine video chunks
    final_clip = concatenate_videoclips(video_chunks)

    video_path = "output_video.mp4"
    print("Saving video...")
    final_clip.write_videofile(video_path, codec="libx264", bitrate="8000k")

    print(f"Video saved at: {os.path.abspath(video_path)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()