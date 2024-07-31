import torch
from diffusers import DiffusionPipeline
from moviepy.editor import ImageSequenceClip
import numpy as np
import os
import gc
from PIL import Image

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Define the target size for the frames
TARGET_SIZE = (1280, 720)  # Width, Height
prompt  = "A majestic eagle soaring over a mountain range during sunrise."
try:
    if torch.cuda.is_available():
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pipe = pipe.to(device)


    print("Generating video frames...")
    output = pipe(prompt, num_inference_steps=50, num_frames=50)
    print(f"Generated frames: {len(output.frames)}")

    if len(output.frames) == 0:
        raise ValueError("No frames were generated.")

    def process_and_resize_frame(frame):
        frame = np.array(frame)
        print(f"Original frame shape: {frame.shape}")
        if frame.ndim == 4:  # Handle the case of (num_frames, height, width, channels)
            frames = [f for f in frame]  # Split into individual frames
        else:
            frames = [frame]
        
        processed_frames = []
        for f in frames:
            if f.max() <= 1.0:
                f = (f * 255).astype(np.uint8)
            else:
                f = f.astype(np.uint8)
            # Resize the frame
            img = Image.fromarray(f)
            img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
            processed_frames.append(np.array(img_resized))
        
        return processed_frames

    # Process all frames
    video_frames = process_and_resize_frame(output.frames[0])  # Assuming all frames are in the first element

    print(f"Number of video frames after processing: {len(video_frames)}")
    print(f"Shape of first processed frame: {video_frames[0].shape}")

    # Save the first few frames for debugging
    debug_frames_path = "debug_frames"
    os.makedirs(debug_frames_path, exist_ok=True)
    for i, frame in enumerate(video_frames[:5]):
        debug_image = Image.fromarray(frame)
        debug_image.save(os.path.join(debug_frames_path, f"frame_{i}.png"))
    print(f"Saved first few frames for debugging in {debug_frames_path}")

    video_path = "output_video.mp4"

    print("Saving video...")
    clip = ImageSequenceClip(video_frames, fps=8)
    clip.write_videofile(video_path, codec="libx264")

    print(f"Video saved at: {os.path.abspath(video_path)}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()

finally:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()