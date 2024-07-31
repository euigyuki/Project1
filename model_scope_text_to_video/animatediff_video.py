import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from moviepy.editor import TextClip, concatenate_videoclips

# Load pre-trained model and tokenizer
model_name = "gpt2"  # You can use any text generation model available on Hugging Face
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Function to create video from text
def create_video_from_text(text, video_filename="output_video.mp4"):
    clips = []
    for sentence in text.split('.'):
        if sentence.strip():
            clip = TextClip(sentence.strip(), fontsize=24, color='white', size=(640, 480))
            clip = clip.set_duration(3)  # Set duration of each sentence display
            clips.append(clip)
    
    video = concatenate_videoclips(clips)
    video.write_videofile(video_filename, fps=24)

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print("Generated Text:", generated_text)

create_video_from_text(generated_text)
