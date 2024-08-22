import requests
import time
import os

# Your TTApi API key
API_KEY = 'e8872c9f-d049-319c-f98b-c1aeb1585d32'

# Directory to save images
SAVE_DIR = 'downloaded_images_for_midjourney'

# Ensure the directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_image(prompt):
    endpoint = "https://api.ttapi.io/midjourney/v1/imagine"
    headers = {
        'TT-API-KEY': API_KEY,
        'Content-Type': 'application/json'
    }
    data = {
        "prompt": prompt,
        "mode": "fast"  # You can change this to "relax" or "turbo" if needed
    }
    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except ValueError:
        print("Response content is not valid JSON")
        return None

def fetch_image(job_id):
    endpoint = "https://api.ttapi.io/midjourney/v1/fetch"
    headers = {
        'TT-API-KEY': API_KEY
    }
    data = {
        "jobId": job_id
    }
    try:
        response = requests.post(endpoint, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Fetch request failed: {e}")
        return None

def download_image(image_url, save_path):
    try:
        image_data = requests.get(image_url).content
        with open(save_path, 'wb') as f:
            f.write(image_data)
        print(f"Saved Image: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")

def main():
    prompts = [
        "a pretty lady at the beach",
        "a futuristic cityscape",
        "a serene mountain landscape"
    ]

    for i, prompt in enumerate(prompts):
        response_data = generate_image(prompt)
        if response_data and response_data.get("status") == "SUCCESS":
            job_id = response_data['data']['jobId']
            print(f"Job {i+1} submitted successfully with jobId: {job_id}")
            
            # Wait for the job to complete
            time.sleep(90)  # Adjust as needed based on expected completion time
            
            # Fetch the result
            fetch_data = fetch_image(job_id)
            if fetch_data and fetch_data.get("status") == "SUCCESS":
                print(f"Image fetched successfully for jobId '{job_id}'")
                print(f"Data: {fetch_data['data']}")

                # Check for image URLs in discordImage and cdnImage
                discord_image_url = fetch_data['data'].get('discordImage')
                cdn_image_url = fetch_data['data'].get('cdnImage')
                
                if discord_image_url:
                    image_path = os.path.join(SAVE_DIR, f'image_{i+1}_discord.png')
                    download_image(discord_image_url, image_path)
                
                if cdn_image_url:
                    image_path = os.path.join(SAVE_DIR, f'image_{i+1}_cdn.png')
                    download_image(cdn_image_url, image_path)

            else:
                print(f"Failed to fetch image for jobId '{job_id}'")
        else:
            print(f"Failed to generate image for prompt '{prompt}'")
        time.sleep(30)  # To avoid hitting rate limits

if __name__ == "__main__":
    main()