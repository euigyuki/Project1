import discord
import asyncio
import os
from dotenv import load_dotenv


os.makedirs('downloaded_images_for_midjourney', exist_ok=True)

load_dotenv()
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
CHANNEL_ID = 1275897611806572544 
MIDJOURNEY_BOT_ID = 662267976984297473

# List of prompts to send
prompts = [
    "a woman drinking coffee",
    "a dog playing with a ball",
    "a cat sitting on a windowsill",
    "a person sitting on a bench in a park",
    # Add more prompts here
]

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user}')
        for guild in self.guilds:
            print(f"Checking guild: {guild.name} (ID: {guild.id})")
            channel = guild.get_channel(CHANNEL_ID)
            if channel:
                print(f"Found channel: {channel.name} (ID: {channel.id}) in guild: {guild.name}")
                for prompt in prompts:
                    try:
                        await channel.send(f'/imagine {prompt}')
                        print(f'Sent prompt: {prompt} to channel: {channel.name} (ID: {channel.id})')
                        await asyncio.sleep(1)
                    except Exception as e:
                        print(f'Error sending prompt: {prompt}. Error: {e}')
            else:
                print(f"Could not find the channel with ID: {CHANNEL_ID} in guild: {guild.name} (ID: {guild.id})")

    async def on_message(self, message):
        # Log all incoming messages for debugging
        print(f"Received message from {message.author}: {message.content}")

        if message.author == self.user:
            return

        if message.author.id == MIDJOURNEY_BOT_ID:
            print(f"Message from Midjourney bot: {message.content}")
            for attachment in message.attachments:
                if attachment.url.lower().endswith(('png', 'jpg', 'jpeg', 'gif')):
                    file_path = os.path.join('downloaded_images_for_midjourney', attachment.filename)
                    await attachment.save(file_path)
                    print(f'Saved image: {file_path}')

client = MyClient(intents=intents)
client.run(DISCORD_BOT_TOKEN)