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
intents.message_content = True

class MyClient(discord.Client):
    async def on_ready(self):
        print(f'Logged in as {self.user}')
        # Sync the command tree to make sure commands are available
        await self.tree.sync()

    async def setup_hook(self):
        # Initialize the command tree
        self.tree = discord.app_commands.CommandTree(self)
        # Add the command to the tree
        self.tree.add_command(self.send_prompt)

    @discord.app_commands.command(name='send_prompt', description='Send a prompt to Midjourney')
    async def send_prompt(self, interaction: discord.Interaction, prompt: str):
        channel = self.get_channel(CHANNEL_ID)
        if channel:
            await channel.send(f'/imagine {prompt}')
            await interaction.response.send_message(f'Sent prompt: {prompt} to Midjourney!', ephemeral=True)
        else:
            await interaction.response.send_message('Channel not found.', ephemeral=True)

client = MyClient(intents=intents)
client.run(DISCORD_BOT_TOKEN)

