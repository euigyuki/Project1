import requests

def download_csv_and_pick_first_of_every_five(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Split the content by lines
        lines = response.text.splitlines()

        # Select the first line from every group of 5 lines
        selected_lines = [lines[i] for i in range(1, len(lines), 5)]

        return selected_lines

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the file: {e}")

# Example usage
csv_url = 'https://raw.githubusercontent.com/cmward/text-scene/master/text_scene/datafiles/sentences.csv'
selected_lines = download_csv_and_pick_first_of_every_five(csv_url)

# Print the selected lines for verification
print("Selected lines:")
print(len(selected_lines))
# for line in selected_lines[:10]:
#     print(line)