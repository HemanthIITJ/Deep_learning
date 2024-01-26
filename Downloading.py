import requests
import re
import os
from urllib.parse import urljoin
#https://deepgenerativemodels.github.io/syllabus.html
# Define the url and the file types
url = "https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/"
file_types = (".pdf", ".pptx")

# Get the html content from the url
response = requests.get(url)
html = response.text

# Find all the links to the files
links = re.findall(r'<a href="(.*?)"', html)

# Create separate folders for each file type
for file_type in file_types:
    folder = file_type[1:]  # Remove the dot
    os.makedirs(folder, exist_ok=True)  # Create the folder if it does not exist

# Download the files and save them in the corresponding folders
for link in links:
    # Check if the link ends with one of the file types
    if link.endswith(file_types):
        # Get the file name from the link
        file_name = link.split("/")[-1]
        # Get the full url of the file
        file_url = urljoin(url, link)
        # Download the file content
        file_content = requests.get(file_url).content
        # Get the file type from the file name
        file_type = os.path.splitext(file_name)[-1].lower()
        # Save the file in the corresponding folder
        with open(file_type[1:] + "/" + file_name, "wb") as f:
            f.write(file_content)
        print(f"Downloaded {file_name} from {file_url}")
