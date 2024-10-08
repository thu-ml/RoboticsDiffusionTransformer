import requests
import os
from tqdm import tqdm

links = []
with open('links.txt', 'r', encoding='utf-8') as file:
    for line in file:
        links.append(line.strip())
        
download_dir = "../datasets/roboset"
os.makedirs(download_dir, exist_ok=True)
        
for link in links:
    filename = os.path.basename(link)
    filepath = os.path.join(download_dir, filename)
    print(f"Downloading {filename} from {link}")

    response = requests.get(link, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    if os.path.exists(filepath):
        local_size = os.path.getsize(filepath)
        if local_size == total_size_in_bytes:
            print(f"{filename} already exists")
            continue
    
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(filepath, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)

    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    print(f"Downloaded {filename}")

print("All files processed.")
