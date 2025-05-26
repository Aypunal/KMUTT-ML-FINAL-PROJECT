import os
import time
import requests
import hashlib
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from duckduckgo_search import DDGS
from tqdm import tqdm
from datetime import datetime

# === Settings ===
SCRAPE_FOLDER = "/opt/airflow/real_image"
TARGET_TOTAL = 100
THREADS = 10

QUERIES = [
    # Animal photography
    "wildlife animal photography site:unsplash.com",
    "zoo animal photo DSLR site:pexels.com",
    "pet dog photography site:unsplash.com",
    "cat portrait real photo site:pexels.com",

    # Food photography
    "cooked food photography site:pexels.com",
    "street food real photography site:unsplash.com",
    "dessert photo DSLR site:pexels.com",
    "fruit photography site:unsplash.com",

    # Plant photography
    "real plant photography site:unsplash.com",
    "flower macro photography site:pexels.com",
    "tree landscape photography site:unsplash.com",
    "forest photography real site:pexels.com",
]

# === Functions ===
def get_image_hash(img):
    return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()

def existing_hashes(folder):
    hashes = set()
    if not os.path.exists(folder):
        return hashes
    for f in os.listdir(folder):
        if f.lower().endswith(".jpg"):
            try:
                with Image.open(os.path.join(folder, f)) as img:
                    hashes.add(get_image_hash(img))
            except:
                continue
    return hashes

def download_image(result):
    try:
        response = requests.get(result["image"], timeout=5)
        img = Image.open(BytesIO(response.content))
        img.verify()
        img = Image.open(BytesIO(response.content))
        return img, len(response.content), result["image"]
    except:
        return None, None, None

def scrape_images():
    os.makedirs(SCRAPE_FOLDER, exist_ok=True)
    hashes = existing_hashes(SCRAPE_FOLDER)
    saved = len(hashes)
    index = saved + 1

    with tqdm(total=TARGET_TOTAL, desc="Saving Images") as pbar, DDGS() as ddgs:
        for query in QUERIES:
            if saved >= TARGET_TOTAL:
                break

            print(f"\nSearching images for: {query}")
            try:
                results = ddgs.images(keywords=query, max_results=300)
            except Exception as e:
                print(f"Hit rate limit or error: {e}, sleeping 10s...")
                time.sleep(10)
                continue

            print(f"Got {len(results)} results for {query}")

            with ThreadPoolExecutor(max_workers=THREADS) as executor:
                futures = [executor.submit(download_image, r) for r in results]

                for future in futures:
                    img, content_size, url = future.result()
                    if img is None:
                        continue
                    img_hash = get_image_hash(img)
                    if img_hash in hashes:
                        continue

                    filename = os.path.basename(url).split("?")[0]
                    if "1x1" in filename.lower():
                        print(f"Skipped fake placeholder image: {filename}")
                        continue

                    # Clean filename
                    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                    if not filename.lower().endswith(".jpg"):
                        filename += ".jpg"

                    save_path = os.path.join(SCRAPE_FOLDER, filename)

                    img = img.convert("RGB")
                    img.save(save_path, format="JPEG")
                    index += 1
                    saved += 1
                    pbar.update(1)

                if saved >= TARGET_TOTAL:
                        break

            time.sleep(15)  # avoid getting blocked

    print(f"\n✅ Done. Saved {saved} images total.")
    
if __name__ == "__main__":
    scrape_images()
