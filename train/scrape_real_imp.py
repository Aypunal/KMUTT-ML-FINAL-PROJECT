import os
import time
import requests
import hashlib
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from duckduckgo_search import DDGS
from tqdm import tqdm

# Settings
SCRAPE_FOLDER = "scraped_temp"
TARGET_TOTAL = 300
THREADS = 10

def get_image_hash(img):
    return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()

def existing_hashes(folder):
    hashes = set()
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
        return img, len(response.content), result["image"]  # RETURN also file size
    except:
        return None, None, None

def download_images(queries, folder, total_target):
    os.makedirs(folder, exist_ok=True)
    hashes = existing_hashes(folder)
    saved = len(hashes)
    index = saved + 1

    pbar = tqdm(total=total_target, desc="Saving Images")

    with DDGS() as ddgs:
        while saved < total_target:
            for query in queries:
                print(f"\nSearching images for: {query}")
                try:
                    results = ddgs.images(keywords=query, max_results=300)
                except Exception as e:
                    print(f"Hit rate limit or error: {e}, sleeping 10s and retrying...")
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

                        # # Validation before saving
                        # if img.width < 300 or img.height < 300:
                        #     print(f"Skipped too small image: {img.width}x{img.height}")
                        #     continue

                        # if content_size is not None and content_size < 50000:  # 50 KB
                        #     print(f"Skipped too small file size: {content_size/1024:.1f} KB")
                        #     continue

                        hashes.add(img_hash)

                        filename = os.path.basename(url).split("?")[0]

                        if "1x1" in filename.lower():
                            print(f"Skipped fake placeholder image: {filename}")
                            continue

                        if not filename.lower().endswith(".jpg"):
                            filename += ".jpg"
                        save_path = os.path.join(folder, f"{index}_{filename}")

                        img = img.convert("RGB")
                        img.save(save_path, format="JPEG")
                        index += 1
                        saved += 1
                        pbar.update(1)

                        if saved >= total_target:
                            break

                if saved >= total_target:
                    break

                time.sleep(15)  # Sleep between queries

    pbar.close()
    print(f"\n✅ Done. Saved {saved} images total.")

if __name__ == "__main__":
    # queries = [
    #     "wild animal photography site:unsplash.com",
    #     "real food photography site:pexels.com",
    #     "real plant photography site:unsplash.com"
    # ]
    queries = [
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
    download_images(queries, SCRAPE_FOLDER, TARGET_TOTAL)
