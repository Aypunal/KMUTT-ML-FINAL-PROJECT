import os
import shutil
import re
import random

# Settings
SCRAPE_FOLDER = "scraped_temp"
FINAL_FOLDER = "real_images"
LOG_FILE = "used_images.log"
DAYS = 7
IMAGES_PER_DAY = 10

def read_used_images():
    if not os.path.exists(LOG_FILE):
        return set()
    with open(LOG_FILE, "r") as f:
        return set(line.strip() for line in f.readlines())

def write_used_images(used_images):
    with open(LOG_FILE, "w") as f:
        for img in used_images:
            f.write(img + "\n")

def get_sorted_images(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    def extract_num(filename):
        match = re.match(r"(\d+)_", filename)
        return int(match.group(1)) if match else 0
    return sorted(files, key=extract_num)

def clear_day_folders(final_folder, days):
    for day in range(1, days + 1):
        day_folder = os.path.join(final_folder, f"day{day}")
        if os.path.exists(day_folder):
            for f in os.listdir(day_folder):
                file_path = os.path.join(day_folder, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

def organize_images_random(scrape_folder, final_folder, days, images_per_day):
    all_images = get_sorted_images(scrape_folder)
    used_images = read_used_images()
    available_images = [img for img in all_images if img not in used_images]

    total_needed = days * images_per_day

    if len(available_images) < total_needed:
        print(f"Not enough images. Needed {total_needed}, available {len(available_images)}")
        return

    selected_images = random.sample(available_images, total_needed)

    clear_day_folders(final_folder, days)

    idx = 0
    for day in range(1, days + 1):
        day_folder = os.path.join(final_folder, f"day{day}")
        os.makedirs(day_folder, exist_ok=True)

        for _ in range(images_per_day):
            src = os.path.join(scrape_folder, selected_images[idx])
            dest = os.path.join(day_folder, selected_images[idx])
            shutil.copy(src, dest)
            print(f"Copied {src} → {dest}")
            idx += 1

    # Update the log
    used_images.update(selected_images)
    write_used_images(used_images)

if __name__ == "__main__":
    organize_images_random(SCRAPE_FOLDER, FINAL_FOLDER, DAYS, IMAGES_PER_DAY)
