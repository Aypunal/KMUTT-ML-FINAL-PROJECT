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
SCRAPE_FOLDER = "/opt/airflow/ai_image"
TARGET_TOTAL = 100
THREADS = 10

# Focused queries for specific categories with AI generation keywords
QUERIES = [
    # === TREE AI IMAGES ===
    "AI generated tree digital art stable diffusion",
    "artificial tree landscape midjourney created",
    "neural network tree forest AI artwork",
    "synthetic tree nature AI generated art",
    "computer generated forest tree landscape",
    "AI tree illustration digital painting",
    
    # === ANIMAL AI IMAGES ===
    "AI generated animal artwork digital art",
    "artificial intelligence animal portrait",
    "stable diffusion wildlife animal art",
    "midjourney animal creation synthetic",
    "neural network generated animal design",
    "AI wildlife digital illustration",
    
    # === DOG AI IMAGES ===
    "AI generated dog portrait digital art",
    "artificial dog illustration midjourney",
    "stable diffusion dog artwork synthetic",
    "neural network dog design AI created",
    "computer generated dog portrait art",
    "AI dog digital painting illustration",
    
    # === CAT AI IMAGES ===
    "AI generated cat artwork digital art",
    "artificial cat portrait stable diffusion",
    "midjourney cat illustration synthetic",
    "neural network cat design AI created",
    "computer generated cat digital art",
    "AI cat portrait digital illustration",
    
    # === FOOD AI IMAGES ===
    "AI generated food photography digital art",
    "artificial food illustration midjourney",
    "stable diffusion food artwork synthetic",
    "neural network food design AI created",
    "computer generated food digital art",
    "AI food photography illustration",
    
    # === STREET FOOD AI IMAGES ===
    "AI generated street food digital art",
    "artificial street food illustration",
    "stable diffusion street food artwork",
    "midjourney street food creation",
    "neural network street food design",
    "AI street vendor food illustration",
    
    # === COOKED FOOD AI IMAGES ===
    "AI generated cooked food digital art",
    "artificial cooked meal illustration",
    "stable diffusion cooked food artwork",
    "midjourney cooked dish creation",
    "neural network cooked food design",
    "AI restaurant food digital painting",
]

# Category detection for filename prefixes
CATEGORIES = {
    "tree": ["tree", "forest", "nature", "landscape"],
    "animal": ["animal", "wildlife", "creature"],
    "dog": ["dog", "canine", "puppy"],
    "cat": ["cat", "feline", "kitten"],
    "food": ["food", "meal", "dish", "cuisine"],
    "street_food": ["street food", "vendor", "market"],
    "cooked_food": ["cooked", "restaurant", "prepared"]
}

# === Functions ===
def get_image_hash(img):
    """Generate hash for duplicate detection"""
    return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()

def existing_hashes(folder):
    """Get hashes of existing images to avoid duplicates"""
    hashes = set()
    if not os.path.exists(folder):
        return hashes
    
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                with Image.open(os.path.join(folder, f)) as img:
                    hashes.add(get_image_hash(img))
            except:
                continue
    return hashes

def detect_category(query, url="", title=""):
    """Detect which category an image belongs to for filename prefix"""
    text_to_check = f"{query} {url} {title}".lower()
    
    for category, keywords in CATEGORIES.items():
        if any(keyword in text_to_check for keyword in keywords):
            return category
    return "ai"

def is_ai_generated(url, title="", description=""):
    """Check if image is likely AI-generated"""
    ai_indicators = [
        "ai", "artificial", "generated", "synthetic", "midjourney", 
        "dall-e", "stable-diffusion", "neural", "gan", "deepfake",
        "machine-learning", "computer-generated", "algorithmic",
        "lexica", "civitai", "openai", "runway", "firefly", "diffusion"
    ]
    
    text_to_check = f"{url} {title} {description}".lower()
    return any(indicator in text_to_check for indicator in ai_indicators)

def has_target_subject(query, url="", title="", description=""):
    """Check if image contains our target subjects"""
    target_subjects = ["tree", "forest", "animal", "dog", "cat", "food", "meal", "dish"]
    text_to_check = f"{query} {url} {title} {description}".lower()
    return any(subject in text_to_check for subject in target_subjects)

def download_image(result, query):
    """Download and validate image"""
    try:
        # Pre-filter based on content
        if not is_ai_generated(result.get("image", ""), 
                             result.get("title", ""), 
                             result.get("source", "")):
            return None, None, None, None, "Not AI-generated"
        
        if not has_target_subject(query, result.get("image", ""), 
                                result.get("title", ""), 
                                result.get("source", "")):
            return None, None, None, None, "No target subject"
        
        response = requests.get(result["image"], timeout=10, 
                              headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        response.raise_for_status()
        
        # Verify it's a valid image
        img = Image.open(BytesIO(response.content))
        img.verify()
        
        # Reopen for processing
        img = Image.open(BytesIO(response.content))
        
        # Filter out very small images
        if img.size[0] < 300 or img.size[1] < 300:
            return None, None, None, None, "Too small"
        
        # Detect category for filename
        category = detect_category(query, result.get("image", ""), result.get("title", ""))
            
        return img, len(response.content), result["image"], category, "Success"
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"

def scrape_focused_ai_images():
    """Main scraping function for focused AI-generated images"""
    os.makedirs(SCRAPE_FOLDER, exist_ok=True)
    
    hashes = existing_hashes(SCRAPE_FOLDER)
    saved = len(hashes)
    
    # Track saves per category
    category_counts = {cat: 0 for cat in CATEGORIES.keys()}
    category_counts["ai"] = 0
    
    index = saved + 1
    
    print(f"🎯 Focused AI Image Collection")
    print(f"📁 Save folder: {SCRAPE_FOLDER}")
    print(f"🎨 Categories: {', '.join(CATEGORIES.keys())}")
    print(f"🎯 Target: {TARGET_TOTAL} images")
    print(f"📊 Already have: {saved} images")
    print("-" * 60)

    with tqdm(total=TARGET_TOTAL, initial=saved, desc="Collecting AI Images") as pbar, DDGS() as ddgs:
        for query_idx, query in enumerate(QUERIES, 1):
            if saved >= TARGET_TOTAL:
                break

            print(f"\n🔍 [{query_idx}/{len(QUERIES)}] {query}")
            
            try:
                results = ddgs.images(keywords=query, max_results=150)
                print(f"📋 Found {len(results)} results")
                
            except Exception as e:
                print(f"⚠️  Search error: {e}")
                print("😴 Sleeping 20s...")
                time.sleep(20)
                continue

            if not results:
                continue

            # Download images in parallel
            downloaded_count = 0
            with ThreadPoolExecutor(max_workers=THREADS) as executor:
                futures = [executor.submit(download_image, r, query) for r in results]

                for future in futures:
                    if saved >= TARGET_TOTAL:
                        break
                        
                    img, content_size, url, category, status = future.result()
                    
                    if img is None:
                        continue
                    
                    # Check for duplicates
                    img_hash = get_image_hash(img)
                    if img_hash in hashes:
                        continue

                    filename = os.path.basename(url).split("?")[0]
                    if not filename or len(filename) < 3:
                        filename = "image"

                    filename = os.path.basename(url).split("?")[0]
                    if not filename or len(filename) < 3:
                        filename = "image"

                    # Clean filename
                    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                    if not filename.lower().endswith(".jpg"):
                        filename += ".jpg"

                    # Save to main folder
                    save_path = os.path.join(SCRAPE_FOLDER, filename)
                    
                    try:
                        img.convert("RGB").save(save_path, format="JPEG", quality=92)
                        
                        hashes.add(img_hash)
                        saved += 1
                        index += 1
                        downloaded_count += 1
                        category_counts[category] += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        continue

            print(f"✅ Downloaded {downloaded_count} images from this query")
            
            # Show category distribution
            if downloaded_count > 0:
                print("📊 Current distribution:", end=" ")
                for cat, count in category_counts.items():
                    if count > 0:
                        print(f"{cat}:{count}", end=" ")
                print()
            
            # Rate limiting
            if saved < TARGET_TOTAL:
                time.sleep(25)

    print(f"\n🎉 Collection Complete!")
    print(f"📊 Total images: {saved}")
    print(f"📁 Location: {os.path.abspath(SCRAPE_FOLDER)}")
    
    # Final category report
    print(f"\n📊 Final Distribution:")
    for category, count in category_counts.items():
        if count > 0:
            print(f"   {category}: {count} images")

def analyze_collection():
    """Analyze the collected images"""
    if not os.path.exists(SCRAPE_FOLDER):
        print("❌ No collection found")
        return
    
    files = [f for f in os.listdir(SCRAPE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    print(f"\n📈 Collection Analysis:")
    print(f"   📊 Total files: {len(files)}")
    
    if files:
        total_size = sum(os.path.getsize(os.path.join(SCRAPE_FOLDER, f)) for f in files)
        print(f"   💾 Total size: {total_size/(1024*1024):.1f} MB")
        print(f"   📏 Average size: {(total_size/len(files))/(1024):.1f} KB per image")
        
        # Category breakdown by filename prefixes
        categories = {}
        for f in files:
            prefix = f.split('_')[0] if '_' in f else 'unknown'
            categories[prefix] = categories.get(prefix, 0) + 1
        
        print(f"   📂 By category:")
        for cat, count in sorted(categories.items()):
            print(f"      {cat}: {count} images")

if __name__ == "__main__":
    print("🎯 Focused AI-Generated Image Scraper")
    print("=" * 50)
    
    try:
        scrape_focused_ai_images()
        analyze_collection()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")