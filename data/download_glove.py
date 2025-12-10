import os
import urllib.request
import zipfile
import sys

def download_progress(block_num, block_size, total_size):
    """Shows download progress bar."""
    downloaded = block_num * block_size
    percent = 100 * downloaded / total_size
    sys.stdout.write(f"\rDownloading GloVe: {percent:.1f}% [{downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB]")
    sys.stdout.flush()

def setup_glove():
    # 1. Create embeddings directory
    embedding_dir = "/home/yuvalmad/Projects/EmotionClassifier/data/embeddings"
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
        print(f"Created directory: {embedding_dir}")

    # Files to check
    # We usually use the 100d or 300d vectors
    target_file = os.path.join(embedding_dir, "glove.6B.100d.txt")
    
    if os.path.exists(target_file):
        print(f"\nGloVe embeddings already exist at: {target_file}")
        return

    # 2. Download zip
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(embedding_dir, "glove.6B.zip")
    
    print(f"Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, zip_path, download_progress)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading: {e}")
        return

    # 3. Unzip
    print("Extracting files (this may take a moment)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(embedding_dir)
    
    print("Extraction complete!")
    
    # 4. Cleanup (remove zip file to save space)
    os.remove(zip_path)
    print("Cleaned up zip file.")
    print(f"Ready to use! Files are in: {os.path.abspath(embedding_dir)}")

if __name__ == "__main__":
    setup_glove()