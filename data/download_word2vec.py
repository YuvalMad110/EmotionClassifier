"""
Download Word2Vec Embeddings
============================
Downloads the Google News Word2Vec pretrained embeddings (300d, 3M words).

File: GoogleNews-vectors-negative300.bin.gz (~1.5GB compressed, ~3.4GB uncompressed)
Source: https://code.google.com/archive/p/word2vec/

Usage:
    python download_word2vec.py [--output_dir ./embeddings]
"""

import os
import gzip
import shutil
import argparse
from pathlib import Path


def download_word2vec(output_dir: str = "./embeddings") -> None:
    """
    Download and extract Word2Vec embeddings.
    
    Args:
        output_dir: Directory to save the embeddings
    """
    try:
        import gdown
    except ImportError:
        print("gdown is required. Install with: pip install gdown")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gz_path = output_dir / "GoogleNews-vectors-negative300.bin.gz"
    bin_path = output_dir / "GoogleNews-vectors-negative300.bin"
    
    # Check if already exists
    if bin_path.exists():
        print(f"Word2Vec embeddings already exist at: {bin_path}")
        return
    
    # Google Drive file ID
    file_id = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print("Downloading Word2Vec embeddings (~1.5GB)...")
    print("This may take a while depending on your connection.\n")
    
    # Download
    gdown.download(url, str(gz_path), quiet=False)
    
    # Extract
    print("\nExtracting...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(bin_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove compressed file
    os.remove(gz_path)
    
    print(f"\nDone! Word2Vec embeddings saved to: {bin_path}")
    print(f"File size: {bin_path.stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Word2Vec embeddings")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save embeddings (default: ./embeddings)"
    )
    args = parser.parse_args()
    
    download_word2vec(args.output_dir)
