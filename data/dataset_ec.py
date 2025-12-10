"""
Dataset Module for Emotion Classification
==========================================
Handles PyTorch Dataset, DataLoader creation, and data processing pipeline.
"""

import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum

from .vocabulary_ec import Vocabulary
from .embeddings_ec import EmbeddingLoader


class EmotionLabels(Enum):
    """Emotion label mapping"""
    SADNESS = 0
    JOY = 1
    LOVE = 2
    ANGER = 3
    FEAR = 4
    SURPRISE = 5


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for emotion detection.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_length: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text strings
            labels: List of emotion labels
            vocab: Vocabulary object
            max_length: Maximum sequence length for truncation (None = no truncation)
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
        # Pre-encode all texts
        self.encoded_texts = [vocab.encode(text) for text in texts]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], int, int]:
        """
        Get a single item.
        
        Returns:
            Tuple of (encoded_text, label, actual_length)
        """
        encoded = self.encoded_texts[idx].copy()
        label = self.labels[idx]
        
        # Truncate if max_length is set
        if self.max_length is not None and len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        
        actual_length = len(encoded)
        return encoded, label, actual_length


def collate_fn(batch: List[Tuple[List[int], int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function with dynamic padding (pads to max length in batch).
    
    Args:
        batch: List of (encoded_text, label, length) tuples
        
    Returns:
        Tuple of (texts_batch, labels_batch, lengths_batch)
    """
    texts, labels, lengths = zip(*batch)
    
    # Find max length in this batch
    max_len = max(lengths)
    
    # Pad sequences to max_len
    padded_texts = []
    for seq in texts:
        if len(seq) < max_len:
            padded = seq + [0] * (max_len - len(seq))  # 0 is PAD_IDX
        else:
            padded = seq
        padded_texts.append(padded)
    
    texts_tensor = torch.tensor(padded_texts, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return texts_tensor, labels_tensor, lengths_tensor


class DataProcessor:
    """
    Main class for processing emotion detection data.
    """
    
    def __init__(
        self,
        embedding_type: str = "glove",
        embedding_dim: int = 100,
        embedding_trainable: bool = False,
        min_word_freq: int = 1,
        max_seq_length: Optional[int] = None,
        embedding_dir: str = "./embeddings"
    ):
        """
        Initialize data processor.
        
        Args:
            embedding_type: Type of embedding ('glove', 'word2vec', 'trainable')
            embedding_dim: Dimension of embeddings (for GloVe/trainable)
            embedding_trainable: Whether pre-trained embeddings should be trainable
            min_word_freq: Minimum word frequency for vocabulary
            max_seq_length: Maximum sequence length for truncation (None = no truncation)
            embedding_dir: Directory for embedding files
        """
        self.embedding_type = embedding_type.lower()
        self.embedding_dim = embedding_dim
        self.embedding_trainable = embedding_trainable
        self.min_word_freq = min_word_freq
        self.max_seq_length = max_seq_length
        self.embedding_dir = embedding_dir
        
        self.vocab: Optional[Vocabulary] = None
        self.embedding: Optional[torch.nn.Embedding] = None
        self.embedding_loader = EmbeddingLoader(embedding_dir)
        self.oov_words: Optional[List[str]] = None
        
        print(f"DataProcessor initialized:")
        print(f"  Embedding type: {self.embedding_type}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Trainable: {embedding_trainable}")
        print(f"  Min word freq: {min_word_freq}")
        print(f"  Max seq length: {max_seq_length or 'no truncation'}")
    
    @staticmethod
    def load_dataset_csv(file_path: str) -> Tuple[List[str], List[int]]:
        """
        Load data from emotion dataset CSV file.
        Expects columns: 'text', 'label'
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row['text'])
                labels.append(int(row['label']))
        
        print(f"Loaded {len(texts)} samples from {file_path}")
        
        # Print label distribution
        label_counts = Counter(labels)
        print("Label distribution:")
        for label in sorted(label_counts.keys()):
            emotion = EmotionLabels(label).name
            count = label_counts[label]
            percentage = count / len(labels) * 100
            print(f"  {label} ({emotion}): {count} ({percentage:.1f}%)")
        
        return texts, labels
    
    def build_vocabulary(
        self,
        texts: List[str],
        vocab: Optional[Vocabulary] = None
    ) -> Vocabulary:
        """
        Build vocabulary from texts or use provided vocabulary.
        
        Args:
            texts: List of text strings
            vocab: Optional pre-built Vocabulary to use
            
        Returns:
            Vocabulary object
        """
        if vocab is not None:
            self.vocab = vocab
            print(f"Using provided vocabulary with {len(vocab)} words")
        else:
            self.vocab = Vocabulary(min_freq=self.min_word_freq)
            self.vocab.build_from_texts(texts)
        return self.vocab
    
    def load_embeddings(self) -> torch.nn.Embedding:
        """
        Load or create embeddings based on configuration.
        
        Returns:
            Embedding layer
        """
        if self.vocab is None:
            raise ValueError("Vocabulary must be built before loading embeddings")
        
        self.embedding, found_count, self.oov_words = self.embedding_loader.load_embeddings(
            embedding_type=self.embedding_type,
            vocab=self.vocab,
            embedding_dim=self.embedding_dim,
            trainable=self.embedding_trainable
        )
        
        # Update embedding_dim for Word2Vec (it has fixed dim based on file)
        if self.embedding_type == "word2vec":
            self.embedding_dim = self.embedding.weight.shape[1]
        
        return self.embedding
    
    def create_dataset(self, texts: List[str], labels: List[int]) -> EmotionDataset:
        """
        Create PyTorch dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels
            
        Returns:
            EmotionDataset object
        """
        if self.vocab is None:
            raise ValueError("Vocabulary must be built before creating dataset")
        
        return EmotionDataset(
            texts=texts,
            labels=labels,
            vocab=self.vocab,
            max_length=self.max_seq_length
        )
    
    def prepare_data(
        self,
        train_path: str,
        val_path: str,
        test_path: Optional[str] = None,
        batch_size: int = 32,
        shuffle_train: bool = True,
        vocab: Optional[Vocabulary] = None
    ) -> Dict:
        """
        Complete data preparation pipeline.
        
        Args:
            train_path: Path to training CSV
            val_path: Path to validation CSV
            test_path: Optional path to test CSV
            batch_size: Batch size for DataLoaders
            shuffle_train: Whether to shuffle training data
            vocab: Optional pre-built vocabulary to use
            
        Returns:
            Dictionary containing datasets, dataloaders, vocabulary, and embedding
        """
        # Load data
        print("\n" + "=" * 50)
        print("Loading data...")
        print("=" * 50)
        train_texts, train_labels = self.load_dataset_csv(train_path)
        val_texts, val_labels = self.load_dataset_csv(val_path)
        
        test_texts, test_labels = None, None
        if test_path and os.path.exists(test_path):
            test_texts, test_labels = self.load_dataset_csv(test_path)
        
        # Build vocabulary from training data only (or use provided)
        print("\n" + "=" * 50)
        print("Building vocabulary...")
        print("=" * 50)
        self.build_vocabulary(train_texts, vocab=vocab)
        
        # Load embeddings
        print("\n" + "=" * 50)
        print("Loading embeddings...")
        print("=" * 50)
        self.load_embeddings()
        
        # Create datasets
        print("\n" + "=" * 50)
        print("Creating datasets...")
        print("=" * 50)
        train_dataset = self.create_dataset(train_texts, train_labels)
        val_dataset = self.create_dataset(val_texts, val_labels)
        
        test_dataset = None
        if test_texts is not None:
            test_dataset = self.create_dataset(test_texts, test_labels)
        
        # Create DataLoaders
        print("\n" + "=" * 50)
        print("Creating DataLoaders...")
        print("=" * 50)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
        
        print(f"\nData preparation complete!")
        print(f"  Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
        print(f"  Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
        if test_loader:
            print(f"  Test samples: {len(test_dataset)}, batches: {len(test_loader)}")
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'vocab': self.vocab,
            'embedding': self.embedding,
            'num_classes': len(EmotionLabels),
            'max_seq_length': self.max_seq_length,
            'embedding_dim': self.embedding_dim,
            'vocab_size': len(self.vocab),
            'oov_words': self.oov_words
        }


def get_data(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    embedding_type: str = "glove",
    embedding_dim: int = 100,
    embedding_trainable: bool = False,
    min_word_freq: int = 1,
    max_seq_length: Optional[int] = None,
    batch_size: int = 32,
    embedding_dir: str = "./embeddings",
    vocab: Optional[Vocabulary] = None
) -> Dict:
    """
    Main function to get prepared data for training.
    
    This is the primary entry point for the run script.
    
    Args:
        train_path: Path to training CSV file
        val_path: Path to validation CSV file
        test_path: Optional path to test CSV file
        embedding_type: Type of embedding ('glove', 'word2vec', 'trainable')
        embedding_dim: Dimension of embeddings (for GloVe/trainable; Word2Vec uses file's dim)
        embedding_trainable: Whether to train embeddings during model training
        min_word_freq: Minimum word frequency for vocabulary inclusion
        max_seq_length: Maximum sequence length for truncation (None = no truncation)
        batch_size: Batch size for DataLoaders
        embedding_dir: Directory containing embedding files
        vocab: Optional pre-built Vocabulary to use instead of building from train data
        
    Returns:
        Dictionary with all prepared data components:
            - train_loader: DataLoader for training
            - val_loader: DataLoader for validation
            - test_loader: DataLoader for testing (if test_path provided)
            - vocab: Vocabulary object
            - embedding: Pre-initialized embedding layer
            - num_classes: Number of emotion classes (6)
            - max_seq_length: Maximum sequence length used
            - embedding_dim: Embedding dimension
            - vocab_size: Size of vocabulary
            - oov_words: List of out-of-vocabulary words (None for trainable)
    
    Example:
        >>> data = get_data(
        ...     train_path="train.csv",
        ...     val_path="validation.csv",
        ...     embedding_type="glove",
        ...     embedding_dim=100,
        ...     embedding_trainable=True,
        ...     batch_size=32
        ... )
        >>> train_loader = data['train_loader']
        >>> embedding_layer = data['embedding']
    """
    processor = DataProcessor(
        embedding_type=embedding_type,
        embedding_dim=embedding_dim,
        embedding_trainable=embedding_trainable,
        min_word_freq=min_word_freq,
        max_seq_length=max_seq_length,
        embedding_dir=embedding_dir
    )
    
    return processor.prepare_data(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=batch_size,
        vocab=vocab
    )
