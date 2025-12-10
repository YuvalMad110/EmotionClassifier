"""
Embeddings Module for Emotion Classification
=============================================
Handles loading pretrained embeddings (GloVe, Word2Vec) and creating embedding layers.

Usage:
    from embeddings_ec import EmbeddingLoader
    
    loader = EmbeddingLoader(embedding_dir="./embeddings")
    
    # Load GloVe
    embedding, found, oov = loader.load_embeddings(
        embedding_type="glove",
        vocab=vocab,
        embedding_dim=100,
        trainable=False
    )
    
    # Load Word2Vec
    embedding, found, oov = loader.load_embeddings(
        embedding_type="word2vec",
        vocab=vocab,
        trainable=False
    )
    
    # Create trainable (random init)
    embedding, found, oov = loader.load_embeddings(
        embedding_type="trainable",
        vocab=vocab,
        embedding_dim=100
    )
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .vocabulary_ec import Vocabulary


class EmbeddingLoader:
    """
    Loader for pre-trained word embeddings.
    Supports GloVe, Word2Vec, and trainable embeddings.
    """
    
    def __init__(self, embedding_dir: str = "./embeddings"):
        """
        Initialize embedding loader.
        
        Args:
            embedding_dir: Directory containing embedding files.
                           Expected files:
                           - GloVe: glove.6B.{dim}d.txt (dim = 50, 100, 200, 300)
                           - Word2Vec: GoogleNews-vectors-negative300.bin
        """
        self.embedding_dir = Path(embedding_dir)
    
    # ==================== Vector Loading (Private) ====================
    
    def _load_glove_vectors(self, embedding_dim: int) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Load raw GloVe vectors from file.
        
        Args:
            embedding_dim: Dimension of embeddings (50, 100, 200, or 300)
            
        Returns:
            Tuple of (word_to_vector dict, vector dimension)
        """
        glove_file = self.embedding_dir / f"glove.6B.{embedding_dim}d.txt"
        assert glove_file.exists(), f"GloVe file not found: {glove_file}"
        
        print(f"Loading GloVe vectors from {glove_file}...")
        vectors = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                vectors[word] = vector
        
        print(f"Loaded {len(vectors)} GloVe vectors")
        return vectors, embedding_dim
    
    def _load_word2vec_vectors(self) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Load raw Word2Vec vectors from file.
        Expects GoogleNews-vectors-negative300.bin in embedding_dir.
            
        Returns:
            Tuple of (word_to_vector dict, vector dimension)
        """
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError("gensim is required for Word2Vec. Install with: pip install gensim")
        
        # Check for binary format first, then text
        w2v_path = self.embedding_dir / "GoogleNews-vectors-negative300.bin"        
        assert w2v_path.exists(), f"Word2Vec file not found: {w2v_path}"
        
        print(f"Loading Word2Vec vectors from {w2v_path}...")
        w2v_model = KeyedVectors.load_word2vec_format(str(w2v_path), binary=True)
        
        # Convert to dict
        vectors = {}
        for word in w2v_model.key_to_index:
            vectors[word] = w2v_model[word]
        
        embedding_dim = w2v_model.vector_size
        print(f"Loaded {len(vectors)} Word2Vec vectors (dim={embedding_dim})")
        return vectors, embedding_dim
    
    def _create_embedding_from_vectors(
        self,
        vectors: Dict[str, np.ndarray],
        vocab: Vocabulary,
        embedding_dim: int,
        trainable: bool = False
    ) -> Tuple[torch.nn.Embedding, int, List[str]]:
        """
        Create embedding matrix from pretrained vectors.
        
        Args:
            vectors: Dictionary mapping words to numpy vectors
            vocab: Vocabulary object
            embedding_dim: Dimension of embeddings
            trainable: Whether embeddings should be trainable
            
        Returns:
            Tuple of (embedding layer, found count, list of OOV words)
        """
        # Initialize with small random values for OOV words
        np.random.seed(42)
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
        
        # Set PAD token to zeros
        embedding_matrix[vocab.pad_idx] = np.zeros(embedding_dim)
        
        # Fill in known words and track OOV
        found_count = 0
        oov_words = []
        
        for word, idx in vocab.word2idx.items():
            if word in vectors:
                embedding_matrix[idx] = vectors[word]
                found_count += 1
            elif word not in [Vocabulary.PAD_TOKEN, Vocabulary.UNK_TOKEN]:
                oov_words.append(word)
        
        # Report coverage
        coverage = found_count / len(vocab) * 100
        print(f"Found {found_count}/{len(vocab)} words ({coverage:.2f}% coverage)")
        print(f"OOV words: {len(oov_words)}")
        if oov_words:
            print(f"Sample OOV words: {oov_words[:20]}")
        
        # Create embedding layer
        embedding = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=not trainable,
            padding_idx=vocab.pad_idx
        )
        
        return embedding, found_count, oov_words
    
    def _create_trainable_embedding(
        self,
        vocab: Vocabulary,
        embedding_dim: int
    ) -> torch.nn.Embedding:
        """
        Create a trainable embedding layer with random initialization.
        
        Args:
            vocab: Vocabulary object
            embedding_dim: Dimension of embeddings
            
        Returns:
            Trainable embedding layer
        """
        embedding = torch.nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=embedding_dim,
            padding_idx=vocab.pad_idx
        )
        
        # Initialize with Xavier uniform
        torch.nn.init.xavier_uniform_(embedding.weight.data)
        # Set padding to zeros
        embedding.weight.data[vocab.pad_idx] = torch.zeros(embedding_dim)
        
        print(f"Created trainable embedding layer: {len(vocab)} x {embedding_dim}")
        return embedding
    
    # ==================== Public Interface ====================
    
    def load_embeddings(
        self,
        embedding_type: str,
        vocab: Vocabulary,
        embedding_dim: int = 100,
        trainable: bool = False
    ) -> Tuple[torch.nn.Embedding, int, Optional[List[str]]]:
        """
        Load embeddings of specified type.
        
        Args:
            embedding_type: One of 'glove', 'word2vec', 'trainable'
            vocab: Vocabulary object
            embedding_dim: Dimension of embeddings (used for GloVe and trainable;
                          Word2Vec uses 300 from the file)
            trainable: Whether embeddings should be trainable
            
        Returns:
            Tuple of (embedding layer, found count, OOV words list or None)
        """
        embedding_type = embedding_type.lower()
        
        if embedding_type == "glove":
            vectors, dim = self._load_glove_vectors(embedding_dim)
            return self._create_embedding_from_vectors(vectors, vocab, dim, trainable)
        
        elif embedding_type == "word2vec":
            vectors, dim = self._load_word2vec_vectors()
            return self._create_embedding_from_vectors(vectors, vocab, dim, trainable)
        
        elif embedding_type == "trainable":
            embedding = self._create_trainable_embedding(vocab, embedding_dim)
            return embedding, len(vocab), None
        
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}. Use 'glove', 'word2vec', or 'trainable'")
