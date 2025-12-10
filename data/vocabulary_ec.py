"""
Vocabulary Module for Emotion Classification
=============================================
Handles word-to-index mapping and tokenization.
"""

import re
from collections import Counter
from typing import Dict, List


class Vocabulary:
    """
    Vocabulary class for mapping words to indices and vice versa.
    """
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self, min_freq: int = 1):
        """
        Initialize vocabulary.
        
        Args:
            min_freq: Minimum frequency for a word to be included in vocabulary
        """
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self._built = False
        
        # Add special tokens
        self.word2idx[self.PAD_TOKEN] = 0
        self.word2idx[self.UNK_TOKEN] = 1
        self.idx2word[0] = self.PAD_TOKEN
        self.idx2word[1] = self.UNK_TOKEN
        
    def build_from_texts(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings
        """
        # Count word frequencies
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # Add words that meet minimum frequency
        idx = 2  # Start after PAD and UNK
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        self._built = True
        print(f"Vocabulary built with {len(self.word2idx)} words (min_freq={self.min_freq})")
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9'\s]", " ", text)
        tokens = text.split()
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of word indices
        """
        tokens = self.tokenize(text)
        return [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        Decode indices back to words.
        
        Args:
            indices: List of word indices
            
        Returns:
            List of words
        """
        return [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.UNK_TOKEN]
