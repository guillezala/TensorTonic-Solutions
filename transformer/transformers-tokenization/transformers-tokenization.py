import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id = {
            self.pad_token: 0, 
            self.unk_token: 1, 
            self.bos_token: 2, 
            self.eos_token: 3
        }
        self.id_to_word = {
            0: self.pad_token, 
            1: self.unk_token, 
            2: self.bos_token, 
            3: self.eos_token
        }
        self.vocab_size = len(self.word_to_id)
        full_text = " ".join(texts)
        words = full_text.split(" ")

        words_set = sorted(set(words))

        for word in words_set:
            self.word_to_id[word] = self.vocab_size
            self.id_to_word[self.vocab_size] = word
            self.vocab_size += 1

        
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        if text == "":
            return []
    
        words = text.lower().split(" ")

        token_ids = []

        for word in words:
            if word in self.word_to_id.keys():
                token_ids.append(self.word_to_id[word])
            else:
                token_ids.append(self.word_to_id[self.unk_token])

        return token_ids
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = [self.id_to_word[id] if id < self.vocab_size else self.unk_token for id in ids]
        
        return " ".join(words)