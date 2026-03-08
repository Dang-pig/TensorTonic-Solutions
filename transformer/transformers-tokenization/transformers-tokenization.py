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
        vocabs = set()
        vocabs.add(self.pad_token)
        vocabs.add(self.unk_token)
        vocabs.add(self.bos_token)
        vocabs.add(self.eos_token)
        for txt in texts:
            for word in txt.split(" "):
                vocabs.add(word)
        vocabs = list(vocabs)
        self.vocab_size = len(vocabs)
        ids = [idx for idx in range(len(vocabs))]
        self.word_to_id = dict(zip(vocabs, ids))
        self.id_to_word = dict(zip(ids, vocabs))
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        return [self.word_to_id[txt] if txt in self.word_to_id else self.word_to_id[self.unk_token] for txt in text.split(" ")]
        pass
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join([self.id_to_word[id] for id in ids])
        pass
