import re

class SimpleTokenizer:
    def __init__(self):
        # Start with special tokens: Padding and Unknown words
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2
        
    def clean_text(self, text):
        """Lowercases text and removes basic punctuation."""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    def build_vocab(self, texts, min_freq=2):
        """Builds the vocabulary dictionary from a list of sentences."""
        word_counts = {}
        for text in texts:
            words = self.clean_text(text).split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.word2id:
                self.word2id[word] = self.vocab_size
                self.id2word[self.vocab_size] = word
                self.vocab_size += 1
                
    def encode(self, text, max_length):
        """Converts a string to a list of integer IDs and pads it to max_length."""
        words = self.clean_text(text).split()
        
        # Convert words to IDs, using <UNK> (1) if the word isn't in vocab
        token_ids = [self.word2id.get(word, 1) for word in words]
        
        # Truncate if too long
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            
        # Pad with <PAD> (0) if too short
        while len(token_ids) < max_length:
            token_ids.append(self.word2id["<PAD>"])
            
        return token_ids
    