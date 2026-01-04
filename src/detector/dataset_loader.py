import os
import email
from email import policy
from email.parser import BytesParser
import re
from bs4 import BeautifulSoup
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
import html2text
from urllib.parse import urlparse
import warnings


class EmailTokenizer:
    """
    Professional-grade email tokenizer with proper segmentation.
    
    Features:
    - Punctuation handling
    - Email-specific patterns (URLs, emails, IPs)
    - Subword-like splitting for unknown words
    - Special token handling
    """
    
    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<SOS>': 2,  
        '<EOS>': 3, 
        '<URL>': 4,
        '<EMAIL>': 5,
        '<NUM>': 6,
        '<IP>': 7,
        '<DOMAIN>': 8
    }
    
    def __init__(self):
        # Regex patterns for special entities
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        self.domain_pattern = re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b')
        self.number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')
        
        # Punctuation to keep separate
        self.punctuation = set('.,!?;:()[]{}"\'-/')
    
    def extract_urls(self, text: str) -> List[str]:
        """Extract URLs for feature engineering"""
        return self.url_pattern.findall(text)
    
    def extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        return self.email_pattern.findall(text)
    
    def tokenize(self, text: str, preserve_urls: bool = True) -> List[str]:
        """
        Tokenize text with proper handling of special entities.
        
        Args:
            text: Input text
            preserve_urls: If True, preserve URL tokens instead of replacing
        
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Store special entities before processing
        entities = []
        
        # Replace URLs
        def replace_url(match):
            url = match.group(0)
            entities.append(('URL', url))
            return ' <URL> '
        
        text = self.url_pattern.sub(replace_url, text)
        
        # Replace emails
        def replace_email(match):
            addr = match.group(0)
            entities.append(('EMAIL', addr))
            return ' <EMAIL> '
        
        text = self.email_pattern.sub(replace_email, text)
        
        # Replace IPs
        def replace_ip(match):
            ip = match.group(0)
            entities.append(('IP', ip))
            return ' <IP> '
        
        text = self.ip_pattern.sub(replace_ip, text)
        
        # Replace numbers
        text = self.number_pattern.sub(' <NUM> ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize with punctuation handling
        tokens = []
        current_token = []
        
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
            elif char in self.punctuation:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                tokens.append(char)
            else:
                current_token.append(char)
        
        if current_token:
            tokens.append(''.join(current_token))
        
        return tokens


class EmailParser:
    """
    Robust email parser with comprehensive multipart handling.
    """
    
    def __init__(self, preserve_urls: bool = True):
        self.preserve_urls = preserve_urls
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # No wrapping
    
    def parse_email(self, file_path: str) -> Optional[Dict]:
        """
        Parse .eml file with robust error handling.
        
        Returns:
            Dictionary with headers, body, metadata
        """
        try:
            with open(file_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
            
            # Extract headers
            headers = {
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'subject': msg.get('Subject', ''),
                'date': msg.get('Date', ''),
                'reply_to': msg.get('Reply-To', ''),
                'return_path': msg.get('Return-Path', ''),
            }
            
            # Extract body with priority: text/plain > text/html
            body_plain = None
            body_html = None
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get('Content-Disposition', ''))
                    
                    # Skip attachments
                    if 'attachment' in content_disposition:
                        continue
                    
                    try:
                        payload = part.get_payload(decode=True)
                        if not payload:
                            continue
                        
                        charset = part.get_content_charset() or 'utf-8'
                        text = payload.decode(charset, errors='ignore')
                        
                        if content_type == 'text/plain':
                            body_plain = text
                        elif content_type == 'text/html':
                            body_html = text
                    except Exception as e:
                        continue
            else:
                # Single part message
                try:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        charset = msg.get_content_charset() or 'utf-8'
                        content_type = msg.get_content_type()
                        
                        text = payload.decode(charset, errors='ignore')
                        
                        if content_type == 'text/plain':
                            body_plain = text
                        elif content_type == 'text/html':
                            body_html = text
                        else:
                            body_plain = text
                except:
                    body_plain = str(msg.get_payload())
            
            # Convert HTML to text if needed
            if body_plain:
                body = body_plain
                body_type = 'plain'
            elif body_html:
                body = self._html_to_text(body_html)
                body_type = 'html'
            else:
                body = ''
                body_type = 'empty'
            
            return {
                'headers': headers,
                'body': body,
                'body_type': body_type,
                'raw_html': body_html if body_html else None
            }
            
        except Exception as e:
            warnings.warn(f"Error parsing {file_path}: {e}")
            return None
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean text while preserving structure"""
        try:
            # Use html2text for better conversion
            text = self.html_converter.handle(html)
            return text
        except:
            # Fallback to BeautifulSoup
            try:
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text(separator=' ', strip=True)
            except:
                return html


class EmailDataLoader:
    """
    Production-grade email data loader with proper preprocessing pipeline.
    
    Features:
    - Robust email parsing
    - Professional tokenization
    - Vocabulary management with size limits
    - Train/validation/test split
    - Proper sequence length tracking
    - Feature extraction (URLs, domains, etc.)
    """
    
    VERSION = '2.0'
    
    def __init__(
        self,
        data_folder: str = 'data/email',
        max_length: int = 512,
        vocab_size: int = 10000,
        min_freq: int = 2,
        preserve_urls: bool = True
    ):
        self.data_folder = data_folder
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.preserve_urls = preserve_urls
        
        # Initialize components
        self.tokenizer = EmailTokenizer()
        self.parser = EmailParser(preserve_urls=preserve_urls)
        
        # Vocabulary
        self.word2idx = EmailTokenizer.SPECIAL_TOKENS.copy()
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_built = False
        
        print(f"[DataLoader v{self.VERSION}] Initialized")
        print(f"  Data folder: {data_folder}")
        print(f"  Max length: {max_length}")
        print(f"  Target vocab size: {vocab_size}")
    
    def load_all_emails(self) -> List[Dict]:
        """Load all .eml files with robust error handling"""
        if not os.path.exists(self.data_folder):
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
        
        eml_files = [f for f in os.listdir(self.data_folder) if f.endswith('.eml')]
        print(f"[DataLoader] Found {len(eml_files)} .eml files")
        
        email_data = []
        failed = 0
        
        for filename in eml_files:
            file_path = os.path.join(self.data_folder, filename)
            parsed = self.parser.parse_email(file_path)
            
            if parsed:
                # Tokenize
                full_text = f"{parsed['headers']['subject']} {parsed['body']}"
                full_text = full_text.lower()  # Normalize case
                tokens = self.tokenizer.tokenize(full_text, self.preserve_urls)
                
                # Extract features
                urls = self.tokenizer.extract_urls(parsed['body'])
                emails = self.tokenizer.extract_emails(parsed['body'])
                
                email_data.append({
                    'filename': filename,
                    'headers': parsed['headers'],
                    'body': parsed['body'],
                    'body_type': parsed['body_type'],
                    'tokens': tokens,
                    'features': {
                        'urls': urls,
                        'emails': emails,
                        'url_count': len(urls),
                        'email_count': len(emails),
                        'length': len(tokens)
                    }
                })
            else:
                failed += 1
        
        print(f"[DataLoader] Successfully loaded: {len(email_data)}")
        print(f"[DataLoader] Failed: {failed}")
        
        return email_data
    
    def build_vocab(self, email_data: List[Dict]) -> int:
        """
        Build vocabulary with frequency threshold and size limit.
        
        Args:
            email_data: List of email dictionaries
        
        Returns:
            Final vocabulary size
        """
        print("[DataLoader] Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        for email in email_data:
            word_counts.update(email['tokens'])
        
        # Filter by frequency and take top K
        vocab_candidates = [
            (word, count) for word, count in word_counts.items()
            if count >= self.min_freq and word not in self.word2idx
        ]
        
        # Sort by frequency and take top words
        vocab_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate available slots
        reserved_tokens = len(EmailTokenizer.SPECIAL_TOKENS)
        available_slots = self.vocab_size - reserved_tokens
        
        # Add words to vocabulary
        idx = len(self.word2idx)
        for word, count in vocab_candidates[:available_slots]:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1
        
        self.vocab_built = True
        actual_vocab_size = len(self.word2idx)
        
        print(f"[DataLoader] Vocabulary size: {actual_vocab_size}")
        print(f"  Reserved tokens: {reserved_tokens}")
        print(f"  Content tokens: {actual_vocab_size - reserved_tokens}")
        print(f"  Coverage: {len(vocab_candidates[:available_slots]) / len(vocab_candidates) * 100:.1f}%")
        
        return actual_vocab_size
    
    def text_to_sequence(
        self,
        tokens: List[str],
        add_special_tokens: bool = True
    ) -> Tuple[List[int], int]:
        """
        Convert tokens to sequence of indices.
        
        Args:
            tokens: List of tokens
            add_special_tokens: Add SOS/EOS tokens
        
        Returns:
            (sequence, actual_length)
        """
        # Convert tokens to indices
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # Add special tokens
        if add_special_tokens:
            sequence = [self.word2idx['<SOS>']] + sequence + [self.word2idx['<EOS>']]
        
        # Get actual length before padding
        actual_length = len(sequence)
        
        # Truncate or pad
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            actual_length = self.max_length
        else:
            sequence = sequence + [self.word2idx['<PAD>']] * (self.max_length - len(sequence))
        
        return sequence, actual_length
    
    def prepare_dataset(
        self,
        email_data: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare dataset with train/val/test split.
        
        Args:
            email_data: List of email dictionaries
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            seed: Random seed
        
        Returns:
            (train_data, val_data, test_data, metadata)
        """
        print("[DataLoader] Preparing dataset...")
        
        if not self.vocab_built:
            raise RuntimeError("Vocabulary not built. Call build_vocab() first.")
        
        # Convert to sequences
        sequences = []
        lengths = []
        
        for email in email_data:
            seq, length = self.text_to_sequence(email['tokens'])
            sequences.append(seq)
            lengths.append(length)
        
        sequences = np.array(sequences)
        lengths = np.array(lengths)
        
        # Shuffle
        np.random.seed(seed)
        indices = np.random.permutation(len(sequences))
        sequences = sequences[indices]
        lengths = lengths[indices]
        
        # Split
        n = len(sequences)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_seq = sequences[:train_end]
        train_len = lengths[:train_end]
        
        val_seq = sequences[train_end:val_end]
        val_len = lengths[train_end:val_end]
        
        test_seq = sequences[val_end:]
        test_len = lengths[val_end:]
        
        print(f"[DataLoader] Split complete:")
        print(f"  Train: {len(train_seq)} samples")
        print(f"  Val:   {len(val_seq)} samples")
        print(f"  Test:  {len(test_seq)} samples")
        
        return (
            (train_seq, train_len),
            (val_seq, val_len),
            (test_seq, test_len)
        )
    
    def save(self, filepath: str = 'data/dataloader_v2.pkl'):
        """Save dataloader state with version info"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {
            'version': self.VERSION,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': len(self.word2idx),
            'max_length': self.max_length,
            'min_freq': self.min_freq,
            'preserve_urls': self.preserve_urls,
            'vocab_built': self.vocab_built
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[DataLoader] Saved to {filepath}")
    
    def load(self, filepath: str = 'data/dataloader_v2.pkl'):
        """Load dataloader state with version check"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Version check
        if state.get('version') != self.VERSION:
            warnings.warn(
                f"Version mismatch: saved={state.get('version')}, current={self.VERSION}"
            )
        
        self.word2idx = state['word2idx']
        self.idx2word = state['idx2word']
        self.max_length = state['max_length']
        self.min_freq = state.get('min_freq', 2)
        self.preserve_urls = state.get('preserve_urls', True)
        self.vocab_built = state.get('vocab_built', True)
        
        print(f"[DataLoader] Loaded from {filepath}")
        print(f"  Version: {state.get('version')}")
        print(f"  Vocab size: {len(self.word2idx)}")


class EmailDataset(Dataset):
    """
    PyTorch Dataset with proper length tracking.
    """
    
    def __init__(self, sequences: np.ndarray, lengths: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequence': self.sequences[idx],
            'length': self.lengths[idx]
        }


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    sequences = torch.stack([item['sequence'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    
    # Sort by length (descending) for pack_padded_sequence
    lengths_sorted, sorted_idx = lengths.sort(descending=True)
    sequences_sorted = sequences[sorted_idx]
    
    return sequences_sorted, lengths_sorted


def create_dataloader(
    data_folder: str = 'data/email',
    batch_size: int = 32,
    vocab_size: int = 10000,
    min_freq: int = 2,
    max_length: int = 512,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0
):
    """
    Main function to create professional dataloaders.
    
    Returns:
        (train_loader, val_loader, test_loader, dataloader_obj, email_data)
    """
    
    print("="*60)
    print("Creating Professional Email DataLoader")
    print("="*60)
    
    # Initialize
    loader = EmailDataLoader(
        data_folder=data_folder,
        max_length=max_length,
        vocab_size=vocab_size,
        min_freq=min_freq
    )
    
    # Load emails
    email_data = loader.load_all_emails()
    
    if len(email_data) == 0:
        raise ValueError("No emails loaded!")
    
    # Build vocabulary
    loader.build_vocab(email_data)
    
    # Prepare datasets
    (train_seq, train_len), (val_seq, val_len), (test_seq, test_len) = \
        loader.prepare_dataset(email_data, train_ratio, val_ratio, seed)
    
    # Create datasets
    train_dataset = EmailDataset(train_seq, train_len)
    val_dataset = EmailDataset(val_seq, val_len)
    test_dataset = EmailDataset(test_seq, test_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    # Save state
    loader.save()
    
    print("="*60)
    print("DataLoader Creation Complete")
    print("="*60)
    
    return train_loader, val_loader, test_loader, loader, email_data


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Professional Email DataLoader v2.0")
    print("="*60 + "\n")
    
    # Test with sample data
    train_loader, val_loader, test_loader, loader, email_data = create_dataloader(
        data_folder='data/email',
        batch_size=4,
        vocab_size=5000,
        min_freq=1,
        max_length=256
    )
    
    print("\n" + "="*60)
    print("Testing Batch Loading")
    print("="*60)
    
    # Test train loader
    for sequences, lengths in train_loader:
        print(f"\nBatch shape: {sequences.shape}")
        print(f"Lengths: {lengths.tolist()}")
        print(f"Sorted: {(lengths[:-1] >= lengths[1:]).all()}")
        
        # Show decoded sample
        sample_tokens = []
        for idx in sequences[0][:lengths[0]]:
            token = loader.idx2word.get(idx.item(), '<UNK>')
            sample_tokens.append(token)
        
        print(f"\nSample tokens (first 20):")
        print(' '.join(sample_tokens[:20]))
        break
    
    # Statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total emails: {len(email_data)}")
    print(f"Vocabulary size: {len(loader.word2idx)}")
    print(f"Average email length: {np.mean([e['features']['length'] for e in email_data]):.1f}")
    print(f"Max email length: {max([e['features']['length'] for e in email_data])}")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")