from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    Custom dataset class for batching
    """
    def __init__(self, texts):
        self.texts = [str(text) if text is not None else "" for text in texts] 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
    
class EvaluationDataset(Dataset):
    """
    custom dataset class for evaluation dataset
    """
    def __init__(self, dataframe, referenceColName):
        self.predictions = dataframe["generated"].astype(str).tolist()
        self.references = dataframe[referenceColName].astype(str).tolist()

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        return self.predictions[idx], self.references[idx]
    
# dataset class for masked
class MaskedTextDataset(Dataset):
    def __init__(self, texts, tokenizer, mask_ratio=0.15, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.mask_ratio = mask_ratio
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx] if isinstance(self.texts[idx], str) else ""

        # Tokenize and move tensors to GPU
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length
        )
        
        input_ids = inputs["input_ids"].squeeze(0).to(device)
        attention_mask = inputs["attention_mask"].squeeze(0).to(device)

        # Apply random masking
        seq_length = input_ids.shape[0]
        num_to_mask = max(1, int(self.mask_ratio * (seq_length - 2)))  # Avoid CLS/SEP
        mask_indices = torch.randperm(seq_length - 2)[:num_to_mask] + 1  # Avoid first and last token

        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_indices] = self.tokenizer.mask_token_id  # Replace with [MASK] token

        return masked_input_ids, attention_mask, input_ids