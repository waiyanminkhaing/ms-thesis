import bert_score
import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils.custom_class import EvaluationDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu import corpus_chrf
from tqdm.notebook import tqdm

# function to compute metrics
def compute_metrics_batch(dataset, referenceColName, device, batch_size=32):
    dataloader = DataLoader(EvaluationDataset(dataset, referenceColName), batch_size=batch_size, shuffle=False)

    all_bleu_scores, all_rouge1, all_rouge2, all_rougeL, all_chrfs, all_berts = [], [], [], [], [], []

    smooth_fn = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    for batch in tqdm(dataloader, desc="Computing Metrics", unit="batch"):
        predictions, references = batch

        # Compute BLEU in batch
        batch_bleu = [sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)
                      for pred, ref in zip(predictions, references)]
        all_bleu_scores.extend(batch_bleu)

        # Compute ROUGE in batch
        batch_rouge = [rouge.score(pred, ref) for pred, ref in zip(predictions, references)]
        all_rouge1.extend([r["rouge1"].fmeasure for r in batch_rouge])
        all_rouge2.extend([r["rouge2"].fmeasure for r in batch_rouge])
        all_rougeL.extend([r["rougeL"].fmeasure for r in batch_rouge])

        # Compute chrF-S in batch
        batch_chrf = corpus_chrf(predictions, [[ref] for ref in references]).score
        all_chrfs.extend([batch_chrf] * len(predictions))  # Apply same batch score to all

        # Compute BERTScore in batch
        batch_bert = bert_score.score(predictions, references, lang="my", device=device)
        all_berts.extend(batch_bert[2].tolist())  # F1 scores from BERTScore

    print("Finished Computing Metrics!")

    # Store results back into dataset
    dataset["bleu"] = all_bleu_scores
    dataset["rouge-1"] = all_rouge1
    dataset["rouge-2"] = all_rouge2
    dataset["rouge-l"] = all_rougeL
    dataset["chrf-s"] = all_chrfs
    dataset["bert_score"] = all_berts

# Function to generate masked predictions
def generate_masked_predictions_batch(dataloader, model, tokenizer, device):
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Masked Predictions"):
            # Move batch data to GPU
            masked_input_ids, attention_mask, _ = [x.to(device) for x in batch]

            # Run model inference on GPU
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)

            # Replace masked tokens with predicted tokens
            predicted_tokens_batch = masked_input_ids.clone()
            for i in range(masked_input_ids.shape[0]):  # Loop over batch
                mask_positions = (masked_input_ids[i] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                for pos in mask_positions:
                    predicted_token_id = torch.argmax(outputs.logits[i, pos], dim=-1).item()
                    predicted_tokens_batch[i, pos] = predicted_token_id

            # Decode predictions
            batch_predictions = tokenizer.batch_decode(predicted_tokens_batch.cpu(), skip_special_tokens=True)
            all_predictions.extend(batch_predictions)

    return all_predictions

# function to generate predictions
def generate_mt5_predictions_batch(dataloader, model, tokenizer, spt_processor, device):
    predictions = []

    for batch in tqdm(dataloader, desc=f"Generating Predictions", unit="batch"):
        # Apply spt
        spt_encoded_batch = [" ".join(spt_processor.encode_as_pieces(text)) for text in batch]
        
        # Tokenize input
        inputs = tokenizer(spt_encoded_batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            # Generate output sequence
            output_tokens = model.generate(**inputs, max_length=128)

        # Decode generated sequences
        decoded_output = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_tokens]
        predictions.extend(decoded_output)

    return predictions

 # Function to compute masked perplexity in batch
def compute_multilingual_masked_perplexity_batch(dataloader, model, tokenizer, device):
    perplexities = []

    model.to(device)  # Move model to GPU
    model = torch.compile(model)
    model.eval()  # Set model to evaluation mode

    for batch in tqdm(dataloader, desc="Computing Perplexity", unit="batch"):
        batch_texts = batch  # Text input batch

        # Tokenize batch with padding & truncation
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)  # Forward pass
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

        temperature = 1.5
        logits = logits / temperature

        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Get token log-likelihoods using true token IDs
        target_ids = inputs["input_ids"]
        log_likelihood = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

        # Apply attention mask to remove padding tokens
        mask = inputs["attention_mask"]
        masked_log_likelihood = log_likelihood * mask  # Zero out padding contributions

        # Compute sentence-level mean log-likelihood
        sentence_log_likelihood = masked_log_likelihood.sum(dim=1) / mask.sum(dim=1)

        # Convert log-likelihood to perplexity
        log_perplexity = -sentence_log_likelihood
        batch_perplexities = torch.exp(log_perplexity).cpu().numpy()

        perplexities.extend(batch_perplexities)

    return perplexities

def compute_multilingual_mt5_perplexity_batch(dataloader, model, tokenizer, spt_processor, device):
    predictions = []

    model.to(device)  # Move model to GPU
    model = torch.compile(model)
    model.eval()  # Set model to evaluation mode

    for batch in tqdm(dataloader, desc=f"Generating Predictions", unit="batch"):
        # Apply spt
        spt_encoded_batch = [" ".join(spt_processor.encode_as_pieces(text)) for text in batch]
        
        # Tokenize input
        inputs = tokenizer(spt_encoded_batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            # Generate output sequence
            output_tokens = model.generate(**inputs, max_length=128)

        # Decode generated sequences
        decoded_output = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in output_tokens]
        predictions.extend(decoded_output)

    return predictions

# function to create mean scores dataset
def convert_to_mean_scores_df(datasets):
    # Compute mean scores dynamically using a dictionary comprehension
    benchmarking_mean_scores = {
        model: {
            "BLEU": dataset["bleu"].mean(),
            "ROUGE-1": dataset["rouge-1"].mean(),
            "ROUGE-2": dataset["rouge-2"].mean(),
            "ROUGE-L": dataset["rouge-l"].mean(),
            "chrF-S": dataset["chrf-s"].mean(),
            "BERT Score": dataset["bert_score"].mean(),
            "Perplexity": dataset["perplexity"].mean(),
        }
        for model, dataset in datasets.items()
    }

    # Convert mean scores dictionary to DataFrame for better visualization
    benchmarking_mean_scores_df = pd.DataFrame.from_dict(benchmarking_mean_scores, orient='index')

    return benchmarking_mean_scores_df