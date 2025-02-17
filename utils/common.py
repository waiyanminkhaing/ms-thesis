import os
import warnings
import bert_score
import torch
import pandas as pd
import tensorflow as tf
from torch.utils.data import DataLoader
from utils.custom_class import EvaluationDataset, TrainerProgressCallback
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu import corpus_chrf
from sacrebleu.metrics import CHRF
from tqdm.notebook import tqdm
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM,
    Trainer, TrainingArguments,
)
from peft import PeftModel, PeftConfig

# Suppress specific UserWarnings
warnings.filterwarnings("ignore", category=UserWarning, module="peft.peft_model")

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

# Function to generate masked predictions using Hugging Face Dataset
def generate_masked_predictions_hf(dataset, model, tokenizer, device):
    def predict_fn(example):
        input = tokenizer(
            example["burmese"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

        masked_input_ids = torch.tensor(input["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(input["attention_mask"]).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        mask_positions = (masked_input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        # Replace masked tokens with the most probable prediction
        predicted_tokens = masked_input_ids.clone()
        for pos in mask_positions:
            predicted_token_id = torch.argmax(logits[0, pos], dim=-1).item()
            predicted_tokens[0, pos] = predicted_token_id

        # Decode and ensure Burmese text is output
        generated_text = tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
        example["generated"] = generated_text

        return example

    dataset = dataset.map(predict_fn, batched=False)
    return dataset

# function to generate predictions for mt5
def generate_mt5_predictions(dataset, model, tokenizer, device):
    # Move to gpu
    model.to(device)

    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        fp16= False,
        bf16= True,
        auto_find_batch_size=True,
    )

    # Load Trainer and Data
    trainer = Trainer(model=model, args=training_args)
    total_batches = len(trainer.get_eval_dataloader(dataset))

    # Add Progress Callback
    trainer.add_callback(TrainerProgressCallback(total_batches))

    # Run Predictions
    outputs = trainer.predict(dataset)

    # Decode Predictions Efficiently
    generated_texts = tokenizer.batch_decode(outputs.predictions, skip_special_tokens=True, batch_size=16)

    # Add Predictions to Dataset
    dataset = dataset.add_column("generated", generated_texts)

    return dataset

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

# Function to Compute Metrics using Hugging Face Dataset
def compute_metrics_hf(dataset, device):
    smooth_fn = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    chrf = CHRF()

    def compute_metrics_batch(example):
        predictions = example["generated"]
        references = example["target"]

        # Compute BLEU Score
        bleu_score = sentence_bleu([references.split()], predictions.split(), smoothing_function=smooth_fn)

        # Compute ROUGE Scores
        rouge_scores = rouge.score(predictions, references)

        # Compute chrF-S Score
        chrf_score = chrf.sentence_score(predictions, [references]).score

        # Compute BERTScore
        bert_p, bert_r, bert_f1 = bert_score([predictions], [references], lang="my", device=device)

        return {
            "bleu": bleu_score,
            "rouge-1": rouge_scores["rouge1"].fmeasure,
            "rouge-2": rouge_scores["rouge2"].fmeasure,
            "rouge-l": rouge_scores["rougeL"].fmeasure,
            "chrf-s": chrf_score,
            "bert_score": bert_f1.mean().item()
        }

    # Apply batch-wise metric computation
    dataset = dataset.map(compute_metrics_batch, batched=False, num_proc=4)

    return dataset


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

# Function to compute masked perplexity for a single
def compute_multilingual_masked_perplexity_single(text, model, tokenizer, device):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)  # Forward pass
        logits = outputs.logits

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
    perplexity_score = torch.exp(log_perplexity).cpu().numpy()[0]

    return perplexity_score

# Function to compute mT5 perplexity for a single example
def compute_multilingual_mt5_perplexity_single(text, model, tokenizer, device):
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)  # Forward pass
        logits = outputs.logits

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
    perplexity_score = torch.exp(log_perplexity).cpu().numpy()[0]

    return perplexity_score

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

# Function to merge fine-tuned model
def merge_fine_tuned_model(base_model_name, adapter_model_name):
    adapter_path = f"model-variants/models/{adapter_model_name}"

    # Load the base model (needed for LoRA)
    base_model = AutoModel.from_pretrained(base_model_name)

    # Load the LoRA adapter configuration
    config = PeftConfig.from_pretrained(adapter_path)
    print(f"🔍 LoRA Configuration:\n{config}")

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)

    # Check LoRA layers before merging
    lora_keys = [name for name, _ in model.named_parameters() if "lora" in name]
    print(f"LoRA Adapter Keys Found: {len(lora_keys)} layers")
    
    if not lora_keys:
        print("⚠️ No LoRA layers found! Ensure the adapter was properly trained and saved.")
        return

    # Now, Merge LoRA into base model
    merged_model = model.merge_and_unload()

    # Save full merged model
    full_model_path = f"{adapter_path}_full"
    merged_model.save_pretrained(full_model_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(full_model_path)

    print(f"Merged model saved at: {full_model_path}")


# function to get fine tuned model
def get_fine_tuned_model(model_name, spt_name, base_model_name, device):
    # load base model
    model = AutoModelForMaskedLM.from_pretrained(base_model_name)

    # Load LoRA Weights
    lora_checkpoint_path = f"model-variants/models/{model_name}_{spt_name.upper()}"
    model = PeftModel.from_pretrained(model, lora_checkpoint_path)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint_path)

    return model, tokenizer

# function to get embeddings fine tuned model
def get_embedded_fine_tuned_model(model_name, spt_name, device):
    model_path = f"model-variants/models/Embedded_{model_name}_{spt_name.upper()}"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    return model, tokenizer

# function to get distilled fine tuned model
def get_distilled_fine_tuned_model(model_name, spt_name, distill_model_name, device):
    model_path = f"model-variants/models/Distilled_{model_name}_{spt_name.upper()}_{distill_model_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)

    return model, tokenizer

# function to extract training and evaluation metrics from TensorBoard event logs.
def extract_metrics_from_logs(name):
    log_dir = f"logs/{name}"
    metrics_dict = {"epoch": [], "train_loss": [], "eval_loss": [], "samples_per_sec": [], "steps_per_sec": []}

    # Locate event files
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f]
    
    for event_file in event_files:
        for summary in tf.compat.v1.train.summary_iterator(event_file):
            for v in summary.summary.value:
                if v.tag == "train/loss":
                    metrics_dict["epoch"].append(summary.step)
                    metrics_dict["train_loss"].append(v.simple_value)
                elif v.tag == "eval/loss":
                    metrics_dict["eval_loss"].append(v.simple_value)
                elif v.tag == "eval/samples_per_second":
                    metrics_dict["samples_per_sec"].append(v.simple_value)
                elif v.tag == "eval/steps_per_second":
                    metrics_dict["steps_per_sec"].append(v.simple_value)

    return pd.DataFrame(metrics_dict).sort_values("epoch")

# function to compute the size of a PyTorch model in megabytes (MB).
def get_model_size(model_name):
    model_name = f"model-variants/models/{model_name}"

    model = torch.load(os.path.join(model_name, "pytorch_model.bin"), map_location="cpu")

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())  # Size of parameters
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())  # Size of buffers
    total_size = (param_size + buffer_size) / (1024 ** 2)  # Convert to MB

    return round(total_size, 2)