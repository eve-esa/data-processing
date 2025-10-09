import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Any


# Create a custom Dataset class to handle batching
# Custom dataset using batch tokenization
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def perplexity(
    predictions: list[str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 1,
    stride: int = 128,
    max_length: int = 2048,
) -> float:

    device = model.device

    def collate_fn(batch):
        encodings = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return encodings

    dataset = TextDataset(predictions)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    nll_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            prev_end_loc = 0
            encoding = batch["input_ids"]
            seq_len = encoding.size(1)
            for begin_loc in tqdm(range(0, seq_len, stride), desc="Windowing"):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = (
                    end_loc - prev_end_loc
                )  # may be different from stride on last loop
                input_ids = encoding[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)

                    # loss is calculated using CrossEntropyLoss which averages over valid labels
                    # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                    # to the left by 1.
                    neg_log_likelihood = outputs.loss

                # Accumulate the total negative log-likelihood and the total number of tokens
                num_valid_tokens = (
                    (target_ids != -100).sum().item()
                )  # number of valid tokens in target_ids
                batch_size = target_ids.size(0)
                num_loss_tokens = (
                    num_valid_tokens - batch_size
                )  # subtract batch_size due to internal label shift
                nll_sum += neg_log_likelihood * num_loss_tokens
                n_tokens += num_loss_tokens

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
        avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
        ppl = torch.exp(avg_nll)

    return ppl.item()
