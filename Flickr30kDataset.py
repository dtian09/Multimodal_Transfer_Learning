import torch 
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
from torchvision import transforms
from transformers import AutoTokenizer
import os

class Flickr30kDataset(Dataset):
    def __init__(self, split="train", transform=None, tokenizer_name="bert-base-uncased", max_length=31):
        self.max_length = max_length

        if os.path.isdir("flickr30k_" + split + "_filtered"):
            self.dataset = load_from_disk("flickr30k_" + split + "_filtered")
        else:   
            # Load full dataset (all splits are under 'test')
            full_dataset = load_dataset("nlphuji/flickr30k", split="test", keep_in_memory=False)

            # Filter by internal 'split' field and keep only 'caption' and 'image'
            filtered = full_dataset.filter(lambda x: x["split"] == split, keep_in_memory=False)
            self.dataset = filtered.remove_columns(
                [col for col in filtered.column_names if col not in {"caption", "image"}]
            )

            filtered.save_to_disk("flickr30k_" + split + "_filtered")

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
 
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"].convert("RGB")
        image = self.transform(image)

        # Ensure caption is a plain string
        caption = str(item["caption"])

        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "caption": caption  # now guaranteed to be a plain Python string
        }
