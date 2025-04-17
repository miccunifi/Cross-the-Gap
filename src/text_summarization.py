import pickle
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import pipeline

from data_utils import PROJECT_ROOT, RETRIEVAL_SPLTS
from utils import get_dataset


def text_summarization(args):
    SYSTEM_PROMPT = "You are a helpful assistant that summarizes long text to under 77 tokens without changing the meaning."

    split = args.split if args.split is not None else RETRIEVAL_SPLTS[args.dataset_name.lower()]["gallery"]

    pipe = pipeline(
        "text-generation",
        args.model_name,
        torch_dtype=torch.float16,
        device="cuda",
        token=args.hf_token)

    torch.compile(pipe.model, fullgraph=True)
    dataset = get_dataset(args.dataroot, args.dataset_name, split, None, return_image=False)
    loader = DataLoader(dataset=dataset, batch_size=1, num_workers=12, pin_memory=False)

    summarized_texts = {}
    captions_path = PROJECT_ROOT / 'data' / 'summarized_texts' / args.dataset_name.lower() / split

    if (captions_path / "summarized_text.pkl").exists():
        print(f"Loading existing summarized texts from {captions_path}")
        with open(captions_path / "summarized_text.pkl", "rb") as f:
            summarized_texts = pickle.load(f)

    for text_idx, batch in enumerate(tqdm(loader, desc=f"Summarizing text")):
        text = batch.get('long_text')
        names = batch.get('text_name')
        if names is None:
            names = batch.get('image_name', batch.get('name'))

        if names[0] in summarized_texts:
            continue

        prompt = [
            {"role": "system",
             "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Please summarize this text to under 77 tokens: {text[0]}. "
                        f"\n Start directly with the summarized text without anything else before"}
        ]

        outputs = pipe(
            prompt,
            max_new_tokens=80,
        )
        sum_text = outputs[0]['generated_text'][-1]['content']
        summarized_texts[names[0]] = sum_text

        if text_idx % args.save_frequency == 0:
            save_text(summarized_texts, captions_path)

    save_text(summarized_texts, captions_path)


def save_text(summarized_texts, captions_path):
    captions_path.mkdir(parents=True, exist_ok=True)
    with open(captions_path / "summarized_text.pkl", "wb") as f:
        pickle.dump(summarized_texts, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True, help="path to datasets root")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--save_frequency", type=int, default=1000)
    parser.add_argument("--hf_token", type=str, default=None)

    args = parser.parse_args()

    text_summarization(args)


if __name__ == '__main__':
    main()
