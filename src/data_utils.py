import os
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List

def clean_text(text: str) -> str:

    text = text.lower()
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'http\S+|www\S+|https\S+', '', text) 
    text = re.sub(r'[^\w\s]', '', text)  
    return ' '.join(text.split())


def load_and_clean_data(file_path: str, max_lines: int = 500_000) -> List[str]:
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for i, line in enumerate(f) if line.strip() and i < max_lines]

    cleaned_dataset = [clean_text(text) for text in lines if clean_text(text)]

    return cleaned_dataset


def save_texts(texts: List[str], file_path: str):

    os.makedirs(Path(file_path).parent, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(f"{text}\n" for text in texts)
    print(f"Сохранено {len(texts)} строк в {file_path}")


# разделяем тексты на train/val/test
def split_and_save_data(cleaned_dataset: List[str], output_dir: str = './data', train_ratio=0.8, val_ratio=0.1):

    save_texts(cleaned_dataset, os.path.join(output_dir, 'dataset_processed.txt'))

    train_texts, temp_texts = train_test_split(cleaned_dataset, test_size=1-train_ratio, random_state=42)
    val_size = val_ratio / (1 - train_ratio)  
    val_texts, test_texts = train_test_split(temp_texts, test_size=1 - val_size, random_state=42)

    save_texts(train_texts, os.path.join(output_dir, 'train_cleaned.txt'))
    save_texts(val_texts, os.path.join(output_dir, 'val_cleaned.txt'))
    save_texts(test_texts, os.path.join(output_dir, 'test_cleaned.txt'))

    print(f"Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_texts)}")


def main():
    BASE_DIR = Path(__file__).parent.parent
    raw_dataset = BASE_DIR / 'data' / 'raw_dataset.txt'
    output_dir = BASE_DIR / 'data'

    cleaned_dataset = load_and_clean_data(raw_dataset)

    split_and_save_data(cleaned_dataset, output_dir=output_dir)


if __name__ == "__main__":
    main()
