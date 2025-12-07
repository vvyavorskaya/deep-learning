import torch
from lstm_model import LSTMLanguageModel
from next_token_dataset import NextTokenDataset
from rouge_metrics import calculate_rouge

def load_trained_lstm_model(model_path, vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMLanguageModel(vocab_size=vocab['vocab_size']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def evaluate_lstm_on_examples(model, vocab, val_file_path, num_samples=100):
    with open(val_file_path, 'r', encoding='utf-8') as f:
        val_texts = [line.strip() for line in f if line.strip()]
    
    if num_samples:
        val_texts = val_texts[:num_samples]
    
    rouge1_scores = []
    rouge2_scores = []
    examples = []
    
    for i, text in enumerate(val_texts):
        if not text.strip():
            continue
            
        tokens = text.split()
        if len(tokens) < 3:
            continue
        
        split_point = max(1, len(tokens) // 2)
        prompt = ' '.join(tokens[:split_point])
        target = ' '.join(tokens[split_point:])
        
        try:
            generated = model.predict_next_tokens(prompt, vocab, num_tokens=len(target.split()))
            
            if generated and target:
                rouge_scores = calculate_rouge(generated, target)
                rouge1_scores.append(rouge_scores['rouge1']['f1'])
                rouge2_scores.append(rouge_scores['rouge2']['f1'])
                
                if len(examples) < 5:
                    examples.append({
                        'prompt': prompt,
                        'target': target,
                        'generated': generated
                    })
        except Exception as e:
            print(f"Ошибка при генерации LSTM для текста {i}: {e}")
            continue
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    
    return avg_rouge1, avg_rouge2, examples

def compare_models():

    train_dataset = NextTokenDataset('./data/train_cleaned.txt')
    vocab = {
        'word_to_idx': train_dataset.word_to_idx,
        'idx_to_word': train_dataset.idx_to_word,
        'vocab_size': train_dataset.vocab_size
    }
    
    lstm_model = load_trained_lstm_model('./models/lstm_model_bs64_accum4.pth', vocab)
    lstm_rouge1, lstm_rouge2, lstm_examples = evaluate_lstm_on_examples(
        lstm_model, vocab, './data/val_cleaned.txt', num_samples=100
    )
    
    from src.transformer_model import evaluate_transformer_model
    transformer_rouge1, transformer_rouge2, transformer_examples = evaluate_transformer_model(
        './data/val_cleaned.txt', num_samples=100
    )
    
    # Выводим результаты
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print("="*50)
    print(f"LSTM Model:")
    print(f"  ROUGE-1 F1: {lstm_rouge1:.4f}")
    print(f"  ROUGE-2 F1: {lstm_rouge2:.4f}")
    print(f"Transformer Model:")
    print(f"  ROUGE-1 F1: {transformer_rouge1:.4f}")
    print(f"  ROUGE-2 F1: {transformer_rouge2:.4f}")
    print("="*50)
    
    print("\nПРИМЕРЫ ГЕНЕРАЦИИ LSTM:")
    for i, example in enumerate(lstm_examples, 1):
        print(f"{i}. Промпт: '{example['prompt']}'")
        print(f"   Цель: '{example['target']}'")
        print(f"   LSTM: '{example['generated']}'")
        print()
    
    print("\nПРИМЕРЫ ГЕНЕРАЦИИ TRANSFORMER:")
    for i, example in enumerate(transformer_examples, 1):
        print(f"{i}. Промпт: '{example['prompt']}'")
        print(f"   Цель: '{example['target']}'")
        print(f"   Transformer: '{example['generated']}'")
        print()