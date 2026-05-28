import torch
import torch.nn as nn
import torch.optim as optim
import time
from rouge_metrics import calculate_rouge

def evaluate_rouge(model, val_loader, vocab, num_samples=100):
    model.eval()
    rouge1_scores = []
    rouge2_scores = []
    
    device = next(model.parameters()).device
    samples_evaluated = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.long().to(device)
            targets = targets.long().to(device)
            
            for i in range(data.size(0)):
                if samples_evaluated >= num_samples:
                    break
                    
                full_indices = data[i]
                full_words = [
                    vocab["idx_to_word"].get(idx.item(), "<UNK>")
                    for idx in full_indices
                    if idx.item() != 0
                ]

                if len(full_words) < 4:
                    continue

                split_point = int(len(full_words) * 0.75)
                input_text = " ".join(full_words[:split_point])
                target_text = " ".join(full_words[split_point:])

                if not target_text:
                    continue

                generated = model.predict_next_tokens(
                    input_text, vocab, 
                    num_tokens=len(target_text.split())
                )

                rouge_scores = calculate_rouge(generated, target_text)

                rouge1_scores.append(rouge_scores["rouge1"]["f1"])
                rouge2_scores.append(rouge_scores["rouge2"]["f1"])

                samples_evaluated += 1

            if samples_evaluated >= num_samples:
                break

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0

    return avg_rouge1, avg_rouge2


def print_gpu_memory():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        free = total - reserved
        
        print(f"GPU Memory - Total: {total:.2f} GB")
        print(f"             Free: {free:.2f} GB") 
        print(f"             Allocated: {allocated:.2f} GB")
        print(f"             Reserved: {reserved:.2f} GB")
    else:
        print("CUDA not available")

def train_model():
    from next_token_dataset import create_data_loaders
    from lstm_model import LSTMLanguageModel

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_loader, val_loader, test_loader, vocab = create_data_loaders()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMLanguageModel(vocab_size=vocab['vocab_size']).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    num_epochs = 5
    
    print("Запускаем обучение:")
    print(f"Размер словаря: {vocab['vocab_size']}")
    print(f"Количество батчей: {len(train_loader)}")
    print(f"Устройство: {device}")
    print("-" * 50)
    
    accumulation_steps = 4  
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        optimizer.zero_grad()  
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.long().to(device)
            targets = targets.long().to(device)
            
            output = model(data)
            loss = criterion(output.reshape(-1, output.size(-1)), targets.reshape(-1))
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps  
            
            if batch_idx % 50 == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                print(f"Эпоха {epoch+1}/{num_epochs} | Батч {batch_idx}/{len(train_loader)} | Loss: {avg_loss_so_far:.4f}")
                
                if batch_idx % 200 == 0:
                    print_gpu_memory()
        
        if len(train_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if epoch == num_epochs - 1:
            print("Вычисляем ROUGE метрики на 100 примерах...")
            rouge1, rouge2 = evaluate_rouge(model, val_loader, vocab, num_samples=100)
            print(f"Эпоха {epoch+1} завершена:")
            print(f"  Средний Loss: {avg_loss:.4f}")
            print(f"  ROUGE-1 F1: {rouge1:.4f}")
            print(f"  ROUGE-2 F1: {rouge2:.4f}")
            print(f"  Время эпохи: {epoch_time:.2f} сек")
        else:
            print(f"Эпоха {epoch+1} завершена:")
            print(f"  Средний Loss: {avg_loss:.4f}")
            print(f"  Время эпохи: {epoch_time:.2f} сек")
        
        print("-" * 30)
    
    torch.save(model.state_dict(), './models/lstm_model_bs32_accum4.pth')
    
    print("\nПримеры предсказаний обученной модели:")
    examples = ["i love", "the weather is", "i want to"]
    for example in examples:
        prediction = model.predict_next_tokens(example, vocab)
        print(f"  '{example}' -> '{prediction}'")

if __name__ == "__main__":
    train_model()