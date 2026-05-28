import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from rouge_metrics import calculate_rouge
import torch.nn.functional as Fcd


class TransformerTextGenerator:
    def __init__(self, model_name='distilgpt2', temperature=0.7, top_k=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.temperature = temperature
        self.top_k = top_k

    def generate_text(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                temperature=self.temperature,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text

    def predict_next_tokens(self, text, num_tokens=10):
        return self.generate_text(text, max_length=num_tokens)


def evaluate_transformer_model(val_file_path, num_samples=100):
    generator = TransformerTextGenerator()
    
    with open(val_file_path, 'r', encoding='utf-8') as f:
        val_texts = [line.strip() for line in f if line.strip()]
    
    if num_samples:
        val_texts = val_texts[:num_samples]
    
    rouge1_scores = []
    rouge2_scores = []
    examples = []

    for i, text in enumerate(val_texts):
        tokens = text.split()
        if len(tokens) < 4:  
            continue
        
        split_point = max(1, len(tokens) * 3 // 4)
        prompt = ' '.join(tokens[:split_point])
        target = ' '.join(tokens[split_point:])
        
        try:
            generated = generator.generate_text(prompt, max_length=len(target.split())*2)
            
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
            print(f"Ошибка при генерации для текста {i}: {e}")
            continue
    
    avg_rouge1 = sum(rouge1_scores)/len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores)/len(rouge2_scores) if rouge2_scores else 0
    
    return avg_rouge1, avg_rouge2, examples


if __name__ == "__main__":
    val_file = './data/val_cleaned.txt'
    print("Загружаем модель distilgpt2...")
    
    rouge1, rouge2, examples = evaluate_transformer_model(val_file, num_samples=100)
    
    print(f"\nTransformer ROUGE-1: {rouge1:.4f}")
    print(f"Transformer ROUGE-2: {rouge2:.4f}")
    
    print("\nПримеры генерации трансформером:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. Промпт: '{example['prompt']}'")
        print(f"   Цель: '{example['target']}'")
        print(f"   Сгенерировано: '{example['generated']}'\n")
