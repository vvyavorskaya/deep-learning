from rouge_score import rouge_scorer

def calculate_rouge(generated_text, reference_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    
    return {
        'rouge1': {
            'precision': scores['rouge1'].precision,
            'recall': scores['rouge1'].recall, 
            'f1': scores['rouge1'].fmeasure
        },
        'rouge2': {
            'precision': scores['rouge2'].precision,
            'recall': scores['rouge2'].recall,
            'f1': scores['rouge2'].fmeasure
        }
    }