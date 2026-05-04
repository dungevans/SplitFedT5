import torch
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer

def val_T5(model, test_loader, device, logger):
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("philschmid/flan-t5-base-samsum")
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Sử dụng hàm generate chính chủ của T5
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=batch['attention_mask'].to(device),
                max_new_tokens=160
            )
            
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            golds = tokenizer.batch_decode(labels.replace(-100, 0), skip_special_tokens=True)
            
            predictions.extend(preds)
            references.extend(golds)

    results = rouge.compute(predictions=predictions, references=references)
    print(f"ROUGE Results: {results}")
    logger.log_info(f"Evaluation ROUGE: {results}")
    return results