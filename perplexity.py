import torch
from tqdm import tqdm

with open('book.txt') as f:
    text_book = f.read()

def calc_perplexity(tokenizer, model, text=None, context_length=100, stride=1, device='cpu'):
    if text is None:
        text = text_book
        
    max_length = model.config.n_positions
    max_length = context_length
    
    all_tokens = tokenizer(text, return_tensors='pt')
    encodings = all_tokens
    
    model = model.to(device)

    nlls = []
    # for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    for i in tqdm(range(0, 5000, stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        # print(begin_loc, end_loc, end_loc-begin_loc)
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len
            # print(outputs.loss.item(), outputs[0].item())

        nlls.append(neg_log_likelihood)

    nlls = torch.stack(nlls)
    nlls = nlls[~nlls.isnan()]
    print(len(nlls), end_loc)
    ppl = torch.exp(nlls.sum() / end_loc)
    return ppl

