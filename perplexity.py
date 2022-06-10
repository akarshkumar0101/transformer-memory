import torch
from tqdm import tqdm

with open('book.txt') as f:
    text_book = f.read()

def calc_perplexity(tokenizer, model, text=None, context_length=100, stride=1, device='cpu', wandb=None):
    if text is None:
        text = text_book
        
    max_length = model.config.n_positions
    max_length = context_length
    
    all_tokens = tokenizer(text, return_tensors='pt')
    encodings = all_tokens
    
    model = model.to(device)

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    # for i in tqdm(range(0, 5000, stride)):
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

        if not torch.isnan(neg_log_likelihood):
            nlls.append(neg_log_likelihood)
        else:
            print('Found NaN value')

        if wandb is not None and i % 100 == 0 and len(nlls) > 0:
            wandb.log({'nll': neg_log_likelihood.item(), 
                       'nll_running_mean': torch.stack(nlls).mean().item(),
                       'running_ppl': (torch.stack(nlls).sum()/end_loc).exp().item()})

    nlls = torch.stack(nlls)
    ppl = torch.exp(nlls.sum() / end_loc)

    if wandb is not None:
        wandb.log({'final_ppl': ppl.item()})
    return ppl

