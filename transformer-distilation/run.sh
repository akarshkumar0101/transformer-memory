# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=rnn          --mode=_             --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=gpt          --mode=_             --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=compress-gpt --mode=random-causal --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=compress-gpt --mode=half-causal   --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=compress-gpt --mode=random-full   --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=compress-gpt --mode=half-full     --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12


# python main.py --track=True --name="{model}_0" --device=cpu --model=gpt          --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_0" --device=cpu --model=weird        --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
python main.py --track=True --name="{model}_0" --device=cpu --model=compress-gpt --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12 --mode=half-full