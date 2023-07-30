# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=rnn          --mode=_             --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=gpt          --mode=_             --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=weird        --mode=_             --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=compress-gpt --mode=random-causal --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=compress-gpt --mode=half-causal   --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=compress-gpt --mode=random-full   --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12
# python main.py --track=True --name="{model}_{mode}" --device=cpu --model=compress-gpt --mode=half-full     --n-iters=500 --n-embd=768 --n-layer=4 --n-head=12

python train.py --track=True --name="{model}_{mode}_0" --device=cuda:0 --model=gpt          --mode=_             --n-iters=5000 --n-embd=768 --n-layer=4 --n-head=12 --save-model="models/gpt.pt"
python train.py --track=True --name="{model}_{mode}_0" --device=cuda:1 --model=compress-gpt --mode=random-causal --n-iters=5000 --n-embd=768 --n-layer=4 --n-head=12 --save-model="models/random-causal.pt"

python train.py --track=False --name="{model}_{mode}_0" --device=cuda:0 --model=gpt          --mode=_             --n-iters=5000 --n-embd=768 --n-layer=4 --n-head=12 --save-model="models/gpt.pt"


python distill.py --track=True --name="distill" --device=cuda:0 --lr=3e-5 --save-mode="models/distilled-rnn.pt"