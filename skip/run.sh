
# python train.py --model transformer --n_batches 2000 --batch_size 64 --seq_len 128 --lr 1e-2 --device cuda:3
# python train.py --model transformer --n_batches 2000 --batch_size 64 --seq_len 128 --lr 1e-3 --device cuda:2

# python train.py --model transformer --n_batches 1000 --batch_size 128 --seq_len 128 --device cuda:2 --wandb &
# python train.py --model transformer --n_batches 1000 --batch_size 128 --seq_len 004 --device cuda:1 --wandb &
# python train.py --model transformer --n_batches 1000 --batch_size 128 --seq_len 008 --device cuda:2 --wandb &
# python train.py --model transformer --n_batches 1000 --batch_size 128 --seq_len 016 --device cuda:3 --wandb &

# python train.py --model transformer --n_batches 2000 --batch_size 64 --seq_len 032 --device cuda:0 --wandb &
# python train.py --model transformer --n_batches 2000 --batch_size 64 --seq_len 064 --device cuda:1 --wandb &
# python train.py --model transformer --n_batches 2000 --batch_size 64 --seq_len 128 --device cuda:2 --wandb &
# python train.py --model transformer --n_batches 2000 --batch_size 64 --seq_len 256 --device cuda:3 --wandb &

# python train.py --model   perceiver --n_batches 2000 --batch_size 64 --seq_len 064 --device cuda:4 --wandb &
# python train.py --model   perceiver --n_batches 2000 --batch_size 64 --seq_len 128 --device cuda:5 --wandb &
# python train.py --model   perceiver --n_batches 2000 --batch_size 64 --seq_len 256 --device cuda:6 --wandb

# wait

python train.py --model  longrange1 --n_batches 2000 --batch_size 32 --seq_len 064 --device cuda:0 --wandb &
python train.py --model  longrange1 --n_batches 2000 --batch_size 32 --seq_len 128 --device cuda:1 --wandb &
python train.py --model  longrange1 --n_batches 2000 --batch_size 32 --seq_len 256 --device cuda:2 --wandb &
 
python train.py --model  longrange2 --n_batches 2000 --batch_size 32 --seq_len 064 --device cuda:3 --wandb &
python train.py --model  longrange2 --n_batches 2000 --batch_size 32 --seq_len 128 --device cuda:4 --wandb &
python train.py --model  longrange2 --n_batches 2000 --batch_size 32 --seq_len 256 --device cuda:5 --wandb

wait

# python train.py --model transformer --n_batches 200 --batch_size 128 --seq_len 032 --lr 1e-3 --device cuda:0
# python train.py --model transformer --n_batches 200 --batch_size 128 --seq_len 064 --lr 1e-3 --device cuda:1
# python train.py --model transformer --n_batches 200 --batch_size 128 --seq_len 128 --lr 1e-3 --device cuda:2
# python train.py --model transformer --n_batches 200 --batch_size 128 --seq_len 256 --lr 1e-3 --device cuda:3

python train.py --model transformer --n_batches 2000 --batch_size 64 --seq_len 128 --device cuda:0
python train.py --model  longrange1 --n_batches 2000 --batch_size 32 --seq_len 128 --device cuda:1