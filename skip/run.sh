# Learning Rate Sweep
python train.py --model transformer --batch_size 64 --seq_len 8 --seed 0 --lr 3e-3 --device cuda:0 --track --name lr_3e-3 &
python train.py --model transformer --batch_size 64 --seq_len 8 --seed 0 --lr 1e-3 --device cuda:0 --track --name lr_1e-3 &
python train.py --model transformer --batch_size 64 --seq_len 8 --seed 0 --lr 6e-4 --device cuda:1 --track --name lr_6e-4 &
python train.py --model transformer --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:1 --track --name lr_3e-4 &
python train.py --model transformer --batch_size 64 --seq_len 8 --seed 0 --lr 1e-4 --device cuda:2 --track --name lr_1e-4 &
python train.py --model transformer --batch_size 64 --seq_len 8 --seed 0 --lr 6e-5 --device cuda:2 --track --name lr_6e-5 &
python train.py --model transformer --batch_size 64 --seq_len 8 --seed 0 --lr 3e-5 --device cuda:3 --track --name lr_3e-5 &
python train.py --model transformer --batch_size 64 --seq_len 8 --seed 0 --lr 1e-5 --device cuda:3 --track --name lr_1e-5 &
wait

# Transformer Model Size Sweep
python train.py --model  transformer_small --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:0 --track --name transformer__small &
python train.py --model transformer_medium --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:1 --track --name transformer_medium &
python train.py --model  transformer_large --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:2 --track --name transformer__large &
python train.py --model transformer_xlarge --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:3 --track --name transformer_xlarge &
wait

# Longrange Model Size Sweep
python train.py --model  longrange_small --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:0 --track --name longrange__small &
python train.py --model longrange_medium --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:1 --track --name longrange_medium &
python train.py --model  longrange_large --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:2 --track --name longrange__large &
python train.py --model longrange_xlarge --batch_size 64 --seq_len 8 --seed 0 --lr 3e-4 --device cuda:3 --track --name longrange_xlarge &
wait

# Transformer Seq Length Sweep
python train.py --model transformer --batch_size 64 --seq_len 002 --seed 0 --lr 3e-4 --device cuda:0 --track --name transformer_002 &
python train.py --model transformer --batch_size 64 --seq_len 004 --seed 0 --lr 3e-4 --device cuda:1 --track --name transformer_004 &
python train.py --model transformer --batch_size 64 --seq_len 008 --seed 0 --lr 3e-4 --device cuda:2 --track --name transformer_008 &
python train.py --model transformer --batch_size 64 --seq_len 016 --seed 0 --lr 3e-4 --device cuda:3 --track --name transformer_016 &
python train.py --model transformer --batch_size 64 --seq_len 032 --seed 0 --lr 3e-4 --device cuda:0 --track --name transformer_032 &
python train.py --model transformer --batch_size 64 --seq_len 064 --seed 0 --lr 3e-4 --device cuda:1 --track --name transformer_064 &
wait

# Longrange Seq Length Sweep
python train.py --model longrange --batch_size 64 --seq_len 002 --seed 0 --lr 3e-4 --device cuda:0 --track --name longrange_002 &
python train.py --model longrange --batch_size 64 --seq_len 004 --seed 0 --lr 3e-4 --device cuda:1 --track --name longrange_004 &
python train.py --model longrange --batch_size 64 --seq_len 008 --seed 0 --lr 3e-4 --device cuda:2 --track --name longrange_008 &
python train.py --model longrange --batch_size 64 --seq_len 016 --seed 0 --lr 3e-4 --device cuda:3 --track --name longrange_016 &
python train.py --model longrange --batch_size 64 --seq_len 032 --seed 0 --lr 3e-4 --device cuda:0 --track --name longrange_032 &
python train.py --model longrange --batch_size 64 --seq_len 064 --seed 0 --lr 3e-4 --device cuda:1 --track --name longrange_064 &
wait
