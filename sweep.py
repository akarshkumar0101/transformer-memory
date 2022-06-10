
import experiment

import numpy as np

def main():

    np.random.seed(0)
    hm = True

    beta1s = [.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    beta2s = [.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    beta3s = [.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lrs = [1., 0.1, 0.01]


    for i in range(24):
        beta1 = np.random.choice(beta1s)
        beta2 = np.random.choice(beta2s)
        beta3 = np.random.choice(beta3s)
        alpha = np.random.choice(alphas)

        lr = np.random.choice(lrs)

        adaptive_alpha = np.random.choice([False])

        args = '--device cuda:0 --wandb'
        if hm:
            args += ' --hm' 
        args += f' --beta1 {beta1} --beta2 {beta2} --beta3 {beta3} --alpha {alpha} --opt sgd --lr {lr}'
        if adaptive_alpha:
            args += ' --adaptive_alpha' 

        args = args.split()

        print(args)
        experiment.main(args)



if __name__=="__main__":
    main()