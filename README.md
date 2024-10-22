This branch contains the code for synthetic tabular data modelling at section 6.2.


We provide 4 variant ED-based methods to train EBMs on synthetic tabular data. To run the code, use the following commands:

```bash
# uniform perturbation
python main.py --dataname rings --method ed_uni --mode train

# grid perturbation
python main.py --dataname rings --method ed_grid --mode train

# cyclic perturbation
python main.py --dataname rings --method ed_cyc --mode train --cat_tnoise 0.005

# ordinal perturbation
python main.py --dataname rings --method ed_ord --mode train --cat_tnoise 0.01

```

To train EBMs with contrastive divergence, use the following commands:

```bash
# uniform perturbation
python main.py --dataname rings --method cd_gibbs --mode train
```