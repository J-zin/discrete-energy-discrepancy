This branch contains the code for tabular data modelling at section 6.2.

## Installation

Our implementation is based on the [tabsyn](https://github.com/amazon-science/tabsyn/tree/main) repository. To set up the environment, please follow the installation instructions provided in that repository. The main functionality of our code closely mirrors the original repo, and we provide detailed usage instructions below.

## Training

To train the model, you can use the following command:
```bash
python main.py --dataname NAME_OF_DATASET --method NAME_OF_BASELINE_METHODS --mode train
```
- `NAME_OF_DATASET` is one of: adult, bank, cardio, churn, mushroom, beijing
- `NAME_OF_BASELINE_METHODS` is one of: ed_uni, ed_grid, ed_ord, ed_cyc

## Sampling
After training, you can sample from the model using the following command:
```bash
python main.py --dataname NAME_OF_DATASET --method NAME_OF_BASELINE_METHODS --mode sample
```

## Evaluation
After sampling, you can use the following command to evaluate the model:
```bash
python eval/eval_mle.py --dataname NAME_OF_DATASET --model METHOD_NAME --path PATH_TO_SYNTHETIC_DATA
```

## Example
Here is an example of how to train, sample, and evaluate the model on the adult dataset using the ed_uni method:
```bash
python main.py --dataname adult --method ed_uni --mode train
python main.py --dataname adult --method ed_uni --mode sample
python eval/eval_mle.py --dataname adult --model ed_uni --path synthetic/adult/ed_uni.csv
```
