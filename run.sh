# python main.py --dataname adult --method ed_uni --mode train --gpu 0
# python main.py --dataname adult --method ed_uni --mode sample --gpu 0
# python eval/eval_mle.py --dataname adult --model ed_uni --path synthetic/adult/ed_uni.csv

python main.py --dataname adult --method ed_grid --mode train --gpu 0
# python main.py --dataname adult --method ed_uni --mode sample --gpu 0
# python eval/eval_mle.py --dataname adult --model ed_uni --path synthetic/adult/ed_uni.csv