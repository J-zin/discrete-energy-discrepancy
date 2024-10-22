This branch contains the code for discrete density estimation at section 6.1.

To run the code, you can use the following command:
```bash
python main.py --data_name <dataset> --methods <method> --vocab_size 2   # 2 states, 32 dimensions
python main.py --data_name <dataset> --methods <method> --vocab_size 5   # 5 states, 16 dimensions
python main.py --data_name <dataset> --methods <method> --vocab_size 10 --discrete_dim 12   # 10 states, 12 dimensions
```

* `data_name` is the name of the dataset, one of <`moons`|`swissroll`|`circles`|`8gaussians`|`pinwheel`|`2spirals`|`checkerboard`>
* `method` is the name of the method, one of <`ed_grid`|`cd_gibbs`>

We also provide the notebooks to quickly reproduce the results

- `ed_binary_toy.ipynb`: training EBMs on binary spaces with 32 dimensions and 2 states
- `ed_category_toy.ipynb`: training EBMs on discrete spaces with 16 dimensions and 5 states
