This branch contains the code for image modelling at section 6.3.

To train EBMs with energy discrepancy, use the following command:
```bash
python train_ed.py --dataset_name <dataset> --algo <method>
```
- `dataset` is the name of the dataset, one of <static_mnist|dynamic_mnist|omniglot>
- `method` is the name of the method, one of <ed_grid|ed_bern>

After training, you can run `eval_ais.py` to evaluate the learned EBM using AIS
```bash
python eval_ais.py \
    --dataset_name static_mnist \
    --algo ed_grid \
    --sampler gwg \
    --model resnet-64 \
    --n_samples 500 \
    --eval_sampling_steps 300000 \
    --viz_every 1000
```

We also provide the code to train a GflowNet to improve the qualitative sample quality as shown in Figure 10. To train the GflowNet given a pretrained EBM, use the following command:
```bash
python train_gflownet.py --ckpt_path <ckpt_path>
```
- `ckpt_path` is the path to the checkpoint of pretrained EBMs

After training, you can run `sample.py` to visualise the generated images
```bash
python sample.py --ckpt_ebm <ckpt_ebm> --ckpt_gfn <ckpt_gfn>
```
- `ckpt_ebm` is the path to the EBM checkpoint
- `ckpt_gfn` is the path to the GFlowNet checkpoint

## Acknowledgement

This implementation is based on [Gibbs_With_Gradient](https://github.com/wgrathwohl/GWG_release), [Discrete_Langevin](https://github.com/ruqizhang/discrete-langevin), and [GFlowNet_EBM](https://github.com/GFNOrg/EB_GFN). Thanks for the amazing work!
