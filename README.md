<h1 align="center">Discrete Energy Discrepancy</h1>
<p align="center">
    <a href="https://neurips.cc/virtual/2024/poster/93171"> <img alt="License" src="https://img.shields.io/static/v1?label=Pub&message=NeurIPS%2724&color=blue"> </a>
    <a href="https://nips.cc/virtual/2024/poster/93171"> <img src="https://img.shields.io/badge/Video-grey?logo=Kuaishou&logoColor=white" alt="Video"></a>
    <a href="https://nips.cc/virtual/2024/poster/93171"> <img src="https://img.shields.io/badge/Slides-grey?&logo=MicrosoftPowerPoint&logoColor=white" alt="Slides"></a>
    <a href="https://nips.cc/virtual/2024/poster/93171"> <img src="https://img.shields.io/badge/Poster-grey?logo=airplayvideo&logoColor=white" alt="Poster"></a>
</p>

This repo contains PyTorch implementation of the paper "[Energy-Based Modelling for Discrete and Mixed Data via Heat Equations on Structured Spaces]()"

by [Tobias Schröder](https://tobias-schroeder.github.io/), [Zijing Ou](https://j-zin.github.io/), [Yingzhen Li](http://yingzhenli.net/home/en/) and [Andrew Duncan](https://www.imperial.ac.uk/people/a.duncan).

> We propose discrete energy discrepancy, a loss function which only requires the evaluation of the energy function at data points and their perturbed counterparts, thus eliminating the need for Markov chain Monte Carlo. We introduce perturbations of the data distribution by simulating a diffusion process on the discrete state space endowed with a graph structure. This allows us to inform the choice of perturbation from the structure of the modelled discrete variable, while the continuous time parameter enables fine-grained control of the perturbation.

## Experiments

We provide the code for different settings under different branches:
- The [`density_estimation`](https://github.com/J-zin/discrete-energy-discrepancy/tree/density_estimation) branch contains the experiment of discrete density estimation at section 6.1.
- The [`tabular_synthetic`](https://github.com/J-zin/discrete-energy-discrepancy/tree/tabular_synthetic) branch contains the experiment of synthetic tabular data modelling at section 6.2.
- The [`tabular_modelling`](https://github.com/J-zin/discrete-energy-discrepancy/tree/tabular_modelling) branch contains the experiment of real-world tabular data modelling at section 6.2.
- The [`image_modelling`](https://github.com/J-zin/discrete-energy-discrepancy/tree/image_modelling) branch contains the experiment of image data modelling at section 6.3.

Please also check [this repository](https://github.com/J-zin/energy-discrepancy) for the code of [continuous energy discrepancy](https://openreview.net/pdf?id=1qFnxhdbxg)!!!