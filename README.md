# Jingwei: Hybrid Graph Learning Reconstructs Global Ocean Oxygen Spatiotemporal Changes

<div align="center">
  <img src="https://github.com/RobinLu1209/Jingwei/blob/main/readme_file/fig-main-fig1_Page1.png" alt="Logo" width="100%" />
</div>

## Requirements
- python 3.9
- pytorch 2.1+
- torch-geometric 2.4.0
- torch-scatter 2.1.2
- torch-sparse 0.6.18
- numpy 1.24.1

## How to Train and Test Jingwei

To run the code, simply execute the following command:

```bash
python main.py
```

This will run the code with the default parameters. If you want to customize the parameters, you can pass them via the command line. Below are the available command-line arguments you can set.

| Argument             | Type     | Default  | Description                                           |
|----------------------|----------|----------|-------------------------------------------------------|
| `--batch_size`        | `int`    | 32       | Batch size for training.                              |
| `--max_patience`      | `int`    | 10       | Maximum patience for early stopping.                 |
| `--lr`                | `float`  | 0.1      | Learning rate for training.                          |
| `--num_epochs`        | `int`    | 1000     | Number of epochs for training.                       |
| `--gpu`               | `int`    | 0        | GPU index to use for training (0 for the first GPU). |
| `--geo_dim`           | `int`    | 5        | Dimension of the DO geo factor.                      |
| `--hidden_dim`        | `int`    | 64       | Dimension of the hidden layer.                       |
| `--time_length`       | `int`    | 11       | Length of the time series.                           |

## Visualization of Reconstruction Result @YouTube

[![Reconstruction_Visualization](https://img.youtube.com/vi/sd5ytC_VPlU/0.jpg)](https://www.youtube.com/watch?v=sd5ytC_VPlU)

## Multi-Source Observation Dataset

In our work, we collect comprehensive dissolved oxygen observation data from five publicly available databases as follows.

| Database | Time | Institution | Source | Access Date|
|----------|------|-------------|--------|------------|
|World Ocean Database (WOD18) |  1900-2023 | National Centers for Environmental Information| [https://www.ncei.noaa.gov/](https://www.ncei.noaa.gov/) | 2023-05|
|CLIVAR and Carbon Hydrographic Database (CCHDO) | 1922-2023 | CLIVAR and Carbon Hydrographic Data Office | [https://cchdo.ucsd.edu/](https://cchdo.ucsd.edu/) | 2023-05 |
|Argo | 2001-2023 | Argo Global Data Assembly Center | [https://argo.ucsd.edu/](https://argo.ucsd.edu/) | 2023-05 |
|Global Ocean Data Analysis Project version2.2022 (GLODAPV2_2022)| 1972-2021| NOAA’s National Centers for Environmental Information (NCEI) | [https://glodap.info/](https://glodap.info/) |  2023-05 |
|Geotraces IDP| 2007-2018| GEOTRACES International Data Assembly Centre (GDAC) |  [https://www.geotraces.org](https://www.geotraces.org) | 2023-10|

## The Name of "Jingwei"

[Jingwei](https://en.wikipedia.org/wiki/Jingwei) is a classic figure in Chinese mythology, featured in the ["Shan Hai Jing"](https://en.wikipedia.org/wiki/Jingwei). 
The story tells of Jingwei, the daughter of Emperor Yan, who drowned in the East Sea. She was reborn as a bird and decided to fill the sea with pebbles and twigs, endeavoring to prevent similar tragedies. Today, Jingwei symbolizes perseverance and determination, embodying the spirit of never giving up despite difficult challenges.

In our work, we thank all marine scientists, researchers, and technicians who tirelessly venture into the field to measure oceanic dissolved oxygen data. Although their collected data might seem as extremely sparse as the pebbles and twigs Jingwei used, their persistent efforts aim to reveal the patterns of global ocean deoxygenation. To honor the spirit of these ocean explorers, we have named the AI-driven algorithm for reconstructing ocean deoxygenation proposed in this paper "Jingwei" and designed its logo as follows. 

<div align="center">
  <img src="https://github.com/RobinLu1209/Jingwei/blob/main/readme_file/jingwei_logo.jpg" alt="Logo" width="25%" />
</div>

## Meet the Team

Our project is supported by a diverse and talented group of experts from various fields. The team is divided into two main groups: the **Information Science Team** and the **Oceanography Team**. Below, you will find an overview of the members in each group, along with their academic backgrounds, research interests, and current roles.

### Information Science Team

| Name | Position| Research Focus | Affiliation |
|------|---------|----------------|-------------|
|[Xinbing Wang](https://www.cs.sjtu.edu.cn/~wang-xb/)|Distinguished Professor|Big Data, Knowledge Graph|Shanghai Jiao Tong University|
|[Xiaoying Gan](https://xiaoyinggan.acemap.info/index.html)|Professor|Data Mining, Crowd Computing|Shanghai Jiao Tong University|
|[Luoyi Fu](https://www.cs.sjtu.edu.cn/~fu-ly/index.html)|Associate Professor|Data-driven IoT, Graph Network|Shanghai Jiao Tong University|
|[Meng Jin](https://yume-sjtu.github.io/)|Associate Professor|IoT, AI for Science|Shanghai Jiao Tong University|
|[Bin Lu](https://robinlu1209.github.io/)|Postdoc|Graph Neural Network, GeoAI|Shanghai Jiao Tong University|
|Ze Zhao|PhD Student|Knowledge Graph|Shanghai Jiao Tong University|
|Haonan Qi|Master Student|AI for Ocean|Shanghai Jiao Tong University|

### Oceanography Team
| Name | Position| Research Focus | Affiliation |
|------|---------|----------------|-------------|
|[Jing Zhang](https://www.researchgate.net/profile/Jing-Zhang-583)|Academician of CAS, Professor| Biogeochemistry and Chemical Oceanography|East China Normal University|
|[Chenghu Zhou](http://english.igsnrr.cas.cn/sourcedb/yw_30508/scientists/En_sklreis/202012/t20201211_456387.html)|Academician of CAS, Professor|Geographic Information Systems, Spatio-temporal Data Mining|The Institute of Geographic Sciences and Natural Resources Research, Chinese Academy of Sciences|
|[Lei Zhou](https://soo-old.sjtu.edu.cn/en/szjyry/3593.html)|Distinguished Professor|Ocean and atmosphere dynamics|Shanghai Jiao Tong University|
|[Lixin Qu](https://soo-old.sjtu.edu.cn/en/szjyry/4416.html)|Tenure-track Associate Professor|Ocean submesoscale processes, Artificial intelligence in oceanography|Shanghai Jiao Tong University|
|[Yuntao Zhou](https://soo-old.sjtu.edu.cn/en/szjyry/4030.html)|Associate Professor|Oceanic Oxygen, Climate Change, Statistics and Geostatistics|Shanghai Jiao Tong University|
|Luyu Han|Master Student|Climate Change, Artificial intelligence in oceanography|Shanghai Jiao Tong University, University of California, San Diego|
|Jingjing Shen|Master Student|Climate Change, Artificial intelligence in oceanography|Shanghai Jiao Tong University|


## Contact

Bin Lu (robinlu1209@sjtu.edu.cn)
