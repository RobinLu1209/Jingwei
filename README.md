# Jingwei: Hybrid Graph Learning Reconstructs Global Ocean Oxygen Spatiotemporal Changes

<div align="center">
  <img src="https://github.com/RobinLu1209/Jingwei/blob/main/readme_file/jingwei_logo.jpg" alt="Logo" width="25%" />
</div>

## Requirements
- python 3.9
- pytorch 2.1+
- torch-geometric 2.4.0
- torch-scatter 2.1.2
- torch-sparse 0.6.18
- numpy 1.24.1

## How to Train Jingwei

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

## Meet the Team

Our project is supported by a diverse and talented group of experts from various fields. The team is divided into two main groups: the **Information Science Team** and the **Oceanography Team**. Below, you will find an overview of the members in each group, along with their academic backgrounds, research interests, and current roles.

### Information Science Team

| Name | Position| Research Focus | Affiliation |
|------|---------|----------------|-------------|
|[Xinbing Wang](https://www.cs.sjtu.edu.cn/~wang-xb/)|Distinguished Professor|Big Data, Knowledge Graph|Shanghai Jiao Tong University|
|[Xiaoying Gan](https://xiaoyinggan.acemap.info/index.html)|Professor|Data Mining, Crowd Computing|Shanghai Jiao Tong University|
|[Luoyi Fu](https://www.cs.sjtu.edu.cn/~fu-ly/index.html)|Associate Professor|Data-driven IoT, Graph Network|Shanghai Jiao Tong University|
|[Meng Jin](https://yume-sjtu.github.io/)|Associate Professor|IoT, AI for Science|Shanghai Jiao Tong University|
|[Bin Lu](https://robinlu1209.github.io/)|PhD Student|Graph Neural Network, GeoAI|Shanghai Jiao Tong University|
|Ze Zhao|Master Student|Knowledge Graph|Shanghai Jiao Tong University|
|Haonan Qi|Master Student|AI for Ocean|Shanghai Jiao Tong University|

### Oceanography Team
| Name | Position| Research Focus | Affiliation |
|------|---------|----------------|-------------|
|[Jing Zhang](https://www.researchgate.net/profile/Jing-Zhang-583)|Academician of CAS, Professor| Biogeochemistry and Chemical Oceanography|East China Normal University|
|[Chenghu Zhou](http://english.igsnrr.cas.cn/sourcedb/yw_30508/scientists/En_sklreis/202012/t20201211_456387.html)|Academician of CAS, Professor|Geographic Information Systems, Spatio-temporal Data Mining|The Institute of Geographic Sciences and Natural Resources Research, Chinese Academy of Sciences|
|[Lei Zhou](https://soo-old.sjtu.edu.cn/en/szjyry/3593.html)|Distinguished Professor|Ocean and atmosphere dynamics|Shanghai Jiao Tong University|
|[Lixin Qu](https://soo-old.sjtu.edu.cn/en/szjyry/4416.html)|Tenure-track Associate Professor|Ocean submesoscale processes, Artificial intelligence in oceanography|Shanghai Jiao Tong University|
|[Yuntao Zhou](https://soo-old.sjtu.edu.cn/en/szjyry/4030.html)|Associate Professor|Oceanic Oxygen, Climate Change，Statistics and Geostatistics|Shanghai Jiao Tong University|
|Luyu Han|Master Student|Climate Change，Artificial intelligence in oceanography|Shanghai Jiao Tong University, University of California, San Diego|


## Contact

Bin Lu (robinlu1209@sjtu.edu.cn)
