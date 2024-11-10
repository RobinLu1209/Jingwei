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

## Contact

Bin Lu (robinlu1209@sjtu.edu.cn)
