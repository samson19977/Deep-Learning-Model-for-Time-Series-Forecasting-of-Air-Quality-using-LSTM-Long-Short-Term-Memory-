# Air Quality PM2.5 Prediction with LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict PM2.5 concentrations based on historical air quality data. The model leverages multiple features (temperature, pressure, dew point, wind speed) over the past 24 hours to forecast the next hour's PM2.5 level.

The dataset used is the **Beijing PM2.5 Data** from the UCI Machine Learning Repository.

---

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Prediction on New Data](#prediction-on-new-data)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

---

## Dataset

The dataset contains hourly air quality measurements from Beijing. It includes the following features:

- `pm2.5`: PM2.5 concentration (target variable)
- `TEMP`: Temperature (in Celsius)
- `PRES`: Pressure (in hPa)
- `DEWP`: Dew point temperature (in Celsius)
- `WSPM`: Wind speed (in m/s)

You can download the dataset from the [UCI repository](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data) (file `PRSA_data_2010.1.1-2014.12.31.csv`).  
Place the CSV file at the path specified in `CONFIG['data_path']` (by default, `C:/Users/Francis Musoke/Downloads/Air Quality.csv`).  
Alternatively, change the path in the configuration.

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- pandas
- Matplotlib
- scikit-learn

All required packages are listed in `requirements.txt`.

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/air-quality-lstm.git
   cd air-quality-lstm
