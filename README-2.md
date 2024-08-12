
# Stock Selection and Clustering for Portfolio Optimization

## Overview

This project involves the selection and clustering of stocks from the S&P 500 index to optimize a diversified portfolio. The process includes downloading historical stock data, performing data preprocessing, and applying unsupervised learning techniques to cluster the stocks based on their performance metrics.

## How to Run the Code

### Prerequisites

- Python 3.x
- Required libraries (see `requirements.txt`)

### Installation

1. Create a virtual environment (optional but recommended):
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the required libraries:
   ```
   pip install -r requirements.txt
   ```

### Running the Code

1. Ensure that you have access to the internet as the code downloads data from Yahoo Finance.
2. Run the Jupyter Notebook file (`Capstone_Stock_Selection_stock_data.ipynb`) in a Jupyter environment.
   ```
   jupyter notebook Capstone_Stock_Selection_stock_data.ipynb
   ```
3. Follow the steps in the notebook to reproduce the results and generate figures.

## Project Structure

- `Capstone_Stock_Selection_stock_data.ipynb`: Main notebook with all code and analysis.
- `requirements.txt`: List of Python libraries needed to run the code.
- `README.md`: This file, containing instructions on how to run the project.

## Data Access Statement

The stock data used in this project is obtained from Yahoo Finance using the `yfinance` library. The data is freely available for non-commercial use. Ensure compliance with Yahoo Finance's data usage policies if the data is used for purposes other than personal or academic projects.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
