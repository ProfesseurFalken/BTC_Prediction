# Crypto-Analysis-Tool

A Python script for analyzing Bitcoin price data, including technical indicators, trading signals, and price prediction.

## Overview

This project fetches Bitcoin (BTC-USD) price data from Yahoo Finance, calculates various technical indicators, generates trading signals based on these indicators, and provides a simple price prediction model. It's designed to give insights into possible trading opportunities by analyzing past price data.

### Features

- **Data Fetching**: Retrieves the last 90 days of hourly Bitcoin price data.
- **Technical Analysis**: 
  - Moving Averages (SMA, EMA)
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Volatility and Volume analysis
- **Trading Signals**: Generates buy and sell signals based on technical indicators.
- **Timing Analysis**: Identifies best trading hours, days, and times based on historical volatility and volume.
- **Price Prediction**: Uses ARIMA model via `pmdarima` for short-term price forecasting.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python libraries:
  - pandas
  - numpy
  - yfinance
  - matplotlib
  - pmdarima
  - sklearn

### Installation

1. **Clone the repository**:


   ```sh
   git clone https://github.com/yourusername/Crypto-Analysis-Tool.git
   cd Crypto-Analysis-Tool

2. Set up a virtual environment:

   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the necessary dependencies:

   pip install -r requirements.txt

   Usage
   To run the script:
   
   python main.py    This will fetch data, perform analysis, and display results via logs and plots.


Contributing  

  Contributions are welcome! Please follow these steps:

  Fork the project.
  Create your feature branch (git checkout -b feature/FooBar).
  Commit your changes (git commit -m 'Add some FooBar').
  Push to the branch (git push origin feature/FooBar).
  Open a pull request.

License
  This project is licensed under the MIT License - see the LICENSE.md file for details.

Acknowledgments
  Yahoo Finance API for providing financial data.
  pmdarima for ARIMA model implementation.

Contact
For any questions or suggestions, please open an issue or contact Emmanuel Jean-Louis Wojcik at wojcikej@orange.fr
