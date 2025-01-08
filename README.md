# **Heston Model Option Pricing**

## Overview
This Python application provides a graphical user interface (GUI) for pricing **European** and **American** options using the **Heston Model**. The application calculates option prices, visualizes the results, and compares European option prices with those obtained from the **Black-Scholes Model**.

---

## Features

### Option Pricing:
- **Heston Model:**
  - Prices **European** and **American** call and put options using Monte Carlo simulations.
  - Simulates stochastic volatility for more accurate modeling of market behavior.
  - Implements backward induction for American option pricing.

### Graphical Visualization:
- Displays interactive graphs showing option price evolution over time to maturity.
- Final option prices are displayed under each graph.
- Hover functionality to inspect option values at specific times.

### Customizable Parameters:
- Allows users to modify the following:
  - Spot price, strike price, risk-free rate, dividend yield.
  - Initial volatility, mean reversion rate (kappa), long-term volatility (theta).
  - Volatility of volatility (sigma) and correlation (rho).
  - Time to maturity.

### User-Friendly GUI:
- Built with `Tkinter` for a simple, intuitive interface.
- Real-time graph updates based on user inputs.

---

## Requirements

### Python 3.8+

### Libraries:
- `numpy`
- `matplotlib`
- `seaborn`
- `tkinter` (bundled with Python)

Install missing dependencies using:
```bash
pip install numpy matplotlib seaborn
```

---

## Installation
1. Clone the repository or copy the script file.
2. Install the required libraries (see above).
3. Run the script:
```bash
python main.py
```

---

## How to Use

### 1. Configure Parameters:
- Adjust the following inputs in the GUI:
  - Spot Price: Current price of the underlying asset.
  - Strike Price: The exercise price of the option.
  - Risk-Free Rate: The annualized risk-free interest rate (e.g., 0.05 for 5%).
  - Dividend Yield: Expected annual dividend yield of the underlying asset.
  - Volatility Parameters: Initial volatility, long-term volatility (theta), volatility of volatility (sigma), and mean reversion rate (kappa).
  - Time to Maturity: Option duration in days.

### 2. Generate Graphs:
- Click "Generate Graphs" to calculate and visualize the option prices.
- The GUI will display four graphs:
  - European Call Option Prices
  - European Put Option Prices
  - American Call Option Prices
  - American Put Option Prices

### 3. Interact with the Graphs:
- Hover over the graphs to inspect the option value at specific times.
- Final option prices are displayed below each graph.

---

## How It Works

### Heston Model:
- Simulates the underlying asset price and its volatility using stochastic differential equations.
- Calculates option prices by simulating multiple paths and averaging the payoffs.

### Backward Pricing for American Options:
- Uses backward induction to determine the optimal exercise strategy for American options.
- Incorporates early exercise premium.

### Black-Scholes Model:
- Calculates European option prices using the Black-Scholes formula for comparison.

---

## Notes
- Ensure valid input values to avoid errors during calculations.
- The application is for educational purposes and should not be used for financial decisions.

---

## Disclaimer
This application is for educational purposes only and does not constitute financial advice. Use at your own risk.

