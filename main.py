import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from typing import Dict, Optional, Tuple, List
from matplotlib.axes import Axes
from dataclasses import dataclass


@dataclass
class HestonParameters:
    """Parameters for the Heston stochastic volatility model."""
    spot_price: float = 100.0
    strike_price: float = 100.0
    time_to_maturity: int = 365
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.02
    initial_volatility: float = 0.1
    mean_reversion_rate: float = 2.0
    long_term_volatility: float = 0.1
    volatility_of_volatility: float = 0.2
    correlation: float = -0.5


class HestonModel:
    """Implementation of the Heston stochastic volatility model."""

    def __init__(self, params: HestonParameters, num_simulations: int = 10000,
                 num_time_steps: int = 252):
        self.params = params
        self.num_simulations = num_simulations
        self.num_time_steps = num_time_steps
        self._setup_random_numbers()

    def _setup_random_numbers(self) -> None:
        """Generate correlated random numbers for Monte Carlo simulation."""
        np.random.seed(42)
        self.z1 = np.random.normal(size=(self.num_simulations, self.num_time_steps))
        self.z2 = (self.params.correlation * self.z1 +
                   np.sqrt(1 - self.params.correlation ** 2) *
                   np.random.normal(size=(self.num_simulations, self.num_time_steps)))

    def simulate_paths(self) -> np.ndarray:
        """
        Simulate price paths using the Heston model.

        Returns:
            np.ndarray: Array of simulated price paths
        """
        dt = (self.params.time_to_maturity / 365) / self.num_time_steps

        vt = np.zeros_like(self.z1)
        vt[:, 0] = self.params.initial_volatility

        St = np.zeros_like(self.z1)
        St[:, 0] = self.params.spot_price

        for i in range(1, self.num_time_steps):
            vt[:, i] = (vt[:, i - 1] +
                        self.params.mean_reversion_rate *
                        (self.params.long_term_volatility - vt[:, i - 1]) * dt +
                        self.params.volatility_of_volatility *
                        np.sqrt(np.maximum(0, vt[:, i - 1] * dt)) * self.z2[:, i])

            St[:, i] = (St[:, i - 1] * np.exp(
                (self.params.risk_free_rate - self.params.dividend_yield - 0.5 * vt[:, i]) * dt +
                np.sqrt(np.maximum(0, vt[:, i] * dt)) * self.z1[:, i]))

        return St


class OptionPricer:
    """Calculate option prices using Monte Carlo simulation."""

    def __init__(self, params: HestonParameters, stock_paths: np.ndarray):
        self.params = params
        self.stock_paths = stock_paths
        self.num_time_steps = stock_paths.shape[1]

    def calculate_values(self) -> Tuple[np.ndarray, np.ndarray, List[float], List[float],
    float, float, float, float]:
        """
        Calculate European and American option values.

        Returns:
            Tuple containing arrays of option values and final prices
        """
        dt = (self.params.time_to_maturity / 365) / self.num_time_steps

        call_payoffs = np.maximum(self.stock_paths - self.params.strike_price, 0)
        put_payoffs = np.maximum(self.params.strike_price - self.stock_paths, 0)

        time_points = np.linspace(0, self.params.time_to_maturity / 365, self.num_time_steps)
        discount_factors = np.exp(-self.params.risk_free_rate * time_points)

        european_call = np.mean(call_payoffs, axis=0) * discount_factors
        european_put = np.mean(put_payoffs, axis=0) * discount_factors

        american_call = self._calculate_american_option(call_payoffs, dt)
        american_put = self._calculate_american_option(put_payoffs, dt)

        return (
            european_call, european_put,
            american_call, american_put,
            european_call[-1], european_put[-1],
            american_call[0], american_put[0]
        )

    def _calculate_american_option(self, payoffs: np.ndarray, dt: float) -> List[float]:
        """
        Calculate American option values using backward induction.

        Args:
            payoffs: Array of option payoffs
            dt: Time step size

        Returns:
            List of option values
        """
        values = payoffs[:, -1]
        path = [np.mean(values)]

        for i in range(self.num_time_steps - 2, -1, -1):
            values = np.maximum(
                payoffs[:, i],
                np.exp(-self.params.risk_free_rate * dt) * values
            )
            path.append(np.mean(values))

        return list(reversed(path))


class OptionPricingGUI:
    """GUI for the Heston model option pricing application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Heston Model Option Pricer")
        self._setup_layout()
        self._setup_parameters()
        self._setup_results_section()
        self._setup_plots()

    def _setup_layout(self) -> None:
        """Set up the main GUI layout."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.main_frame, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _setup_parameters(self) -> None:
        """Set up parameter input fields."""
        param_frame = ttk.LabelFrame(self.left_frame, text="Model Parameters", padding=10)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        default_params = HestonParameters()
        self.params = {
            "Spot Price": tk.DoubleVar(value=default_params.spot_price),
            "Strike Price": tk.DoubleVar(value=default_params.strike_price),
            "Risk-Free Rate": tk.DoubleVar(value=default_params.risk_free_rate),
            "Dividend Yield": tk.DoubleVar(value=default_params.dividend_yield),
            "Initial Volatility": tk.DoubleVar(value=default_params.initial_volatility),
            "Mean Reversion Rate (Kappa)": tk.DoubleVar(value=default_params.mean_reversion_rate),
            "Long-Term Volatility (Theta)": tk.DoubleVar(value=default_params.long_term_volatility),
            "Volatility of Volatility (Sigma)": tk.DoubleVar(value=default_params.volatility_of_volatility),
            "Brownian Motion Correlation (Rho)": tk.DoubleVar(value=default_params.correlation),
            "Time to Maturity (Days)": tk.IntVar(value=default_params.time_to_maturity)
        }

        for param, var in self.params.items():
            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{param}:", anchor="w",
                      width=25).pack(side=tk.LEFT, padx=5)
            ttk.Entry(frame, textvariable=var).pack(side=tk.RIGHT,
                                                    expand=True, fill=tk.X)

        ttk.Button(self.left_frame, text="Calculate Options",
                   command=self.generate_graphs).pack(pady=10, fill=tk.X)

    def _setup_results_section(self) -> None:
        """Set up the results section in the left frame."""
        results_frame = ttk.LabelFrame(self.left_frame, text="Final Values", padding=10)
        results_frame.pack(fill=tk.X, pady=10)

        self.result_vars = {
            "European Call": tk.StringVar(value="-"),
            "European Put": tk.StringVar(value="-"),
            "American Call": tk.StringVar(value="-"),
            "American Put": tk.StringVar(value="-")
        }

        for label, var in self.result_vars.items():
            frame = ttk.Frame(results_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{label}:", anchor="w").pack(side=tk.LEFT, padx=5)
            ttk.Label(frame, textvariable=var).pack(side=tk.RIGHT, padx=5)

    def _setup_plots(self) -> None:
        """Set up the matplotlib plots."""
        sns.set_theme(style="whitegrid")
        self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.annotations = {ax: None for ax in self.axs.flatten()}
        self.canvas.mpl_connect("motion_notify_event", self._handle_hover)

    def _handle_hover(self, event) -> None:
        """
        Handle mouse hover events on the plots.

        Args:
            event: Mouse event object
        """
        if not event.inaxes:
            self._clear_annotations()
            return

        ax = event.inaxes
        for line in ax.get_lines():
            xdata, ydata = line.get_data()
            if len(xdata) == 0:
                continue

            idx = np.searchsorted(xdata, event.xdata)
            idx = min(max(idx, 1), len(xdata) - 1)
            x_closest = xdata[idx]
            y_closest = ydata[idx]

            x_display, y_display = ax.transData.transform((x_closest, y_closest))
            distance = np.sqrt((event.x - x_display) ** 2 +
                               (event.y - y_display) ** 2)

            if distance < 10:
                self._update_annotation(ax, x_closest, y_closest)
                return

        self._clear_annotations()

    def _update_annotation(self, ax: Axes, x: float, y: float) -> None:
        """
        Update or create annotation on plot.

        Args:
            ax: Matplotlib axes object
            x: X-coordinate for annotation
            y: Y-coordinate for annotation
        """
        if self.annotations[ax] is None:
            self.annotations[ax] = ax.annotate(
                f"({x:.2f}, {y:.2f})",
                xy=(x, y),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3",
                          edgecolor="black", facecolor="white"),
                arrowprops=dict(arrowstyle="->", color="black")
            )
        else:
            self.annotations[ax].xy = (x, y)
            self.annotations[ax].set_text(f"({x:.2f}, {y:.2f})")
            self.annotations[ax].set_visible(True)
        self.canvas.draw()

    def _clear_annotations(self) -> None:
        """Clear all annotations from plots."""
        for annotation in self.annotations.values():
            if annotation:
                annotation.set_visible(False)
        self.canvas.draw()

    def generate_graphs(self) -> None:
        """Generate and update option price plots."""
        params = HestonParameters(
            spot_price=self.params["Spot Price"].get(),
            strike_price=self.params["Strike Price"].get(),
            time_to_maturity=self.params["Time to Maturity (Days)"].get(),
            risk_free_rate=self.params["Risk-Free Rate"].get(),
            dividend_yield=self.params["Dividend Yield"].get(),
            initial_volatility=self.params["Initial Volatility"].get(),
            mean_reversion_rate=self.params["Mean Reversion Rate (Kappa)"].get(),
            long_term_volatility=self.params["Long-Term Volatility (Theta)"].get(),
            volatility_of_volatility=self.params["Volatility of Volatility (Sigma)"].get(),
            correlation=self.params["Brownian Motion Correlation (Rho)"].get()
        )

        model = HestonModel(params)
        stock_paths = model.simulate_paths()
        pricer = OptionPricer(params, stock_paths)

        results = pricer.calculate_values()
        self._update_plots(results, params.time_to_maturity)
        self._update_results(results[4:])

    def _update_results(self, final_values: tuple) -> None:
        """
        Update the final values in the left panel.

        Args:
            final_values: Tuple of final option values
        """
        final_european_call, final_european_put, final_american_call, final_american_put = final_values

        self.result_vars["European Call"].set(f"{final_european_call:.4f}")
        self.result_vars["European Put"].set(f"{final_european_put:.4f}")
        self.result_vars["American Call"].set(f"{final_american_call:.4f}")
        self.result_vars["American Put"].set(f"{final_american_put:.4f}")

    def _update_plots(self, results: Tuple, maturity: int) -> None:
        """
        Update all plots with new data.

        Args:
            results: Tuple containing option values and final prices
            maturity: Time to maturity in days
        """
        (european_call, european_put, american_call, american_put,
         final_european_call, final_european_put,
         final_american_call, final_american_put) = results

        time_steps = np.linspace(0, maturity, len(european_call))

        for ax in self.axs.flatten():
            ax.clear()

        plot_configs = [
            (self.axs[0, 0], european_call, "blue", "European Call", final_european_call),
            (self.axs[0, 1], european_put, "green", "European Put", final_european_put),
            (self.axs[1, 0], american_call, "orange", "American Call", final_american_call),
            (self.axs[1, 1], american_put, "red", "American Put", final_american_put)
        ]

        for ax, data, color, label, final_value in plot_configs:
            ax.plot(time_steps, data, color=color, label=f"{label} (Path)")
            ax.set_title(f"{label} Option (Final: {final_value:.4f})")
            ax.set_xlabel("Time to Maturity (Days)")
            ax.set_ylabel("Option Value")
            ax.legend()

        self.fig.tight_layout()
        self.canvas.draw()
        self.annotations = {ax: None for ax in self.axs.flatten()}


# Main Function
def main():
    """Main Function"""
    root = tk.Tk()
    app = OptionPricingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()