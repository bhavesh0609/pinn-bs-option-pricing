# Physics-Informed Neural Network for Option Pricing

PyTorch implementation of a **Physics-Informed Neural Network (PINN)** for solving the **Black–Scholes–Merton (BSM) option-pricing PDE** using automatic differentiation, terminal/boundary constraints, and physics-based loss optimization.

## Key Features

- Analytical **Black–Scholes** benchmark for European call pricing.
- PINN with **Tanh neural layers** and automatic differentiation for first- and second-order PDE derivatives.
- Physics-informed training using **PDE, terminal, and boundary losses** with gradient clipping and learning-rate scheduling.
- Independent evaluation on **25K points** against the analytical BSM solution.
- Comparison with a parameter-matched **data-driven neural-network baseline**.
- PDE-residual, boundary/terminal-condition, error, moneyness, and Greek diagnostics.
- Market-option extension with a joint **price and state-dependent volatility network**.
- Market calibration with **PDE and Greek-based regularization** and held-out contract evaluation.
- Lightweight **ablation study** and an interactive **PINN vs. BSM explorer**.

## Results

| Experiment | Metric |
|---|---:|
| PINN vs. analytical BSM | **R² 0.999** |
| PINN vs. analytical BSM | **MAE 0.62** |
| Independent evaluation set | **25K points** |
| Market calibration | **62 option contracts** |

The data-driven network achieves lower pointwise pricing error when analytical labels are directly available, while the PINN provides physics-based constraints through the governing PDE and boundary conditions.

## Tech Stack

**Python · PyTorch · NumPy · Pandas · Matplotlib · yfinance · Jupyter Notebook**

## Repository

```text
├── BSM_PDE_solution_using_PINNs.ipynb
└── README.md
```

## Scope

The market experiment is a small, single-expiry cross-sectional calibration. A broader volatility-surface study would require multiple expirations and dates with temporal out-of-sample validation.
