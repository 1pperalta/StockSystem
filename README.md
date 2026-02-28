# StockSystem

Reinforcement Learning environment for stock and crypto trading. A DQN agent built manually with PyTorch learns to buy, sell, or hold assets to maximize portfolio profit. Trained and evaluated on 5 assets across different volatility profiles.

---

# StockSystem (Espanol)

Ambiente de aprendizaje por refuerzo para operar en el mercado de acciones y criptomonedas. Un agente DQN construido manualmente con PyTorch aprende a comprar, vender o mantener activos para maximizar la ganancia del portafolio. Entrenado y evaluado en 5 activos con diferentes perfiles de volatilidad.

---

## Environment Design / Diseno del Ambiente

| Component | Description |
|-----------|-------------|
| State | 5 daily percentage returns + invested ratio + cash ratio (7 floats) |
| Actions | Sell all, Sell half, Hold, Buy half, Buy all (Discrete 5) |
| Reward | Normalized change in total portfolio value per step |
| Penalties | Transaction fee (0.1%), hold penalty (1% per step), terminal loss penalty |
| Episode length | 60 trading days (training), 180 trading days (evaluation) |

The observation uses percentage returns instead of raw prices so the agent learns scale-invariant patterns that transfer across assets with very different price magnitudes.

El estado usa retornos porcentuales en lugar de precios absolutos, lo que permite al agente aprender patrones independientes de la escala que funcionan en activos con magnitudes de precio muy diferentes.

---

## Results / Resultados

Trained for 2000 episodes per asset (60-day windows). Evaluated on a 180-day window.

| Asset | Agent Profit | Buy-Hold Profit | Agent Advantage |
|-------|-------------|-----------------|-----------------|
| AAPL | +$399,819 | +$65,641 | +$334,178 |
| BTC-USD | +$360,920 | +$150,039 | +$210,881 |
| ETH-USD | +$360,738 | +$185,477 | +$175,261 |
| SOL-USD | +$223,467 | -$14,564 | +$238,031 |
| SCHD | +$64,263 | -$15,057 | +$79,320 |

All agents beat buy-and-hold on their respective assets. / Todos los agentes superaron la estrategia de comprar y mantener en sus activos respectivos.

---

## Evaluation Charts / Graficos de Evaluacion

![Agent Decisions on AAPL](images/evaluation_AAPL.png)
![Agent Decisions on BTC](images/evaluation_BTC-USD.png)
![Agent Decisions on ETH](images/evaluation_ETH-USD.png)
![Agent Decisions on SOL](images/evaluation_SOL-USD.png)
![Agent Decisions on SCHD](images/evaluation_SCHD.png)

---

## Project Structure / Estructura del Proyecto

```
stock_env.py                    - Custom Gymnasium environment
agent.py                        - MLP network, replay buffer, DQN agent
train.py                        - Training loop (2000 episodes per asset)
evaluate.py                     - Evaluation and chart generation
download/download_stock_info.py - Downloads historical data from Yahoo Finance
test/test.py                    - Environment test with random actions
data/                           - Stock price CSV files
images/                         - Result charts
```

---

## Setup / Instalacion

```bash
uv sync
```

## Usage / Uso

```bash
# Download stock data / Descargar datos
uv run python download/download_stock_info.py

# Test environment / Probar el ambiente
uv run python test/test.py

# Train the agent / Entrenar el agente
uv run python train.py

# Evaluate and generate charts / Evaluar y generar graficos
uv run python evaluate.py
```

---

## Tech Stack

- Python 3.11+
- Gymnasium
- PyTorch
- NumPy
- Pandas
- Matplotlib
- yfinance
