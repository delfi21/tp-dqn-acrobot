# DQN en Acrobot-v1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/delfi21/tp-dqn-acrobot/blob/main/dqn_acrobot.ipynb)

Implementación de **Deep Q-Network** (Mnih et al., 2015) en PyTorch sobre `Acrobot-v1` de gymnasium,
para el TP final de *Introducción teórico-práctica al Aprendizaje por Refuerzo* (UBA, 2026).
Está basada en [`cleanrl/dqn.py`](https://docs.cleanrl.dev/rl-algorithms/dqn/).

## Cómo correrlo

1. Hacé clic en el botón **Open in Colab** de arriba.
2. En Colab: *Entorno de ejecución → Ejecutar todas*.

No hace falta instalar nada en la computadora. Para corridas largas, poné `USE_DRIVE = True`
en el notebook para guardar los resultados en Google Drive.

## Contenido

| Archivo | Qué tiene |
|---|---|
| `dqn_acrobot.ipynb` | El código completo: red Q, replay buffer, ε-greedy, target, loop de entrenamiento y gráficos |
| `results/` | Gráficos y resultados de los experimentos |

## Componentes de DQN

- **Red Q(s, a; θ):** MLP 6 → 120 → 84 → 3.
- **Replay buffer:** guarda transiciones y entrena con minibatches al azar, para que los datos se parezcan a i.i.d.
- **Target network θ⁻:** copia de la red que calcula los targets y se actualiza cada C pasos, para que el objetivo no se mueva en cada paso.
- **ε-greedy:** ε baja linealmente de 1 a 0,05.

## Resultados

_Pendiente._
