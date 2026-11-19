# Semana 14 · Aprendizaje por refuerzo: reposición de un producto perecedero

Curso: Introducción a Machine Learning (CAD2092023103) · Universidad de Cundinamarca · Especialización en Analítica y Ciencia de Datos

Un agente aprende cuántas unidades pedir cada día para una panadería de barrio (producto con 3 días de vida útil y demanda Poisson con estacionalidad semanal). Se comparan **Q-Learning** y **SARSA** tabulares con la política óptima exacta (**Value Iteration**, solo como referencia) y con reglas fijas.

**Integrantes:** Brayan Yesid Vega Niño · 

## Estructura
```
notebooks/Semana14_RL_Brayan_Vega.ipynb   Análisis completo (listo para Google Colab)
scripts/rl_panaderia.py                   Mismo código en formato script
figures/                                  Figuras del informe (PNG)
data/                                     Parámetros, resultados (CSV) y tablas Q (NPY)
requirements.txt
```

## Cómo ejecutar
**Google Colab:** subir el notebook, o abrirlo desde GitHub, y ejecutar todas las celdas (no hay que instalar nada).

**Local:**
```bash
pip install -r requirements.txt
python scripts/rl_panaderia.py      # genera figures/ y data/ en la carpeta actual
```
La variable `RAPIDO` (al inicio) reduce el presupuesto de entrenamiento: `RAPIDO = True` tarda ~1 minuto pero da resultados menos precisos; `RAPIDO = False` (por defecto) reproduce el informe en ~5 minutos.

## Resultados principales (ganancia por día, miles de COP, demanda nueva)
| Política | Ganancia/día |
|---|---|
| Aleatoria | 1,381 |
| Mejor regla fija (S = 5) | 3,653 |
| SARSA | 3,457 ± 0,045 |
| **Q-Learning** | **3,759 ± 0,056** |
| Óptima (Value Iteration) | 3,914 |

El problema es simulado: demanda y costos son supuestos del caso, no datos reales.
