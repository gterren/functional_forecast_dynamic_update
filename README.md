# Day-ahead Load, Wind and Solar Resource Operation Forecast Dynamic Updating for Power Grids Planning

Having set of scenarios obtained from an unbiased day-ahead forecast of load demand, solar, and wind generation across Texas-7k 2030 grid. This respository explores a set of functional data analysis tools to determine the functional envolep of a forecast. In addition, a dynamic updating method is implemented and the performances are assesed for our problem.

## Functional Envelop

The functional envelop is defined by the depth notioned known as Modifided Band Depth (MBD). This depth notion ranks the forecasting scenarios quantifing the proportion of time that a scenarios is enveloped by each other pair of scenarios in the set. The hyperparameter $\alpha$ defines the number of scenarios ranked by MBD included in the set of scenarios to provide a desired confidence interval. The first part of **forecast_dynamic_updating.ipynb** notebook validates and test the $\alpha$ for multiple confidence intervals from 95% to 65%.

## Dynamic Updating

The next to sections in the **forecast_dynamic_updating.ipynb** notebook explores the dynamic update of a forecast as actual observation become available. The dynamic updating in this notebook evaluates the resulting forecast after updating it after the 1th to 23rd hourt. Notice that the day-ahead operation forecast has hourly resolution so a forecast is provided for each hour of following day.

### Functional k-Nearest Neighbors


**forecast_dynamic_updating.ipynb**
