# Application of ordinary differential equations to model a gene regulatory network 

Mathematical modelling is frequently used in systems biology to develop and simulate models of gene regulatory networks and signalling pathways. Such models are often built on systems of differential equations, including ordinary differential equations, and simulate the rate of change in gene expression, mRNA and protein synthesis and decay over time. For the model predictions to be considered accurate, the model has to account for the complex dynamics of gene regulatory networks and include quantitative parameters, which is why Hill kinetics and Michaelis-Menten equation are often implemented to define activation, repression, and strength of the regulatory mechanisms. 

In this notebook, we are going to model a hypothetical gene regulatory network consisting of three genes that form a negative feedback loop coupled with an additional activation mechanism, demonstrating the interplay of the regulatory network over time. 

Before we start, make sure you import the following required libraries:
```
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
%matplotlib inline
```

We are modelling a gene regulatory network where gene G1 activates genes G2 and G3, G2 activates G3, and G3 inhibits G1. The model is defined by the following equations:

<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

$$\frac{d[G1](t)}{dt} = k_{1} \frac{1}{1 + K_{13} \cdot [G3]} - d_{1} \cdot [G1]$$

$$\frac{d[G2](t)}{dt} = k_{2} \frac{K_{21} \cdot [G1]}{1 + K_{21} \cdot [G1]} - d_{2} \cdot [G2]$$

$$\frac{d[G3](t)}{dt} = k_{3} \frac{K_{31} \cdot [G1] \cdot K_{32} \cdot [G2]}{(1 + K_{31} \cdot [G1]) \cdot (1 + K_{32} \cdot [G2])} - d_{3} \cdot [G3]$$


