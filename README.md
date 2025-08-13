# ➕Integral-Calculator ![Static Badge](https://img.shields.io/badge/Original-blue?style=for-the-badge)
![Static Badge](https://img.shields.io/badge/Python-blue?logo=Python&logoColor=white)
![Static Badge](https://img.shields.io/badge/Jupyter_Notebook-orange?logo=Jupyter&logoColor=white)
![Static Badge](https://img.shields.io/badge/status-completed-green)

## 🗺️A world without numerical methods

Let's imagine a world where Riemann integral's were never meant to be computed through numerical methods. Could we build a tool that can do it for us? The answer is **yes**.

Usually, integrals are computed through summing the heights of a function multiplying them by a certain width, what is called the [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule). For high dimensions (multiple integrals) the computation might be difficult, specially when the bounds of the integral are strange or depend on the integration variables.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20241231110823198597/trapezoid_rule.webp" alt="Alt text" width="440" height="340"/>

The ''solution'' to this problem could be training a **Neural Network** (**NN**) to do the job for us. Obviously, for 1 variable functions the job could be counterproductive; this is just an exercise to learn [**TensorFlow**](https://www.tensorflow.org/?hl=es-419).

The example we will use will be the integration through the $(-3,3)$ interval.

## 📑Preparing the dataset

Firstly, we have to import the libraries that we will use.

```Python
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
```

The dataset to train the model will be a list of polynomial functions (of order 4) and they correspondant area. Since they are polynomial, the area can be calculated analytically (that was the whole starting point).

```math
\begin{align}
f(x)=&Ax^4+Bx^3+Cx^2+Dx+E \\
F(x)=&\frac A5x^5 + \frac B4 x^4 + \frac C3 x^3 + \frac D2 x^2 + Ex+K \\
\int_{-3}^3f(x)dx =& F(3)-F(-3)
\end{align}
```

The parameters $A,B,C,D,E,F$ of the function will be randomly generated.
