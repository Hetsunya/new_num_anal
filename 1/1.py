import numpy as np
import pandas as pd

def F(x):
    x1, x2 = x
    f1 = np.sinh(x1 + 0.2*x2 + np.tan(0.1*x1*x2)) - 0.8
    f2 = np.sinh(0.6*x1 - 0.1*x2 + np.tan(0.2*x1*x2)) - 0.1
    return np.array([f1, f2])

def simple_iteration(phi, x0, eps=1e-6, max_iter=1000):
    x_prev = np.array(x0, dtype=float)
    for k in range(max_iter):
        x_next = phi(x_prev)
        if np.linalg.norm(x_next - x_prev, ord=np.inf) < eps:
            return x_next, k+1, F(x_next)
        x_prev = x_next
    raise RuntimeError("Метод не сошелся")

λ = 0.1
def phi(x):
    return x - λ * F(x)

x0 = [0.5, 0.5]
sol, iters, Fval = simple_iteration(phi, x0, eps=1e-6)

# количество знаков после запятой
prec = 6
x1_r = round(sol[0], prec)
x2_r = round(sol[1], prec)
f1_r = round(Fval[0], prec)
f2_r = round(Fval[1], prec)

df = pd.DataFrame({
    "x1":[x1_r],
    "x2":[x2_r],
    "iters":[iters],
    "F1":[f1_r],
    "F2":[f2_r]
})
print(df)

# строка с подстановкой
print(f"\nПроверка: f1({x1_r}, {x2_r}) = {f1_r}, f2({x1_r}, {x2_r}) = {f2_r}")


# Итерация 80424
# x = [0.29544832 1.90446668]
# Phi(x) = 2.5808692164764667e-25
# grad Phi(x) = [ 2.59095788e-15 -1.10992755e-14]
# Точность достигнута: 1e-60
# Решение: [0.29544832 1.90446668]
# F(x) = [-1.54987134e-13 -7.01536051e-13]

# --- Проверка решения ---
# x* = [0.29544832 1.90446668]
# F(x*) = [-1.54987134e-13 -7.01536051e-13]
# ||F(x*)|| = 7.184523945922188e-13
# Phi(x*) = 2.5808692164764667e-25
# Система удовлетворена с требуемой точностью.

# количество знаков после запятой
sol[0] = 0.29544832
sol[1] = 1.90446668
x1_r = round(sol[0], prec)
x2_r = round(sol[1], prec)
f1_r = round(Fval[0], prec)
f2_r = round(Fval[1], prec)

df = pd.DataFrame({
    "x1":[x1_r],
    "x2":[x2_r],
    "iters":[iters],
    "F1":[f1_r],
    "F2":[f2_r]
})
print(df)

# строка с подстановкой
print(f"\nПроверка: f1({x1_r}, {x2_r}) = {f1_r}, f2({x1_r}, {x2_r}) = {f2_r}")
