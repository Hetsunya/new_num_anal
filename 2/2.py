import numpy as np
import pandas as pd

# --- Система F(x) = 0 ---
def F(x):
    x1, x2 = x
    f1 = np.sinh(x1 + 0.2*x2 + np.tan(0.1*x1*x2)) - 0.8
    f2 = np.sinh(0.6*x1 - 0.1*x2 + np.tan(0.2*x1*x2)) - 0.1
    return np.array([f1, f2])

# --- Функционал Phi = 1/2||F||^2 ---
def Phi(x):
    f = F(x)
    return 0.5*np.dot(f, f)

# --- Градиент Phi (численно) ---
def grad_Phi(x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = h
        g[i] = (Phi(x+dx) - Phi(x-dx)) / (2*h)
    return g

# --- Градиентный спуск ---
def gradient_descent(x0, alpha=0.1, eps=1e-6, max_iter=100000):
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        grad = grad_Phi(x)
        x_new = x - alpha*grad
        if np.linalg.norm(x_new - x) < eps:
            return x_new, k+1
        x = x_new
    raise RuntimeError("Метод не сошелся за максимум итераций")

# --- Запуск ---
x_start = [0.0, 0.0]
solution, iterations = gradient_descent(x_start, alpha=0.1, eps=1e-6)
Fx = F(solution)
norm_F = np.linalg.norm(Fx)
Phi_val = Phi(solution)

# --- Вывод ---
print("\n" + "="*50)
print("РЕЗУЛЬТАТ РЕШЕНИЯ СИСТЕМЫ")
print("="*50)
print(f"Количество итераций: {iterations}")
print(f"Найденное решение x* = [{solution[0]:.6f}, {solution[1]:.6f}]")
print(f"Функционал Phi(x*) = {Phi_val:.6e}")
print(f"Градиент grad Phi(x*) = [{grad_Phi(solution)[0]:.6e}, {grad_Phi(solution)[1]:.6e}]")
print(f"Норма остатка ||F(x*)|| = {norm_F:.6e}")
print("-"*50)

# --- Таблица результатов ---
prec = 6
df = pd.DataFrame({
    "x1":[round(solution[0], prec)],
    "x2":[round(solution[1], prec)],
    "F1":[round(Fx[0], prec)],
    "F2":[round(Fx[1], prec)]
})
print("\nТаблица значений в найденной точке:")
print(df.to_string(index=False))

# --- Проверка подстановкой ---
print(f"\nПроверка подстановкой: f1({solution[0]:.6f}, {solution[1]:.6f}) = {Fx[0]:.6e}, "
      f"f2({solution[0]:.6f}, {solution[1]:.6f}) = {Fx[1]:.6e}")
print("="*50 + "\n")
