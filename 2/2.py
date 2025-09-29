import numpy as np

# Система F(x) = 0
def F(x):
    x1, x2 = x
    f1 = np.sinh(x1 + 0.2*x2 + np.tan(0.1*x1*x2)) - 0.8
    f2 = np.sinh(0.6*x1 - 0.1*x2 + np.tan(0.2*x1*x2)) - 0.1
    return np.array([f1, f2])

# Функционал Phi = 1/2||F||^2
def Phi(x):
    f = F(x)
    return 0.5*np.dot(f, f)

# Градиент Phi (через численное дифференцирование)
def grad_Phi(x, h=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = h
        g[i] = (Phi(x+dx) - Phi(x-dx))/(2*h)
    return g

def gradient_descent(x0, alpha=0.01, eps=1e-4, max_iter=100000):
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        grad = grad_Phi(x)
        x_new = x - alpha*grad
        if np.linalg.norm(x_new - x) < eps:
            print(f"Итерация {k}")
            print(f"x = {x_new}")
            print(f"Phi(x) = {Phi(x_new)}")
            print(f"grad Phi(x) = {grad_Phi(x_new)}")
            print(f"Точность достигнута: {eps}")
            return x_new
        x = x_new
    print("Максимум итераций")
    return x

# Пример запуска:
x_start = [0.0, 0.0]  # начальное приближение
solution = gradient_descent(x_start, alpha=0.01, eps=1e-60)
print("Решение:", solution)
print("F(x) =", F(solution))

# Проверка решения
Fx = F(solution)
norm_F = np.linalg.norm(Fx)
Phi_val = Phi(solution)

print("\n--- Проверка решения ---")
print(f"x* = {solution}")
print(f"F(x*) = {Fx}")
print(f"||F(x*)|| = {norm_F}")
print(f"Phi(x*) = {Phi_val}")
if norm_F < 1e-3:
    print("Система удовлетворена с требуемой точностью.")
else:
    print("Остаток ещё заметен, можно уменьшить eps или шаг alpha.")
