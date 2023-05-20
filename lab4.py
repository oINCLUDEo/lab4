# 11.	Формируется матрица F следующим образом: скопировать в нее А и если в С сумма чисел
# в нечетных столбцах больше, чем произведение чисел в четных строках, то поменять местами В и С симметрично,
# иначе Е и В поменять местами несимметрично. При этом матрица А не меняется.
# После чего если определитель матрицы А больше суммы диагональных элементов матрицы F,
# то вычисляется выражение:A-1*AT – K * F-1, иначе вычисляется выражение (AТ +G-1-F-1)*K,
# где G-нижняя треугольная матрица, полученная из А.
# Выводятся по мере формирования А, F и все матричные операции последовательно.
import numpy as np
import matplotlib.pyplot as plt

N = int(input("Введите размерность матрицы N: "))
K = int(input("Введите значение K: "))

# генерация матриц
B = np.random.randint(-10, 11, size=(N, N))
C = np.random.randint(-10, 11, size=(N, N))
D = np.random.randint(-10, 11, size=(N, N))
E = np.random.randint(-10, 11, size=(N, N))

# создание матриц
A = np.block([[B, C], [D, E]])
F = np.copy(A)

# меняем местами
if np.sum(C[:, ::2]) > np.prod(C[::2, :], axis=1).sum():
    F[:N, :N] = np.copy(C[::-1, ::-1])
    F[:N, N:] = np.copy(B[::-1, ::-1])
    F[N:, :N] = np.copy(D[:N, :N])
    F[N:, N:] = np.copy(E[:N, :N])
else:
    F[:N, :N] = E[:N, :N]
    F[:N, N:] = C[:N, :N]
    F[N:, :N] = D[:N, :N]
    F[N:, N:] = B[:N, :N]
print("Сумма чисел в нечетных столбцах C:", np.sum(C[:, ::2]))
print("Произведение чисел в четных строках C:", np.prod(C[::2, :], axis=1).sum())
print("Матрица A")
print(A)
print("Матрица F")
print(F)
print('Определитель A')
print(np.linalg.det(A))
print('Сумма диагональных элементов F')
print(np.trace(F))
print('Матрица A транспонированная')
print(A.T)
if np.linalg.det(A) > np.trace(F):
    result = np.linalg.inv(A) @ A.T - K * np.linalg.inv(F)
else:
    G = np.tril(A)
    # проверка чтобы матрицы не были сингулярны
    if np.linalg.det(F) != 0:
        if np.linalg.det(G) != 0:
            result = (A.T + np.linalg.inv(G) - np.linalg.inv(F)) * K
        else:
            result = (A.T + G - np.linalg.inv(F)) * K
    else:
        if np.linalg.det(G) != 0:
            result = (A.T + np.linalg.inv(G) - F) * K
        else:
            result = (A.T + G - F) * K

print("Результат выражения:")
print(result)

labels = ['B', 'C', 'D', 'E']
colors = ['red', 'blue', 'green', 'yellow']
sizes = [np.sum(B), np.sum(C), np.sum(D), np.sum(E)]
sizes = [max(0, size) for size in sizes]
graphic, graphics = plt.subplots(1, 3, figsize=(15, 5))

# тепловая
graphics[0].imshow(F, cmap="cool")
graphics[0].set_title("Матрица F")
for i in range(F.shape[0]):
    for j in range(F.shape[1]):
        text = graphics[0].text(j, i, F[i, j], ha="center", va="center", color="b")

# круговая диаграмма
graphics[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
graphics[1].set_title("Круговая диаграмма F")

# столбчатая
x = [1, 2, 3, 4]
heights = [np.sum(B), np.sum(C), np.sum(D), np.sum(E)]
graphics[2].bar(x, heights, color=colors)
graphics[2].set_xticks(x)
graphics[2].set_xticklabels(labels)
graphics[2].set_title("Сумма чисел в каждой матрице")

plt.tight_layout()
plt.show()
