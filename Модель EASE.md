Реализация модели EASE
$$
\min_B \| \, X - X \, B \, \|_F^2 + \lambda \, \| \, B \, \|_F^2, \quad \text{diag}\, B = 0,
$$
где $X \in \mathbb{R}^{|U| \times |I|}$ -- бинарная матрица взаимодействий, $B \in \mathbb{R}^{|I| \times |I|}$ -- матрица весов, $\lambda$ -- параметр регуляризации на языке Python через разложение Холецкого. Матрица $X^T \, X$ квадратная, симметричная и положительно-определенная, поэтому ее можно переписать в виде $X^T\,X = L\, L^T$ .

Матрица $B$ запишется в виде
$$
B = 
\begin{cases}
0, \ \text{if} \ i = j,\\
- \dfrac{P_{ij}}{P_{jj}}, \ \text{otherwise},
\end{cases}
$$
где $P = (X^T X + \lambda \, I)^{-1}$.

То есть на главной диагонали матрицы стоят нули, чтобы защититься от тривиальных решений, а элементы матрицы вне диагонали определяются как отношение элементов матрицы $P$ к диагональным элементам матрицы $P$ с обратным знаком.

Простейшая реализация
```python
def ease(
	rating_matrix: t.Union[np.array, pd.DataFrame],
	lambda_: float = 250.0
) -> np.array:
    P = np.linalg.inv(rating_matrix.T @ rating_matrix + lambda_ * np.eye(rating_matrix.shape[1]))
    weight_matrix = - P / np.diag(P)
    weight_matrix[np.diag_indices_from(weight_matrix)] = 0

    return weight_matrix
```

```python
X = np.array(...)
G = X.T.dot(X)
diag_indeces = np.diag_indices(G.shape[0])
G[diag_indeces] += l2_lambda
L = np.linalg.cholesky(G)
L_inv = np.linalg.inv(L)
P = L_inv.T.dot(L_inv)
B = - P / np.diag(P)
B[diag_indices] = 0
```
