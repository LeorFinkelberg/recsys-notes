### Общие замечания

Есть библиотека с реализацией SANSA https://github.com/glami/sansa

SANSA вместо того, чтобы "в лоб" обращать Граммиан $X^T X$ матрицы взаимодействий "пользователь-айтем", эффективно ищет разреженное приблежение к обращенной $X^T X$ .
### Установка
	
Эта реализация модели SANSA использует библиотеку `scikit-sparse`, которая зависит от SuiteSparse https://github.com/DrTimothyAldenDavis/SuiteSparse, поэтому перед установкой библиотеки нужно поставить `suite-sparse`
```bash
# MacOS X
$ brew install suite-sparse

# Ubuntu
$ sudo apt-get install suite-sparse
```
Затем библиотеку можно установить как обычно с помощью `pip`, `conda`, `uv` etc.
```bash
$ uv add sansa
```
