Подробности можно найти на странице документации https://rectools.readthedocs.io/en/stable/examples/tutorials/transformers_tutorial.html 

_SASRec_ -- это трансформерная модель на последовательностях с _однонаправелнным механизмом внимания_ и "Shifted Sequence" целевой функцией. Итоговое лотентное представление последовательности взаимодействий пользователя используется для предсказания всех айтемов в последовательности взаимодействий на каждой позиции, где каждый прогноз айтема основывается исключительно на предшествующих айтемах.

_BERT4Rec_ -- это трансформерная модель на последовательностях с _двунаправленным механизмом внимания_ и "Item Masked" (он же MLM) целевой функцией. Итоговое латентное представление последовательности взаимодействия пользователя используется для предсказания замаскированных айтемов.

Особенности реализации моделей в библиотеке RecTools:
- в модели BERT4Rec мера близости латентного представления сессии и эмбединга айтема реализовано как скалярное произведение, а не как полносвязный слой. И еще используется PyTorch-реализация многоголового внимания (Multi-Head Attention),
- используется PyTorch-реализация многоголового внимания (Multi-Head Attention).

В RecTools поддерживаются различные функции потерь: BCE, softmax и gBCE с различным количеством негативных айтемов.

Как использовать модель:
- задать максимальную длину истории взаимодействия пользователя с айтемами `session_max_len`,
- задать целевую функцию `loss` (softmax, BCE, gBCE),
- задать размерность латентного пространства `n_factors`,
- задать число трансформерных блоков `n_blocks`,
- задать число голов внимания `n_heads`,
- задать `dropout_rate`,
- задать темп обучения `lr`,
- задать размер пакета `batch_size`,
- задать число эпох обучения `epochs`,
- задать флаг `deterministic=True` для детерминированного обучения модели,
- задать `verbose`.

NB! Для модели BERT4Rec нужно еще задать долю замаскированных айтемов `mask_prob`
```python
sasrec = SASRecModel(
  session_max_len=20,
  loss="softmax",
  n_factors=64,
  n_blocks=1,
  n_heads=4,
  dropout_rate=0.2,
  lr=0.001,
  batch_size=128,
  epochs=1,
  verbose=1,
  deterministic=True,
)

bert4rec = BERT4RecModel(
  mask_prob=0.15,
  deterministic=True,
)

sasrec.fit(dataset)

recos = sasrec.recommend(
  users=[test_user],
  dataset=dataset,
  k=3,
  filter_viewed=True,
  on_unsupported_targets="warn",
)
recos.merge(items[["item_id", "title_orig"]], on="item_id").sort_values(["user_id", "rank"])
```

Для каждой пары "признак-значение признака" создается эмбединг категориальной фичи. Вещественные принаки не поддерживаются.
```python
sasrec_ids_only = SASRecModel(
  deterministic=True,
  loss="softmax",
  item_net_block_types=(IdEmbeddingsItemNet,)
)

sasrec_ids_and_categories = SASRecModel(
  deterministic=True,
  loss="softmax",
  item_net_block_types=(IdEmbeddingsItemNet, CatFeaturesItemId)
)

sasrec_categories_only = SASRecModel(
  deterministic=True,
  loss="softmax",
  item_net_block_types=(CatFeaturesItemNet,)
)
```

RecTools поддерживает 3 функции потерь:
- softmax: не требует дополнительных параметров; вычисляется на полном каталоге айтемов (используется по умолчанию).
- BCE: пользователь может задать число негативных айтемов `n_negatives`,
- gBCE: пользовать может задать число негативных айтемов `n_negatives` и параметр калибровки `gbce_t`.

Чтобы добавить айтемные фичи в процесс обучения айтемных эмбедингов, необходимо передать эти фичи в RecTools набор данных
```python
item_features.head(5)
# id | value | features
# 10711 | drama | genre
# 10711 | foreign | genre
# ...

# Construct dataset
dataset = Dataset.construct(
  interactions_df=interactions,
  item_features_df=item_features,
  cat_item_features=["genre", "director"],
)
```
а также передать объекты `IdEmbeddingsItemNet` и/или `CatFeaturesItemNet` формальному параметру `item_net_block_types`, например
```python
sasrec_ids_and_categories = SASRecModel(
  deterministic=True,
  loss="softmax",
  item_net_block_types=(IdEmbeddingsItemNet, CatFeaturesItemNet), # NB!
)
```

NB! В обучающем наборе данных должно быть как минимум два айтема и для SASRec, и для BERT4Rec.

В модели SASRec однонаправленное внимание (Uni-directional attention) реализуется с помощью _каузальной маски_, которая не позволяет смотреть в будущее.

В модели BERT4Rec используется _padding маска_, которая маскирует PAD-айтемы, чтобы они не влияли на результат.
### Функции потерь

#### Softmax loss

Softmax loss -- это по сути перекрестная энтропия (Cross Entropy Loss), которая вычисляется на полном каталоге айтемов. Softmax loss ищет распределения вероятностей по всем айтемам и потому дает наиболее точные результаты, но ==для больших каталогов айтемов такие вычисления жутко неэффективны==.

В RecTools реализация использует `torch.nn.CrossEntropyLoss("none")`
$$
L = \{ l_1, l_2, \ldots, l_N\}^T, \quad

l_n = -w_{y_n} \log \dfrac{ \exp{x_{n, y_n}} }{ \sum_{c=1}^C\exp(x_{n, c}) } I \{ y_n \neq \text{ignore index} \}
$$
#### Loss with negative sampling

_Функции потерь с отрицательным семплированием_ необходимы для решения проблемы вычислительной неэффективности, присущей вычислениям на полном каталоге. Для этого на каждый позитивный айтем отбирается  $n$ отрицательных. В RecTools негативы отбираются из обучающего набора случайно и равномерно.
##### BCE loss

Бинарная перекрестная энтропия направлена по повышение эффективности вычислений за счет использования расчетов не на полном каталоге, а на нескольких негативных айтемах. 

$x_n$ -- склееные логиты позитивных и негативных айтемов. $y_n$ -- позитивные айтемы, помеченные как 1 (негативные айтемы помечаются 0).

RecTools использует `torch.nn.BCEWithLogitLoss("none")`
$$
L = \{ l_1, l_2, \ldots, l_N\}^T, \quad 

l_n = -w_{y_n} \, [ \, y_n \log \sigma(x_n) + (1 - y_n) \log(1 - \sigma(x_n)) \, ]
$$
##### gBCE loss

==Модели, обученные с использованием _отрицательного семплирования_ (BCE loss), склонны переоценивать вероятность положительных взаимодействий==. Для смягчения этого эффекта можно использовать gBCE loss.

Модификация логитов применяется только к положительным логитам, а отрицательные логиты остаются неизменными
$$
\begin{align}
\text{transformed positive logits} = \log \big( \dfrac{1}{\sigma^{-\beta} (s^+) - 1} \big),\\

\beta = \alpha \, \big( t (1 - \dfrac{1}{\alpha}) + \dfrac{1}{\alpha} \big), \\

\alpha = \dfrac{1}{\text{number of unique items - 1}},
\end{align}
$$
где $t$ -- калибровочный гиперпараметр.