Детали можно найти в статье ALBERT: A Lite BERT for Self-supervised Learning of Language Representations https://arxiv.org/pdf/1909.11942 и в блокноте RecTools https://github.com/MobileTeleSystems/RecTools/blob/main/examples/tutorials/transformers_customization_guide.ipynb

У ALBERT есть две техники уменьшения числа параметров для снижения потребления памяти и увеличения скорости обучения, которые можно использовать вместе или по отдельности:
- Обучение эмбеддингов небольшого размера и последующее линейное проецирование в эмбеддинги нужного размера (факторизованная параметризация эмбедингов)
- Шаринг весов между слоями трансформера.

Пример кастомизации с помощью библиотеки RecTools

```python
# Special ALBERT logic for embeddings - Factorized embedding parametrization

class AlbertSumConstructor(SumOfEmbeddingsConstructor):
    def __init__(
	    self,
	    n_items: int,
	    n_factors: int,
	    item_net_blocks: t.Sequence[ItemNetBase],
    ) -> None:
        super().__init__(n_items=n_items, item_net_blocks=item_net_blocks)
		# Project to actual required hidden space
		self.item_emb_proj = nn.Linear(emb_factors, n_factors)

    @classmethod
    def from_dataset(
	    cls,
	    dataset: Dataset,
	    n_factors: int,
		dropout_rate: float,
        item_net_block_types: t.Sequence[t.Type[ItemNetBase]],
		emb_factors: int, 
    ) -> tpe.Self:
        n_items = dataset.item_id_map.size

        item_net_blocks: t.List[ItemNetBase] = []
		for item_net in item_net_block_types:
		    # Item net blocks will work in lower dimensioanl space
            item_net_block =
                item_net.from_dataset(dataset, emb_factors, dropout_rate)
			if item_net_block is not None:
	            item_net_blocks.append(item_net_block)
				
		return cls(n_items, n_factors, item_net_blocks, emb_factors)

    @classmethod
    def from_dataset_schema(
	    cls,
	    dataset_schema: DatasetSchema,
	    n_factors: int,
	    dropout_rate: float,
	    item_net_block_types: t.Sequence[t.Type[ItemNetBase]],
	    emb_factors: int, # accept new kwarg for lower dimensional space size
    ) -> tpe.Self:
        n_items = dataset_schema.items.n_hot

        item_net_blocks: t.List[ItemNetBase] = []
        for item_net in item_net_block_types:
            item_net_block = item_net.from_dataset_schema(
                dataset_schema, emb_factors, dropout_rate
            )
			if item_net_block is not None:
			    item_net_blocks.append(item_net_block) 

        return cls(n_items, n_factors, item_net_blocks, emb_factors)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
		# create embeddings in lower dimensional space
        item_embs = super().forward(items)  
		# project to actual required hidden space
		item_embs = self.item_emb_proj(item_embs)
```

https://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/modeling_albert.py Реализация `modeling_albert`
```python
# Special ALBERT logic for transformer layers - Cross-layer parameter sharing

class AlbertLayers(TransformerLayersBase):
    def __init__(
	    self,
	    n_blocks: int,
	    n_factors: int,
	    n_heads: int,
	    dropout_rate: float,
	    ff_factors_multiplier: int = 4,
	    n_hidden_groups: int = 1, # accept new kwarg
	    n_inner_groups: int = 1,  # accpet new kwarg
    ) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.n_hidden_groups = n_hidden_groups
        self.n_inner_groups = n_inner_groups
        n_fitted_blocks = int(n_hidden_groups * n_inner_groups)
		self.transformer_blocks = nn.ModuleList(
		    [
			    PreLNTransformerLayer(
				    # number of encoder layer (AlBERTLayers)
					n_factors,
					n_heads,
					dropout_rate,
					ff_factors_multiplier,
			    ) for _ in range(n_fitted_blocks)
		    ]
        )
		self.n_layers_per_group = n_blocks / n_hidden_groups

    def forward(
	    self,
	    seqs: torch.Tensor,
	    timeline_mask: torch.Tensor,
	    attn_mask: t.Optional[torch.Tensor],
		key_padding_mask: t.Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block_idx in range(self.n_blocks):
            group_idx = int(block_idx / self.n_layers_per_group)
			for inner_layer_idx in range(self.n_inner_groups):
			    layer_idx = gropu_idx * self.n_inner_groups + inner_layer_idx
				seqs = self.transformer_blocks[block_idx](
				    seqs, attn_mask, key_padding_mask
				)

        return seqs
```

Теперь нужно передать эти кастомные классы и их ключевые слова
```python
CONSTRUCTOR_KWARGS = {"emb_factors": 64}

TRANSFORMER_LAYERS_KWARGS = {
    "n_hidden_groups": 2,
    "n_inner_groups": 2,
}

albert_model = BERT4RecModel(
    item_net_constructor_type=AlbertSumConstructor,       # type
    item_net_constructor_kwargs=CONSTRUCTOR_KWARGS,       # kwargs
	transformer_layers_type=AlbertLayers,                 # type
	transformer_layers_kwargs=TRANSFORMER_LAYERS_KWARGS,  # kwargs
	get_trainer_func=get_debug_trainer,
)

albert_model.fit(dataset)
```

Можно создать и ALSASRec
```python
alsasrec_model = SASRecModel(
    item_net_constructor_type=AlbertSumConstructor,
    item_net_constructor_kwargs=CONSTRUCTOR_KWARGS,       # kwargs
	transformer_layers_type=AlbertLayers,                 # type
	transformer_layers_kwargs=TRANSFORMER_LAYERS_KWARGS,  # kwargs
	get_trainer_func=get_debug_trainer,
)
```