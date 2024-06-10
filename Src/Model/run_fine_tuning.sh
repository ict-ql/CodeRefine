python fine_tune.py \
  -m="codellama/CodeLlama-7b-Instruct-hf" \
  -token-num=2500 \
  -lr=1e-4 \
  -dataset-dir=$the/path/you/store/dataset$ \
  -epochs=10 \
  -output-dir=$where/you/want/store/the/adapters$ 