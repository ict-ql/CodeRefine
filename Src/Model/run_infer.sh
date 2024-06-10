python infer.py \
  -m="codellama/CodeLlama-7b-Instruct-hf" \
  -token-num=2500 \
  -infer-kind="CodeRefine" \
  -test-dataset-dir=$the/path/you/store/test/set$ \
  -adapters-dir=$the/path/the/best/checkpoint/in$ \
  -output-dir=$where/you/want/store/the/result$ 