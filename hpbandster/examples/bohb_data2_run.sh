#!/usr/bin/env bash
num_iters=1000
for i in {0..100}
do
  echo "Run $i times"
  python example_5_mnist.py --n_iterations $num_iters
  mkdir trial$i
  cp results.json trial$i/
  cp configs.json trial$i/ 
done
