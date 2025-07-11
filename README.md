This is the codebase for the AISTATS'25 paper entitled

[Domain Adaptation and Entanglement: An Optimal Transport Perspective](https://openreview.net/forum?id=ZDyi1BeTu7)

# Contents

The files included in the codebase and how to generate the experiment results are explained below:

./run_uda.py

Contains the main scripts to generate the experimental results included in the paper.

To run all scenarios across all models, call the generate_results() routine, otherwise
run run_single_case_table1() or run_single_case_table2() to generate results for each table.
Results are saved to the 'results' folder.
Change the 'device' variable used across experiments depending on your GPU and OS.

./adapt.py

Contains the UDA algorithms implemented as classes.

./shifts.py

Contains the various distribution shift scenarios used to test the UDA algorithms.

./utils.py

Contains a bunch of utility functions used throughout the code.

models/

Contains various deep neural networks implemented in PyTorch.

# Citing the paper

The *bibtex* of the paper is as follows

```
@inproceedings{
koc2025domain,
title={Domain Adaptation and Entanglement: an Optimal Transport Perspective},
author={Okan Koc and Alexander Soen and Chao-Kai Chiang and Masashi Sugiyama},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=ZDyi1BeTu7}
}
```
