# FEM-MCT
Finite-Element Method solver combined with the mode-coupling theory of the glass transition, to predict flow properties of viscoelastic shear-thinning / yield-stress fluids using a microscopically derived constitutive equation.

For application of the code, see [Steinhäuser, Treskatis, Turek, and Voigtmann, arXiv:2307.12764 (2023)](https://arxiv.org/abs/2307.12764).

For work-in-progress documentation, see https://femmct.readthedocs.io/

## Installation

Clone the repository and create the Conda environment:

```bash
conda env create -f environment.yml
conda activate femmct-fenicsx
```

Then install the package locally:

```bash
pip install -e .
```
