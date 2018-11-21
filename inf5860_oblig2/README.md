# Mandatory assignment 2

INF5860/INF9860 - Machine Learning for Image Analysis\
University of Oslo\
Spring 2018

## Content

Everything you need for this exercise is contained in this folder. A brief description of the
content follows.

### `uio_inf5860_s2018_mandatory2_assignment.ipynb`

Everything related to the assignment. This should be self-contained, and all information is found
in this notebook. You can start the notebook from the command line with

```
$ jupyter notebook uio_inf5860_s2018_mandatory2_assignment.ipynb
```

### `src`

This folder contains the whole program. All functions that you are to implement in this exercise is
found in `src/model.py`, but you are of course free to edit everything you want.

When you have implemented everything, you should be able to test your classifier with

```
$ python src/main.py
```

### Content of supplied code

The exercise contains this notebook, some figures, and a `src` folder:

```
$ tree
.
├── figures
│   ├── cifar10_progress_default.png
│   ├── mnist_progress_default.png
│   └── svhn_progress_default.png
├── README.md
├── src
│   ├── import_data.py
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── run.py
│   └── tests.py
└── uio_inf5860_s2018_mandatory2_assignment.ipynb
```

#### `main.py`

Handles program flow, data input and configurations. You should be able to run this file as an executable: `$ python src/main.py`.

You should not need to change anything here.


#### `import_data.py`

Handles import of the following three datasets

- MNIST
- CIFAR10
- SVHN

You should not need to change anything here.


#### `run.py`

Contains training and evaluation routines.

You should not need to change anything here

#### `model.py`

Implements all the important logic of the classifier.

Everything you need to implement will be located in this file.

#### `tests.py`

In this file, predefined arrays are defined. To be used when checking your implementations.
