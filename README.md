<p align="center">
<img src="https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/docs/_static/readme_logo.svg" alt="PyNeuraLogic" title="PyNeuraLogic"/>
</p>

[![PyPI version](https://badge.fury.io/py/neuralogic.svg)](https://badge.fury.io/py/neuralogic)
[![License](https://img.shields.io/pypi/l/neuralogic)](https://badge.fury.io/py/neuralogic)
[![Tests Status](https://github.com/LukasZahradnik/PyNeuraLogic/actions/workflows/tests.yml/badge.svg)](https://github.com/LukasZahradnik/PyNeuraLogic/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/pyneuralogic/badge/?version=latest)](https://pyneuralogic.readthedocs.io/en/latest/?badge=latest)


[Documentation](https://pyneuralogic.readthedocs.io/en/latest/) | [Examples](#Examples) | [Papers](#Papers)

PyNeuraLogic lets you use Python to write **Differentiable Logic Programs**


[comment]: <> (PyNeuraLogic is a framework built on top of [NeuraLogic]&#40;https://github.com/GustikS/NeuraLogic&#41; which combines relational and deep learning.)

---

## About

Logic programming is a declarative coding paradigm in which you declare your logical _variables_ and _relations_ between them. These can be further composed into so-called _rules_ that drive the computation. Such a rule set then forms a _logic program_, and its execution is equivalent to performing logic inference with the rules.

PyNeuralogic, through its [**NeuraLogic**](https://github.com/GustikS/NeuraLogic) backend, then makes this inference process _differentiable_ which, in turn, makes it equivalent to forward propagation in deep learning. This lets you learn numeric parameters that can be associated with the rules, just like you learn weights in neural networks.

### What is this good for?

Many things! For instance - ever heard of [Graph Neural Networks](https://distill.pub/2021/gnn-intro/) (GNNs)? Well, a _graph_ happens to be a special case of a logical relation - a binary one to be more exact. Now, at the heart of any GNN model there is a so-called _propagation rule_ for passing 'messages' between the neighboring nodes. Particularly, the representation ('message') of a node `X` is calculated by aggregating the previous representations of adjacent nodes `Y`, i.e. those with an `edge` between `X` and `Y`.

Or, a bit more 'formally':

```
Relation.message2(Var.X) <= (Relation.message1(Var.Y), Relation.edge(Var.X,Var.Y))
```

...and that's the actual _code_! Now for a classic learnable GNN layer, you'll want to add some weights, such as

```
Relation.message2(Var.X)[5,10] <= (Relation.message1(Var.Y)[10,20], Relation.edge(Var.X,Var.Y))
```

to project your `[1,20]` input node embeddings ('message1') through a learnable ``[10,20]`` layer before the aggregation, and subsequently a `[5,10]` layer after the aggregation.

If you don't like the default settings, you can of course [specify](https://pyneuralogic.readthedocs.io/en/latest/language.html) various additional details, such as the particular aggregation and activation functions

```
R.message2(V.X)[5,10] <= (R.message1(V.Y)[10,20], R.edge(V.X,V.Y)) | [Activation.RELU, Aggregation.AVG]
```

to instantiate the classic GCN layer specification, which you can directly train now!


### How is it different from other GNN frameworks?

Naturally, PyNeuralogic is by no means limited to GNN models, as the expressiveness of _relational_ logic goes much further beyond graphs. Hence, nothing stops you from playing directly with:
- multiple relations and object types
- hypergraphs, nested graphs, relational databases
- relational pattern matching, various subgraph GNNs
- alternative propagation schemes
- inclusion of logical background knowledge
- and more...

In [PyNeuraLogic](https://dspace.cvut.cz/bitstream/handle/10467/97065/F3-DP-2021-Zahradnik-Lukas-Extending-Graph-Neural-Networks-with-Relational-Logic.pdf?sequence=-1&isAllowed=y), all these ideas take the same form of simple small logic programs. These are commonly highly transparent and easy to understand, thanks to their declarative nature. Consequently, there is no need to design a zoo of blackbox class names for each small modification of the GNN rule - you code directly at the level of the logical principles here!

The [backend engine](https://jair.org/index.php/jair/article/view/11203) then creates the underlying differentiable computation (inference) graphs in a fully automated and dynamic fashion, hence you don't have to care about aligning everything into some static (tensor) operations.


### How does it perform?

While PyNeuraLogic allows you to easily declare highly expressive models with capabilities far [beyond the common GNNs](https://arxiv.org/abs/2007.06286), it does not come at the cost of performance for the basic GNNs either. On the contrary, for a range of common GNN models and applications, such as learning with molecules, PyNeuraLogic is actually _considerably_ faster than the popular GNN frameworks, as demonstrated in our [benchmarks](https://pyneuralogic.readthedocs.io/en/latest/benchmarks.html).


<!-- Running a few experiments shows that PyNeuraLogic can outperform popular GNN frameworks in regards to time performance while achieving nearly the same accuracy. Results for a few experiments can be seen in the chart below. -->

<!-- vsechny ty detaily bych nechal az do nejake podrobnejsi dokumentace : Note that PyNeuraLogic requires some time to startup (e.g., translate logic programs into neural networks). In the experiments below, the PyNeuraLogic required to startup, on average, _20.9s (+- 0.85s)_ for GCN, _22.5s (+- 0.49s)_ for GraphSAGE, and _58.48s (+- 1.84s)_ for GIN. The startup is done only once before the training and is compensated for in a few epochs due to PyNeuraLogic higher speed performance.  -->


<p align="center">
<img src="https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/docs/_static/benchmark.svg" alt="Benchmark of PyNeuraLogic" title="Benchmark of PyNeuraLogic"/>
</p>

<!-- i ty popisky grafu bych tady zjednodusil: nadpisy jen "average time per epoch" a "average accuracies" - zbytek plyne z kontextu <sub>
  Benchmarks in the plot above were run on the NCI 786_0 dataset on the CPU using scripts that you can find <a href="https://github.com/LukasZahradnik/PyNeuraLogic/tree/master/benchmarks" target="_blank">here</a>. Models were trained in 100 epochs using Adam optimizer with a learning rate of 1.5e-5.
</sub> -->

</br>

We hope you'll find the framework useful in designing _your own_ deep **relational** learning ideas beyond the GNNs!
Please let us know if you need some guidance or would like to cooperate!


## Getting started


### Prerequisites

To use PyNeuraLogic, you need to install the following prerequisites:

```
Python >= 3.7
Java >= 1.8
```

### Installation

To install PyNeuraLogic's latest release from the PyPI repository, use the following command:

```commandline
$ pip install neuralogic
```

## Examples
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/IntroductionIntoPyNeuraLogic.ipynb) [Simple XOR example](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/IntroductionIntoPyNeuraLogic.ipynb)
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/Mutagenesis.ipynb) [Molecular GNNs](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/Mutagenesis.ipynb)
<br />
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/PatternMatching.ipynb) [Subgraph Patterns](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/PatternMatching.ipynb)
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingKRegularGraphs.ipynb) [Distinguishing k-regular graphs](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingKRegularGraphs.ipynb)
<br />
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingNonRegularGraphs.ipynb) [Distinguishing non-regular graphs](https://github.com/LukasZahradnik/PyNeuraLogic/blob/master/examples/DistinguishingNonRegularGraphs.ipynb)

<br />

## Papers

[Beyond Graph Neural Networks with Lifted Relational Neural Networks](https://arxiv.org/abs/2007.06286) Machine Learning Journal, 2021

[Lifted Relational Neural Networks](https://arxiv.org/abs/1508.05128) Journal of Artificial Intelligence Research, 2018


[Lossless compression of structured convolutional models via lifting](https://arxiv.org/abs/2007.06567) ICLR, 2021
