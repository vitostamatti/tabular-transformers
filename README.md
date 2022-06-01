# Tabular Transformers

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/vitostamatti/tabular-transformers.svg)](https://github.com/vitostamatti/tabular-transformers/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/vitostamatti/tabular-transformers.svg)](https://github.com/vitostamatti/tabular-transformers/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)



## 📝 Table of Contents

- [About](#about)
- [Setup](#setup)
- [Usage](#usage)


## About <a name = "about"></a>

Implementation in ``tensorflow`` of a Transformer Model for tabular data.

## Setup <a name = "setup"></a>

To get started, clone this repo and check that you have all requirements installed.

```
git clone https://github.com/vitostamatti/tabular-transformers.git
pip install -r requirements.txt
``` 

## Usage <a name = "usage"></a>

In the [notebooks](/notebooks/) directory you can find examples of
usage for each object in this repo.

- [Layers](/notebooks/layers.ipynb) ([source code](/src/layers.py))
- [Metrics](/notebooks/metrics.ipynb) ([source code](/src/metrics.py))
- [Models](/notebooks/models.ipynb) ([source code](/src/models.py))


If you want to see a real world use case, 
go to this [example](/notebooks/example.ipynb).

## Roadmap

- [X] First commit of the tabular transformer model.
- [ ] Complete documentation of examples notebooks.
- [ ] Benchmark Tabular Transformer with GBDT models.
- [ ] Create setup.py to make it pip installable.


## Acknowledgement
- https://arxiv.org/abs/2012.06678
- https://arxiv.org/abs/2203.05556

## License

[MIT](LICENSE.TXT)