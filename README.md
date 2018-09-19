# Sklearn
Utilities for using [scikit-learn](http://scikit-learn.org/) with [galaxy](https://github.com/galaxyproject/galaxy).

**skj-pickle** converts arbitary scikit-learn model to JSON-compatiable python `dict` object, and vice versa. The `dict` object can be easily save to and retrive from disk by python `json`.  Compared to conventional pickled model, the JSON model provides ***full human readability*** and ***better compatibility*** across platform and python versions. Supports python 2.7 and 3.4+. 

## Usage

Convert a model oject to `dict` (support all models and pipeline)

```
skj-pickle.dumpc(model)
```

Build a model from a `dict` generated by skj-pickle

```
skj-pickle.loadc(data)
```

## Test
```
git clone https://github.com/qiagu/sklearn.git
cd sklearn
python skj-pickle.py
```
More tests
```
python skj-pickle.py ./test-data/rfr_model01
python skj-pickle.py ./test-data/gbr_model01
```
Check test results
```
ls -l ./test-data/
```
