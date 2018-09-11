# Sklearn
Utilities for using [scikit-learn](http://scikit-learn.org/) with [galaxy](https://github.com/galaxyproject/galaxy).

## Usage
Convert a scikit-learn model to JSON-compatible dict (support all models and pipeline)

```
jsonpickler.dump(model)
```

Build a scikit-learn model from a dict generated by jsonpickler

```
jsonpickler.load(data)
```
