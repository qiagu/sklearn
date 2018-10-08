# Sklearn
Utilities for using [scikit-learn](http://scikit-learn.org/) with [galaxy](https://github.com/galaxyproject/galaxy).

**jpickle** converts arbitrary scikit-learn model to JSON-compatible python `dict` object, and vice versa. The `dict` object can be easily save to and retrieve from disk by python `json`.  Compared to conventional pickled model, the JSON model provides ***full human readability*** and ***better compatibility*** across platform and python versions. Supports python 2.7 and 3.4+. 

## Usage

Convert a model oject to `dict` (support all models and pipeline)

```
jpickle.dumpc(model)
```
```
{"-cpython-":"3.7.0","-numpy-":"1.15.1","-object-":{"-reduce-":{"--func-":{"-global-":["copyreg","_reconstructor"]},"-args-":{"-tuple-":[{"-global-":["sklearn.ensemble.gradient_boosting","GradientBoostingRegressor"]},{"-global-":["builtins","object"]},null]},"-state-":{"-keys-":["_sklearn_version","alpha","criterion","estimators_","init","init_","learning_rate","loss","loss_","max_depth","max_features","max_features_","max_leaf_nodes","min_impurity_decrease","min_impurity_split","min_samples_leaf","min_samples_split","min_weight_fraction_leaf","n_classes_","n_estimators","n_features_","presort","random_state","subsample","train_score_","verbose","warm_start"],"_sklearn_version":"0.19.2","alpha":0.9,"criterion":"friedman_mse","estimators_":{"-np_ndarray-":{"-dtype-":{"-reduce-":{"--func-":{"-global-":["numpy","dtype"]},"-args-":{"-tuple-":["O8",0,1]},"-state-":{"-tuple-":[3,"|",null,null,null,-1,-1,63]}}},"-values-":[[{"-reduce-":{"--func-":{"-memo-":0},"-args-":{"-tuple-":[{"-global-":["sklearn.tree.tree","DecisionTreeRegressor"]},...
```

Build a model from a `dict` generated from jpickle

```
jpickle.loadc(data)
```
```
GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, presort='auto', random_state=42,
             subsample=1.0, verbose=0, warm_start=False)
```

## Test
```
git clone https://github.com/qiagu/sklearn.git
cd sklearn
python jpickle.py
```
More tests
```
python jpickle.py ./test-data/rfr_model01
python jpickle.py ./test-data/gbr_model01
```
Check test results
```
ls -l ./test-data/
```
