import sys
import logging
import json
import yaml
import pickle
import types
import numpy
import sklearn
from sklearn import (cluster, decomposition, ensemble, feature_extraction, feature_selection,
                    gaussian_process, kernel_approximation, kernel_ridge, linear_model,
                    metrics, model_selection, naive_bayes, neighbors, pipeline, preprocessing,
                    svm, linear_model, tree, discriminant_analysis)

log = logging.getLogger(__name__)

class ModelToDict:
    """
    Follow the trace of python pickle 
    turn a scikit-learn model to a JSON-compatiable dictionary
    """
    def __init__(self):
        self.memo = {}

    def clear_memo(self):
        """
        clears the 'momo'
        """
        self.memo.clear()
    
    def memoize(self, obj):
        """
        store an object id in the memo
        """
        assert id(obj) not in self.memo
        idx = len(self.memo)
        self.memo[id(obj)] = (idx, obj)
        log.debug("Object saved: %d %s" % (idx, str(obj)))

    def save(self, obj):

        # Check the memo
        x = self.memo.get(id(obj))
        if x:
            rval = {'_op_': 'memo'}
            rval['_idx_'] = x[0]
            return rval

        #check type dispath table
        t = type(obj)
        f = self.dispatch.get(t)
        if f:
            return f(self, obj)

        # Check for a class with a custom metaclass; treat as regular class
        try:
            issc = issubclass(t, type)
        except TypeError:
            issc = 0
        if issc:
            return self.save_global(obj)
        return self.save_reduce(obj)

    def save_reduce(self, obj):

        reduce = getattr(obj, "__reduce__", None)
        if reduce:
            rv = reduce()
        else:
            raise Exception("Can't reduce %r object: %r" %(t.__name__, obj))        
        assert (type(rv) is tuple),\
            "%s must return a tuple, but got %s" % (reduce, type(rv))
        l = len(rv)
        assert (l in [2, 3]),\
            "Reduce tuple is expected to return 2- 3 elements, but got %d elements" % l

        save = self.save

        retv = {'_op_': 'reduce'}

        func = rv[0]
        assert callable(func), "func from reduce is not callable"
        retv['_func_'] = save(func)

        args = rv[1]
        retv['_args_'] = save(args)

        if l == 3:
            state = rv[2]
            retv['_state_'] = save(state)

        self.memoize(obj)
        return retv
    
    dispatch = {}

    def save_primitive(self, obj):
        return obj
    
    dispatch[type(None)] = save_primitive
    dispatch[bool] = save_primitive
    dispatch[int] = save_primitive
    dispatch[long] = save_primitive
    dispatch[float] = save_primitive
    dispatch[complex] = save_primitive
    dispatch[str] = save_primitive
    dispatch[unicode] = save_primitive
    #dispatch[bytearray] = save_primitive

    def save_list(self, obj):
        newlist = []
        for e in obj:
            newlist.append(self.save(e))
        #self.memoize(obj)
        return newlist

    dispatch[list] = save_list

    def save_tuple(self, obj):
        newdict = {'_op_': 'tuple'}
        newdict['_aslist_'] = self.save(list(obj))
        #self.memoize(obj)
        return newdict

    dispatch[tuple] = save_tuple

    def save_set(self, obj):
        newdict = {'_op_': 'set'}
        newdict['_aslist_'] = self.save(list(obj))
        #self.memoize(obj)
        return newdict

    dispatch[set] = save_set

    def save_dict(self, obj):
        newdict = {}
        _keys_ = obj.keys()
        newdict['_keys_'] = _keys_
        for k in _keys_:
            newdict[k] = self.save(obj[k])
        #self.memoize(obj)
        return newdict

    dispatch[dict] = save_dict

    def save_global(self, obj):
        name = getattr(obj, '__name__', None)
        if name is None:
            raise Exception("Can't get global name for object %r" % obj)
        module_name = getattr(obj, '__module__', None)
        if module_name is None:
            raise Exception("Can't get global module name for object %r" % obj)

        newdict = {'_op_': 'global'}
        newdict['_module_'] = module_name
        newdict['_name_'] = name
        self.memoize(obj)
        return newdict

    dispatch[types.FunctionType] = save_global
    dispatch[types.BuiltinFunctionType] = save_global

    def save_numpy_ndarray(self, obj):
        newdict = {'_op_': 'numpy_ndarray'}
        newdict['_dtype_'] = self.save(obj.dtype)
        newdict['_values_'] = self.save(obj.tolist())
        self.memoize(obj)
        return newdict

    dispatch[numpy.ndarray] = save_numpy_ndarray

    def save_numpy_datatype(self, obj):
        newdict = {'_op_': 'numpy_datatype'}
        newdict['_datatype_'] = self.save( type(obj) )
        newdict['_value_'] = self.save(obj.item())
        self.memoize(obj)
        return newdict

    dispatch[numpy.bool_] = save_numpy_datatype
    dispatch[numpy.int_] = save_numpy_datatype
    dispatch[numpy.intc] = save_numpy_datatype
    dispatch[numpy.intp] = save_numpy_datatype
    dispatch[numpy.int8] = save_numpy_datatype
    dispatch[numpy.int16] = save_numpy_datatype
    dispatch[numpy.int32] = save_numpy_datatype
    dispatch[numpy.int64] = save_numpy_datatype
    dispatch[numpy.uint8] = save_numpy_datatype
    dispatch[numpy.uint16] = save_numpy_datatype
    dispatch[numpy.uint32] = save_numpy_datatype
    dispatch[numpy.uint64] = save_numpy_datatype
    dispatch[numpy.float_] = save_numpy_datatype
    dispatch[numpy.float16] = save_numpy_datatype
    dispatch[numpy.float32] = save_numpy_datatype
    dispatch[numpy.float64] = save_numpy_datatype
    dispatch[numpy.complex_] = save_numpy_datatype
    dispatch[numpy.complex64] = save_numpy_datatype
    dispatch[numpy.complex128] = save_numpy_datatype


class DictToModel:
    """
    rebuild a scikit-learn model from dict data generated by ModelToDict.save
    """

    def __init__(self):
        # store newly-built object
        self.memo = {}

    def memoize(self, obj):
        l = len(self.memo)
        self.memo[l] = obj
        log.debug("Object rebuilt: %d %s" % (l, str(obj)))

    def load(self, data):
        """
        the main method to generate an object from dict data
        """
        dispatch = self.dispatch

        t = type(data)
        if t is dict:
            _op_ = data.get('_op_')
            if _op_:
                f = dispatch.get(_op_)
                if f:
                    return f(self, data)
                else:
                    raise Exception("Dispatch table doesn't contain the _op_: %s" % _op_)
            else:
                return dispatch[dict](self, data)
        f = dispatch.get(t)
        if f:
            return f(self, data)

    dispatch = {}

    def load_memo(self, data):
        _idx_ = data.get('_idx_')
        if _idx_ is not None:
            obj = self.memo.get(_idx_)
            if obj is not None:
                return obj
            else:
                raise Exception("Object was referenced before being built and stored in memo: %s" % str(data))

    dispatch['memo'] = load_memo

    def load_primitive(self, data):
        return data

    dispatch[type(None)] = load_primitive
    dispatch[bool] = load_primitive
    dispatch[int] = load_primitive
    dispatch[long] = load_primitive
    dispatch[float] = load_primitive
    dispatch[complex] = load_primitive
    dispatch[str] = load_primitive

    def load_list(self, data):
        newlist = []
        for e in data:
            newlist.append( self.load(e) )
        #self.memoize(newlist)
        return newlist

    dispatch[list] = load_list

    def load_tuple(self, data):
        _aslist_ = self.load( data['_aslist_'] )
        obj = tuple(_aslist_)
        #self.memoize(obj)
        return obj

    dispatch['tuple'] = load_tuple

    def load_set(self, data):
        _aslist_ = self.load( data['_aslist_'] )
        obj = set(_aslist_)
        #self.memoize(obj)
        return obj

    dispatch['set'] = load_set

    def load_dict(self, data):
        newdict = {}
        _keys_ = data['_keys_']
        for k in _keys_:
            try:
                v = data[k]
            except KeyError:
                #json dumps int key to str
                v = data[str(k)]
            newdict[k] = self.load(v)
        #self.memoize( newdict )
        return newdict

    dispatch[dict] = load_dict

    def find_class(self, module, name):
        __import__(module, level=0)
        mod = sys.modules[module]
        return getattr(mod, name)

    def load_global(self, data):
        module = data['_module_']
        name = data['_name_']
        func = self.find_class(module, name)
        self.memoize(func)
        return func

    dispatch['global'] = load_global

    def load_reduce(self, data):
        """
        build object
        """
        _func_ = data.get('_func_')
        func = self.load( _func_)
        assert callable(func)

        _args_ = data.get('_args_')
        args = self.load( _args_)
        assert (type(args) is tuple), "args for rebuild an object must be tuple: %r" % args

        obj = func(*args)

        _state_ = data.get('_state_')
        if _state_:
            state = self.load( _state_)
            setstate = getattr(obj, "__setstate__", None)
            if setstate:
                setstate(state)
            else:
                for k, v in state.items():
                    setattr(obj, k, v)

        self.memoize(obj)
        return obj

    dispatch['reduce'] = load_reduce

    def load_numpy_ndarray(self, data):
        _dtype_ = self.load( data.get('_dtype_') )
        _values_ = self.load( data.get('_values_') )
        obj = numpy.array(_values_, dtype=_dtype_)
        self.memoize(obj)
        return obj

    dispatch['numpy_ndarray'] = load_numpy_ndarray

    def load_numpy_datatype(self, data):
        _datatype_ = self.load( data['_datatype_'] )
        _value_ = self.load( data['_value_'] )
        obj = _datatype_(_value_)
        self.memoize(obj)
        return obj

    dispatch['numpy_datatype'] = load_numpy_datatype


def dump(obj):
    return  ModelToDict().save(obj)

def load(data):
    return DictToModel().load(data)


if __name__ == "__main__":
    # load a model from pickle
    with open('./test-data/grad_Boos_classifier.pickle', 'rb') as f:
        model = pickle.load(f)

    #convert a model object to dictionary
    model_dict = dump(model)

    #dump model dict to json file
    with open('./test-data/jbc_model.json', 'w') as f:
        json.dump(model_dict, f)

    # use yaml.load instead of json.load to avoid unicode in python 2
    with open('./test-data/jbc_model.json', 'r') as f:
        new_dict = yaml.load(f)

    # re-build the model from dict
    re_model = load(new_dict)