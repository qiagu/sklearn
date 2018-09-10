import pickle
import types
import numpy
import sklearn
from sklearn import (cluster, decomposition, ensemble, feature_extraction, feature_selection,
                    gaussian_process, kernel_approximation, kernel_ridge, linear_model,
                    metrics, model_selection, naive_bayes, neighbors, pipeline, preprocessing,
                    svm, linear_model, tree, discriminant_analysis)

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

        if l == 3:
            state = rv[2]
            retv['_state_'] = save(state)
 
        args = rv[1]
        retv['_args_'] = save(args)

        func = rv[0]
        assert callable(func), "func from reduce is not callable"
        retv['_func_'] = save(func)

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
        return newlist
    dispatch[list] = save_list

    def save_tuple(self, obj):
        newdict = {'_op_': 'tuple'}
        newdict['_items_'] = self.save(list(obj))
        return newdict
    dispatch[tuple] = save_tuple

    def save_set(self, obj):
        newdict = {'_op_': 'set'}
        newdict['_items_'] = self.save(list(obj))
        return newdict
    dispatch[set] = save_set

    def save_dict(self, obj):
        newdict = {}
        for k, v in obj.items():
            newdict[k] = self.save(v)
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

        return newdict
    dispatch[types.FunctionType] = save_global
    dispatch[types.BuiltinFunctionType] = save_global

    def save_numpy_ndarray(self, obj):
        newdict = {'_op_': 'numpy_ndarray'}
        newdict['_dtype_'] = pickle.dumps(obj.dtype)
        newdict['_values_'] = self.save(obj.tolist())
        return newdict
    dispatch[numpy.ndarray] = save_numpy_ndarray

    def save_numpy_datatype(self, obj):
        newdict = {'_op_': 'numpy_datatype'}
        newdict['_datatype_'] = pickle.dumps(obj.dtype)
        newdict['_value_'] = self.save(obj.item())
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

