import sys
from collections import defaultdict
from types import MethodType

_custom_forward_dict = defaultdict(dict)


def register_method(fn):
    module = sys.modules[fn.__module__]
    architecture_class = getattr(module, '_architecture', None)
    assert architecture_class is not None, f'please specify the architecture of the model in {module}'
    architecture_method = fn.__name__
    _custom_forward_dict[architecture_class][architecture_method] = fn
    return fn


def apply_new_method(model):
    method_dict = _custom_forward_dict[model.__class__]
    for method_name in method_dict:
        method = method_dict[method_name]
        setattr(model, method_name, MethodType(method, model))  # todo
