from collections import defaultdict

_distiller_dict = defaultdict()


def register_distiller(distiller):
    distiller_name = distiller.__name__.lower()
    _distiller_dict[distiller_name] = distiller
    return distiller


def get_distiller(distiller_name):
    return _distiller_dict[distiller_name]
