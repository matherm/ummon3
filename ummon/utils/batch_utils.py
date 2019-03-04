
def label_of(data_obj):
    if type(data_obj) == tuple or type(data_obj) == list:
        X, y = data_obj
        return y
    try:
        from torch_geometric.data import Data
        if isinstance(data_obj, Data):
            return data_obj.y
    except NameError:
        pass
    raise NotImplementedError("Datatype not known. Type was {}.".format(type(data_obj)))

def input_of(data_obj):
    if type(data_obj) == tuple or type(data_obj) == list:
        X, y = data_obj
        return X
    try:
        from torch_geometric.data import Data
        if isinstance(data_obj, Data):
            return data_obj
    except NameError:
        pass
    raise NotImplementedError("Datatype not known. Type was {}.".format(type(data_obj)))