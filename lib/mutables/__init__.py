from .spaces import *
from .ops import *

'''
                               'spaces': 
['Mutable', 'OperationSpace', 'InputSpace', 'MutableScope', 'ValueSpace']
                                 ||
                                 ||
                                 \/
                                'ops'
['Conv1d', 'Conv2d', 'Conv3d', 'Linear', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d]
                                 ||
                                 ||
                                 \/
                                'layers'
['MBLayer', ...]
'''