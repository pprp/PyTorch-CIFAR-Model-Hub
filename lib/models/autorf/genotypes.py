from collections import namedtuple

Genotype = namedtuple("Genotype", "normal normal_concat")

# PRIMITIVES = [
#     "none",
#     "skip_connect",
#     "sep_conv_3x3",
#     #'sep_conv_5x5',
#     "dil_conv_3x3",
#     #'dil_conv_5x5',
#     "sep_conv_3x3_spatial",
#     #'sep_conv_5x5_spatial',
#     "dil_conv_3x3_spatial",
#     #'dil_conv_5x5_spatial',
#     "SE", # xx
#     "SE_A_M", # xx 
#     "CBAM",
# ]

Attention = Genotype(
    normal=[
        ("skip_connect", 0),
        ("sep_conv_3x3_spatial", 0),
        ("sep_conv_3x3", 1),
        ("CBAM", 0),
        ("skip_connect", 2),
        ("sep_conv_3x3", 1),
        ("sep_conv_3x3", 0),
        ("skip_connect", 1),
        ("sep_conv_3x3", 3),
        ("dil_conv_3x3", 2),
    ],
    normal_concat=range(1, 5),
)

Attention_Searched = Genotype(
    normal=[
        ("skip_connect", 0),
        ("CBAM", 0),
        ("sep_conv_3x3", 1),
        ("skip_connect", 1),
        ("dil_conv_3x3_spatial", 2),
        ("CBAM", 0),
        ("skip_connect", 1),
        ("CBAM", 0),
        ("dil_conv_3x3", 2),
        ("SE_A_M", 3),
    ],
    normal_concat=range(1, 5),
)

Attention_Searched_2 = Genotype(normal=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_5x5', 0), ('skip_connect', 2), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], normal_concat=range(1, 5))


Attention_Searched_3 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('noise', 1), ('max_pool_3x3', 0), ('noise', 1), ('noise', 2), ('max_pool_3x3', 0), ('noise', 1), ('noise', 3), ('noise', 2)], normal_concat=range(1, 5)) 

RFSTEP3 = Genotype(normal=[('max_pool_3x3', 0), ('max_pool_3x3', 0), ('noise', 1), ('avg_pool_5x5', 0), ('noise', 1), ('noise', 2)], normal_concat=range(0, 4))

DARTS = Attention
