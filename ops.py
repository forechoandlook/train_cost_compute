import json 
from collections import defaultdict
import math
import numpy as np

dtype_map = {"float16": 2, "float32": 4, "int32": 4, "const": 2}

fix_op = {}
def warp_fix_fn():
    def _warp_fix_fn(fn):
        name = fn.__name__
        fix_op[name[:-len("_fix")]] = fn
        def warp(*args, **kwargs):
            return fn(*args, **kwargs)
        return warp
    return _warp_fix_fn

@warp_fix_fn()
def cat_fix(node):
    node["input_dtype"] = node["output_dtype"] * 2
    return node

node_fn      = {}
ops_info     = {}
dma_ops_info = {}
def warp_dma_op_shape_node(layout          = None, 
                           out_layout      = None,
                           binary_op       = False, 
                           element_wise_op = False,
                           norm_op         = False,
                           linear_op       = False,
                           only_dma        = False,
                           combine_op      = False,
                           special_op      = False,
                           op_rate         = 1.0):
    def _warp_dma_op_shape_node(fn):
        name      = fn.__name__
        node_name = name[:-len("_dma_op")]
        ops_info[node_name] = fn
        dma_ops_info[node_name]                    = {}
        dma_ops_info[node_name]["layout"]          = layout
        dma_ops_info[node_name]["out_layout"]      = out_layout
        dma_ops_info[node_name]["binary_op"]       = binary_op
        dma_ops_info[node_name]["only_dma"]        = only_dma
        dma_ops_info[node_name]["norm_op"]         = norm_op
        dma_ops_info[node_name]["linear_op"]       = linear_op
        dma_ops_info[node_name]["combine_op"]      = combine_op
        dma_ops_info[node_name]["special_op"]      = special_op
        dma_ops_info[node_name]["op_rate"]         = op_rate
        dma_ops_info[node_name]["element_wise_op"] = element_wise_op
        def warp(*args, **kwargs):
            return fn(*args, **kwargs)
        return warp
    return _warp_dma_op_shape_node

def element_wise_op_calc(node):
    # output_shape * rate only one output shape
    name = node["name"]
    rate = dma_ops_info[name]["op_rate"]
    node["ops"] = rate * np.prod(node['output_shape'][0])
    return node

def linear_dma_op_calc(node):
    # 8 10 20  8 20 30  => 8 10 30
    input_shape_0 = node['input_shape'][0]
    input_shape_1 = node['input_shape'][1]
    return np.prod(input_shape_0) * input_shape_1[2]

def norm_dma_op_calc(node):
    # layer norm + group norm
    # bias
    info    = node["info"]
    op_rate = dma_ops_info[node["name"]]["op_rate"]
    dtype_len = dtype_map[node["input_dtype"][0]]
    affine  = info.split("affine=")[1].split(")")[0] == "True"
    s2l_dma = np.prod(node['input_shape'][0]) * dtype_len
    l2s_dma = np.prod(node['output_shape'][0])* dtype_len
    ops     = np.prod(node['input_shape'][0]) * op_rate
    dim     = node["input_shape"][0][-1]
    if affine:
        s2l_dma += 2*dim*dtype_len
        ops      = np.prod(node['input_shape'][0]) * (op_rate + 2 )
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    node["ops"]     = ops
    return node

def basic_dma_op(node):
    if dma_ops_info[node["name"]]["norm_op"]:
        return norm_dma_op_calc(node)
    input_len    = len(node['input_shape'])
    output_len   = len(node['output_shape'])
    input_dtype  = [dtype_map[i] for i in node['input_dtype']]
    output_dtype = [dtype_map[i] for i in node['output_dtype']]
    s2l_dma = sum([ input_dtype[i]*np.prod(node['input_shape'][i]) for i in range(input_len) ])
    l2s_dma = sum([ output_dtype[i]*np.prod(node['output_shape'][i]) for i in range(output_len) ])
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    if dma_ops_info[node["name"]]["binary_op"] or dma_ops_info[node["name"]]["element_wise_op"]:
        element_wise_op_calc(node)
    elif dma_ops_info[node["name"]]["only_dma"]:
        node["ops"] = 0
        return node
    elif dma_ops_info[node["name"]]["linear_op"]:
        node["ops"] = linear_dma_op_calc(node)
    else:
        print(node["name"], "not support yet")
    return node

class layout:
    # 4 dim
    n_model_h_w       = "n_model_h_w"
    n_h_w_model       = "n_h_w_model"
    n_model_hw_model  = "n_model_hw_model"
    n_hw_model_model  = "n_hw_model_model"
    n_seq_model_model = "n_seq_model_model"
    n_model_model_model = "n_model_model_model"
    n_model_seq_model = "n_model_seq_model"
    # 3 dim
    n_seq_model       = "n_seq_model"
    n_seq_r           = "n_seq_r"
    nmodel_model_hw   = "nmodel_model_hw"
    nmodel_hw_seq     = "nmodel_hw_seq"
    nmodel_model_seq  = "nmodel_model_seq"
    nmodel_seq_model  = "nmodel_seq_model"
    n_hw_r            = "n_hw_r"
    # 3 dim
    n_hw_model        = "n_hw_model"
    n_hw_hw           = "n_hw_hw"
    nmodel_hw_hw      = "nmodel_hw_hw"
    nmodel_hw_model   = "nmodel_hw_model"
    # 2 dim
    n_model           = "n_model"
    # 1 dim
    n                 = "n"
    model             = "model"

# 考虑广播
@warp_dma_op_shape_node(layout=[[layout.n_model_h_w, 
                                layout.n_hw_model, 
                                layout.n_seq_model]],
                        binary_op=True)
def add_dma_op(node):
    
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.nmodel_hw_hw, 
                                 layout.nmodel_hw_seq]], linear_op=True)
def bmm_dma_op(node):
    # import pdb;pdb.set_trace()
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.nmodel_hw_hw, 
                                 layout.nmodel_hw_seq,
                                 layout.nmodel_hw_model]], special_op=True)
def baddbmm_dma_op(node):

    input_len  = len(node['input_shape'])
    output_len = len(node['output_shape'])
    input_dtype = [dtype_map[i] for i in node['input_dtype']]
    s2l_dma = sum([ input_dtype[i]*np.prod(node['input_shape'][i]) for i in range(input_len) ])
    l2s_dma = np.prod(node['output_shape'][0]) * dtype_map[node['output_dtype'][0]]
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    # alpha * batch1 @ batch2 + input * belta
    ops = np.prod(node["input_shape"][1]) * node["input_shape"][2][-1] + np.prod(node["input_shape"][0]) * 1
    node["ops"] = ops
    return node

@warp_dma_op_shape_node(special_op=True)
def fuse_lora_dma_op(node):
    info = node["comment"]
    node["info"] = info
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    # kernel_shape=[320,320], rank=4, weight_dtype=float16, lora_dtype=float32, fuse
    kernel_shape = [int(info.split(",")[0].split("[")[1]), int(info.split(",")[1].split("]")[0])]
    lora_rank    = int(info.split(",")[2].split("=")[1])
    lora_up      = [kernel_shape[0], lora_rank]
    lora_down    = [lora_rank, kernel_shape[1]]
    dtype_len    = dtype_map[node['input_dtype'][0]]
    s2l_dma      = np.prod(input_shape[0]) * dtype_len + np.prod(kernel_shape) * dtype_len + np.prod(lora_up) * dtype_len * 2 + np.prod(lora_down) * dtype_len * 2
    l2s_dma      = np.prod(output_shape[0]) * dtype_len
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    ops          = 0
    ops          += np.prod(input_shape[0]) * kernel_shape[1]
    ops          += np.prod(input_shape[0]) * lora_rank * 8
    ops          += np.prod(lora_rank) * kernel_shape[1] * 8
    ops          += np.prod(output_shape[0]) * 2
    node["ops"]  = ops
    return node

@warp_dma_op_shape_node(special_op=True)
def scaled_dot_attention_dma_op(node):
    # input 3 output 1
    input_len    = len(node['input_shape'])
    output_len   = len(node['output_shape'])
    input_dtype  = [dtype_map[i] for i in node['input_dtype']]
    output_dtype = [dtype_map[i] for i in node['output_dtype']]
    s2l_dma      = sum([ input_dtype[i]*np.prod(node['input_shape'][i]) for i in range(input_len) ])
    l2s_dma      = sum([ output_dtype[i]*np.prod(node['output_shape'][i]) for i in range(output_len) ])
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    # ops 
    shape0 = node['input_shape'][0]
    shape1 = node['input_shape'][1]
    shape2 = node['input_shape'][2]
    temp   = [shape0[0] , shape0[1] , shape1[2]]
    ops = 0
    ops += np.prod(shape0) * shape1[2] # q * k
    ops += np.prod(temp) * 7 # / sqrt(d_k) + softmax + mask
    ops += np.prod(temp) * shape2[1] # v * softmax
    node["ops"] = ops
    return node

@warp_dma_op_shape_node(layout=[[layout.n_model_h_w, 
                                 layout.n_model]], only_dma=True)
def cat_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_hw_model]], only_dma=True)
def chunk_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_model_h_w]], only_dma=True)
def contiguous_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_model_h_w]], special_op=True)
def conv2d_dma_op(node):

    input_shape     = node['input_shape']
    info            = node['info']
    dtype_len       = dtype_map[node['input_dtype'][0]]
    kernel          = info.split("kernel_size=(")[1].split(")")[0].split(", ")
    kernel_size     = [int(i) for i in kernel]
    output_shape    = node['output_shape']
    ic              = input_shape[0][1]
    oc              = output_shape[0][1]
    kernel_shape    = [ ic, oc, kernel_size[0], kernel_size[1] ]
    kernel_reorder  = [ oc, kernel_size[0] * kernel_size[1], math.ceil(ic*dtype_len/64)*64 ]
    s2l_dma         = np.prod(input_shape[0])  * dtype_len + np.prod(kernel_shape)   * dtype_len + np.prod(kernel_shape) * dtype_len
    l2s_dma         = np.prod(output_shape[0]) * dtype_len
    node["reorder_weight_dma"] = np.prod(kernel_reorder) * dtype_len
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    node["ops"]     = np.prod(output_shape[0]) * kernel_size[0] * kernel_size[1] * ic
    return node

@warp_dma_op_shape_node(layout=[[layout.n_model]],element_wise_op=True, op_rate=5)
def cos_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_hw_model]], special_op=True)
def dropout_dma_op(node):
    input_len  = len(node['input_shape'])
    output_len = len(node['output_shape'])
    input_dtype = [dtype_map[i] for i in node['input_dtype']]
    output_dtype = [dtype_map[i] for i in node['output_dtype']]
    s2l_dma = sum([ input_dtype[i]*np.prod(node['input_shape'][i]) for i in range(input_len) ])
    l2s_dma = sum([ output_dtype[i]*np.prod(node['output_shape'][i]) for i in range(output_len) ])
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    node["ops"]     = np.prod(node['input_shape'][0])
    return node

@warp_dma_op_shape_node(layout=[[layout.n_model_h_w, 
                                 layout.model, 
                                 layout.n_hw_model]],element_wise_op=True, op_rate=1)
def div_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.model]], only_dma=True)
def expand_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.model]], element_wise_op=True, op_rate=5)
def exp_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(only_dma=True)
def empty_pass_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.model]], only_dma=True)
def float_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_model_h_w]], norm_op=True, op_rate=5)
def group_norm_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(special_op=True)
def getitem_dma_op(node):

    node["ops"] = 0
    node["s2l_dma"] = 0
    node["l2s_dma"] = 0
    return node

ops_info["__getitem__"] = getitem_dma_op

@warp_dma_op_shape_node(layout=[[layout.n_hw_model]], element_wise_op=True, op_rate=5)
def gelu_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_model_h_w]], element_wise_op=True, op_rate=4)
def interpolate_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_hw_model, layout.n_hw_r]],special_op=True)
def linear_dma_op(node):
    info     = node["info"]
    is_bias  = info.split("bias=")[1].split(")")[0] == "True"
    dtype_len = dtype_map[node["input_dtype"][0]]
    input_shape  = node['input_shape'][0]
    output_shape = node['output_shape'][0]
    in_features  = int(info.split(",")[0].split("=")[1])
    out_features = int(info.split(",")[1].split("=")[1])
    s2l_dma      = np.prod(input_shape) * dtype_len + in_features * out_features * dtype_len
    l2s_dma      = np.prod(output_shape) * dtype_len
    ops          = np.prod(input_shape) * output_shape[-1]
    if is_bias:
        s2l_dma += output_shape[-1] * dtype_len
        ops    = np.prod(input_shape) * (output_shape[-1] + 1)
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    node["ops"]     = ops
    return node

@warp_dma_op_shape_node(layout=[[layout.n_hw_model, layout.n_model ]], norm_op=True, op_rate=5)
def layer_norm_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.model, 
                                 layout.n_seq_model, 
                                 layout.n_hw_model], [layout.model]], element_wise_op=True, op_rate=1)
def mul_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_model_h_w,
                                 layout.n_hw_model_model, 
                                 layout.n_seq_model_model]], only_dma=True)
def permute_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[layout.n_model_h_w, 
                                layout.n_hw_model,
                                layout.nmodel_hw_model, 
                                layout.n_hw_model_model ],special_op=True)
def reshape_dma_op(node):
    return getitem_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_model_h_w]], element_wise_op=True, op_rate=5)
def silu_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_model]], element_wise_op=True, op_rate=5)
def sin_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_hw_hw]], element_wise_op=True, op_rate=5)
def softmax_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(special_op=True)
def to_dma_op(node):
    input_dtype  = node["input_dtype"][0]
    output_dtype = node["output_dtype"][0]
    if input_dtype == output_dtype:
        node["s2l_dma"] = 0
        node["l2s_dma"] = 0
    else:
        node["s2l_dma"] = np.prod(node['input_shape'][0]) * dtype_map[input_dtype]
        node["l2s_dma"] = np.prod(node['output_shape'][0]) * dtype_map[output_dtype]
    node["ops"] = 0
    return node

@warp_dma_op_shape_node(only_dma=True)
def transpose_dma_op(node):
    return basic_dma_op(node)

def warp_node():
    def _warp_node(fn):
        name      = fn.__name__
        node_name = name[:-len("_node")]
        node_fn[node_name] = fn
        def warp(*args, **kwargs):
            return fn(*args, **kwargs)
        return warp
    return _warp_node


def handle_input_shape(shape):
    if len(shape) == 1 and isinstance(shape[0], list) and len(shape[0]) > 0:
        return handle_input_shape(shape[0])
    res = [1,1,1,1]
    for i in range(len(shape)):
        res[-len(shape)+i] = shape[i]
    return res

def basic_node(node):
    # only consider the input shape and dtype
    node_key = list(node.keys())[0]
    input_shape = node[node_key]['input_shape']
    res = []
    if len(input_shape) == 3:
        res.append( handle_input_shape(input_shape[0]) )
        res.append( handle_input_shape(input_shape[1]) )
    elif len(input_shape) == 2:
        res.append( handle_input_shape(input_shape[0]) )
        res.append( handle_input_shape(input_shape[1]) )
    else:
        res.append( handle_input_shape(input_shape[0]) )
        res.append( handle_input_shape([0,0,0,0]) )
    return res

@warp_node()
def add_node(node):
    return basic_node(node)

# b 
@warp_node()
def bmm_node(node):
    return basic_node(node)

@warp_node()
def baddbmm_node(node):
    return basic_node(node)

@warp_node()
def cat_node(node):
    return basic_node(node)

@warp_node()
def chunk_node(node):
    return basic_node(node)

@warp_node()
def contiguous_node(node):
    return basic_node(node)

@warp_node()
def conv2d_node( node ):
    # Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1)
    # LoRACompatibleConv(320, 320, kernel_size=(3, 3), s
    node_key     = list(node.keys())[0]
    input_shape  = node[node_key]['input_shape']
    info         = node[node_key]['info']
    kernel       = info.split("kernel_size=(")[1].split(")")[0].split(", ")
    kernel_size  = [int(i) for i in kernel]
    output_shape = node[node_key]['output_shape']
    weight_shape = [output_shape[0][1], input_shape[0][1], kernel_size[0], kernel_size[1]]
    return [ handle_input_shape(input_shape[0]), weight_shape ]

@warp_node()
def fuse_lora_node(node):
    return basic_node(node)

@warp_node()
def scaled_dot_attention_node(node):
    return basic_node(node)

@warp_node()
def cos_node(node):
    return basic_node(node)

@warp_node()
def dropout_node(node):
    return basic_node(node)

@warp_node()
def div_node(node):
    return basic_node(node)

@warp_node()
def expand_node(node):
    return basic_node(node)

@warp_node()
def exp_node(node):
    return basic_node(node)

@warp_node()
def empty_pass_node(node):
    return basic_node(node)

@warp_node()
def float_node(node):
    return basic_node(node)

@warp_node()
def group_norm_node(node):
    # GroupNorm(32, 320, eps=1e-05, affine=True)
    node_key = list(node.keys())[0]
    info     = node[node_key]['info']
    is_affine = info.split("affine=")[1].split(")")[0] == "True"
    group_num = int(info.split("GroupNorm(")[1].split(",")[0])
    input_shape = handle_input_shape(node[node_key]['input_shape'])
    if is_affine:
        kernel_shape = [group_num, input_shape[1]//group_num, 1, 1]
        return [input_shape, kernel_shape]
    else:
        return [input_shape, [group_num,1,1,1]]

@warp_node()
def getitem_node(node):
    return basic_node(node)

node_fn["__getitem__"] = getitem_node
node_fn["empty-pass"]  = basic_node

@warp_node()
def gelu_node(node):
    return basic_node(node)

@warp_node()
def interpolate_node(node):
    return basic_node(node)

@warp_node()
def linear_node(node):
    # use batch to show have/not bias
    # Linear(in_features=320, out_features=320, bias=True)
    node_key = list(node.keys())[0]
    info     = node[node_key]['info']
    is_bias  = info.split("bias=")[1].split(")")[0] == "True"
    input_shape = handle_input_shape(node[node_key]['input_shape'])
    out_features = int(info.split("out_features=")[1].split(",")[0])
    if is_bias:
        weight_shape = [ 1, input_shape[1], input_shape[2], out_features ]
        return [input_shape, weight_shape]
    else:
        weight_shape = [ 0, input_shape[1], input_shape[2], out_features ]
        return [input_shape, weight_shape]

@warp_node()
def layer_norm_node(node):
    # LayerNorm(320, eps=1e-05, elementwise_affine=True)
    node_key = list(node.keys())[0]
    info     = node[node_key]['info']
    is_affine = info.split("elementwise_affine=")[1].split(")")[0] == "True"
    input_shape = handle_input_shape(node[node_key]['input_shape'])
    if is_affine:
        kernel_shape = [1, input_shape[1], 1, 1]
        return [input_shape, kernel_shape]
    else:
        return [input_shape, [0,0,0,0]]

@warp_node()
def mul_node(node):
    return basic_node(node)

@warp_node()
def permute_node(node):
    return basic_node(node)

@warp_node()
def reshape_node(node):
    return basic_node(node)

@warp_node()
def silu_node(node):
    return basic_node(node)

@warp_node()
def sin_node(node):
    return basic_node(node)

@warp_node()
def softmax_node(node):
    return basic_node(node)

@warp_node()
def to_node(node):
    return basic_node(node)

@warp_node()
def transpose_node(node):
    return basic_node(node)


backward_node_fn   = {}
backward_node_info = {}

def check_no_dma(node):
    
    name = node['name']
    if name in ["empty-pass", "float", "__getitem__", "contiguous", "reshape"]:
        return True
    return False

def calc_conv_weight_reorder_ops(node):
    
    input_shape     = node['input_shape']
    output_shape    = node['output_shape'][0]
    # oc ic h w
    kernel_shape    = node['kernel_shape']
    back_grad_idx = node['back_grad_idx']
    dtype           = node['input_dtype']
    dtype_len       = [ dtype_map[i] for i in dtype[:len(input_shape)] ][0]
    # output shape 和 weight 都可能需要weight reorder  
    # 这个的weight还有一个transpose的过程 反转180度 暂时不算 反向有超大卷积核对于conv
    dma = 0
    dma += (kernel_shape[0] // (64 // dtype_len)) * (64 / dtype_len) * kernel_shape[1] * kernel_shape[2] * kernel_shape[3] * dtype_len * (0 in back_grad_idx)
    dma += (output_shape[0] // (64 // dtype_len)) * (64 / dtype_len) * output_shape[1] * output_shape[2] * output_shape[3] * dtype_len * (1 in back_grad_idx)
    return dma

def calc_dma_backops(node):
    
    name     = node['name']
    if check_no_dma(node):
        node["back_dma"] = 0
        return node
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    input_dtype  = node['input_dtype']
    input_dtype_len = [ dtype_map[i] for i in input_dtype[:len(input_shape)] ]
    output_dtype = node['output_dtype']
    output_dtype_len = [ dtype_map[i] for i in output_dtype[:len(output_shape)] ]
    back_dma = 0
    back_dma += sum( np.prod(shape) * dtype_len for shape, dtype_len in zip(input_shape, input_dtype_len) )
    back_dma += sum( np.prod(shape) * dtype_len for shape, dtype_len in zip(output_shape, output_dtype_len) )
    if name == 'conv2d':
        back_dma += calc_conv_weight_reorder_ops(node)
    node["back_dma"] = back_dma
    return node

def back_warp_node(special_fn = False, 
                   output_num=1,
                   only_dma=False
                   ):
    def _warp_node(fn):
        name      = fn.__name__
        node_name = name[:-len("_node")][len("back_"):]
        backward_node_info[node_name] = {}
        backward_node_fn[node_name] = fn
        backward_node_info[node_name]['special_fn'] = special_fn
        backward_node_info[node_name]['output_num'] = output_num
        backward_node_info[node_name]['only_dma']   = only_dma
        def warp(node):
            calc_dma_backops(node)
            return fn(node)
        return warp
    return _warp_node


def handle_input_shape(shape):
    if len(shape) == 1 and isinstance(shape[0], list) and len(shape[0]) > 0:
        return handle_input_shape(shape[0])
    res = [1,1,1,1]
    for i in range(len(shape)):
        res[-len(shape)+i] = shape[i]
    return res

def basic_node(node):
    # only consider the input shape and dtype
    node_key = list(node.keys())[0]
    input_shape = node[node_key]['input_shape']
    res = []
    if len(input_shape) == 3:
        res.append( handle_input_shape(input_shape[0]) )
        res.append( handle_input_shape(input_shape[1]) )
    elif len(input_shape) == 2:
        res.append( handle_input_shape(input_shape[0]) )
        res.append( handle_input_shape(input_shape[1]) )
    else:
        res.append( handle_input_shape(input_shape[0]) )
        res.append( handle_input_shape([0,0,0,0]) )
    return res

def help_mul_ops_dma(shape1, shape2, left_trans=False, right_trans=False):
    # 1,4096,320 1280,320 right_trans=True
    # print(shape1, shape2, left_trans, right_trans)
    ops = 0
    if right_trans:
        # kn mk
        ops += np.prod(shape1) * shape2[-2]
    else:
        # km
        ops += np.prod(shape1) * shape2[-1]
    return ops

def sum_grad(grad_shape, cur_shape):
    # if cur_shape is const 
    return np.prod(grad_shape) 

# add boardcast
@back_warp_node(output_num=2)
def back_add_node(node):
    # c = a + b
    # grad_a = grad_c
    # grad_b = grad_c if b is not a constant
    # if b is boardcast, grad_b = sum(grad_c)
    # node has output num , output shape , dtype , input shape 
    node_key     = list(node.keys())[0]
    output_shape = node['output_shape'][0]
    input_shape  = node['input_shape']
    back_grad_idx = node['back_grad_idx']
    left         = input_shape[0]
    right        = input_shape[1]
    back_ops     = 0
    back_ops     += sum_grad(output_shape, left)  * (0 in back_grad_idx)
    back_ops     += sum_grad(output_shape, right) * (1 in back_grad_idx)
    # output shape is same as input shape ?
    node["back_ops"] = back_ops
    return node

@back_warp_node(output_num=1)
def back_fuse_lora_node(node):
    input_shape  = node['input_shape'][0]
    output_shape = node['output_shape'][0]
    back_grad_idx = node['back_grad_idx']
    info         = node['info']
    # kernel_shape=[320,320], rank=4, weight_dtype=float16, lora_dtype=float32, fuse
    kernel_shape = [int(info.split(",")[0].split("[")[1]), int(info.split(",")[1].split("]")[0])]
    lora_rank    = int(info.split(",")[2].split("=")[1])
    lora_up      = [kernel_shape[0], lora_rank]
    lora_down    = [lora_rank, kernel_shape[1]]
    dtype_len    = dtype_map[node['input_dtype'][0]]
    dma  = 0
    dma += np.prod(input_shape[0]) * dtype_len * 2
    dma += np.prod(kernel_shape) * dtype_len
    dma += np.prod(lora_up) * dtype_len * 2 * 2   #consider f32
    dma += np.prod(lora_down) * dtype_len * 2 * 2 #consider f32
    node["back_dma"] = dma
    ops = 0
    # input : x
    # y = x @ kernel + const * ( x @ lora_up @ lora_down )
    ops += np.prod(input_shape) * kernel_shape[1] # add
    ops += np.prod(input_shape) * (lora_rank + 1) * 8
    ops += np.prod( [input_shape[0], input_shape[1], lora_rank ] ) * kernel_shape[1] * 8
    ops += np.prod(input_shape)
    node["back_ops"] = ops
    return node

@back_warp_node(output_num=3)
def back_scaled_dot_attention_node(node):
    # c = scaled_dot_attention(q, k, v) = softmax(q @ k.T / sqrt(d_k)) @ v
    # c = temp @ v.T
    # grad_v = temp.T @ grad_c
    # temp = q @ k.T
    # q: [b, m ,n]
    # k: [b, t, n]
    # temp: [b, m, t]
    # grad_k = q.T @ grad_temp
    # grad_q = grad_temp @ k  [b, m t] @ [b, t, n] = [b, m, n]
    # 
    output_shape  = node['output_shape'][0]
    input_shape   = node['input_shape']
    back_grad_idx = node['back_grad_idx']
    q             = input_shape[0]
    k             = input_shape[1]
    v             = input_shape[2]
    ops     = 0
    # print("memory efficient backward pass with flash attention backward")
    # compute qk first 
    temp = [q[0], q[1], k[2]]
    ops = 0
    recompute_ops = 0
    ops += np.prod(q) * k[2]
    # then compute softmax
    ops += np.prod(temp) * 7
    temp_ops = recompute_ops
    # then compute grad v
    ops += help_mul_ops_dma(temp, output_shape, True, False) * (2 in back_grad_idx)
    # then compute grad softmax
    ops += np.prod(temp) * 7 * ((0 in back_grad_idx) | (1 in back_grad_idx))
    # then compute grad q
    ops += help_mul_ops_dma(temp, k, False, True) * (0 in back_grad_idx)
    # then compute grad k
    ops += help_mul_ops_dma(q, temp, True, False) * (1 in back_grad_idx)
    node["back_ops"] = ops
    node["back_recompute"] = temp_ops
    return node

@back_warp_node(special_fn=True, output_num=2)
def back_bmm_node(node):
    # C = A @ B
    # grad_C shape and dtype is same as C
    # grad_A = grad_C @ B.T
    # grad_B = A.T @ grad_C
    # cal dma + ops
    output_shape = node['output_shape'][0]
    input_shape  = node['input_shape']
    back_grad_idx = node['back_grad_idx']
    left         = input_shape[0]
    right        = input_shape[1]
    back_ops     = 0
    back_ops     += help_mul_ops_dma(output_shape, right, False, True) * (0 in back_grad_idx) * (0 in back_grad_idx)
    back_ops     += help_mul_ops_dma(left, output_shape, True, False) * (1 in back_grad_idx) * (1 in back_grad_idx)
    node["back_ops"] = back_ops
    return node

@back_warp_node(special_fn=True, output_num=3)
def back_baddbmm_node(node):
    # D = beta * C + alpha * A @ B : output 顺序: C A B
    # grad_A = grad_D @ B.T
    # grad_B = A.T @ grad_D
    # grad_C = grad_D
    output_shape    = node['output_shape'][0]
    input_shape     = node['input_shape']
    back_grad_idx = node['back_grad_idx']
    C               = input_shape[0]
    A               = input_shape[1]
    B               = input_shape[2]
    back_ops        = 0
    back_ops        += np.prod(C) * (0 in back_grad_idx)
    back_ops        += help_mul_ops_dma(output_shape, B, False, True) * (1 in back_grad_idx)
    back_ops        += help_mul_ops_dma(A, output_shape, True, False) * (2 in back_grad_idx)
    node["back_ops"] = back_ops
    return node

@back_warp_node(only_dma=True, output_num=2)
def back_cat_node(node):
    # c = torch.cat([a,b], dim=1)
    # grad_a = grad_c[:, :a.shape[1]]
    # grad_b = grad_c[:, a.shape[1]:]
    # 是否存在copy
    return back_chunk_node(node)

@back_warp_node(only_dma=True, output_num=1)
def back_chunk_node(node):
    # c = torch.chunk(a, 2, dim=1)
    # grad_a = torch.cat([grad_c[0], grad_c[1]], dim=1)
    input_shape = node['input_shape']
    output_shape = node['output_shape']
    back_ops = 0
    node["back_ops"] = back_ops
    return node

@back_warp_node(only_dma=True, output_num=1)
def back_contiguous_node(node):
    # c = a.contiguous()
    # grad_a = grad_c
    return back_chunk_node(node)

conv_ops = lambda output_shape, kernel_shape: np.prod(output_shape) * np.prod(kernel_shape) / kernel_shape[0]

@back_warp_node(special_fn=True, output_num=3)
def back_conv2d_node( node ):
    # c = F.conv2d(a, weight, bias, kernel_size=3, stride=1, padding=1)
    # grad_a = F.conv_transpose2d(grad_c, weight, kernel_size=3, stride=1, padding=1)
    # grad_weight = F.conv2d(a, grad_c, kernel_size=3, stride=1, padding=1)
    # grad_bias = sum(grad_c)
    # Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1)
    # LoRACompatibleConv(320, 320, kernel_size=(3, 3)
    node_key     = list(node.keys())[0]
    output_shape = node['output_shape'][0]
    input_shape  = node['input_shape']
    back_grad_idx = node['back_grad_idx']
    input_shape  = input_shape[0]
    # fix
    kernel_shape = node['kernel_shape']
    # kernel_shape: oc,ic,h,w
    back_ops     = 0
    back_ops     += conv_ops(input_shape, kernel_shape) * (0 in back_grad_idx)
    back_ops     += conv_ops(kernel_shape, output_shape) * (1 in back_grad_idx)
    back_ops     += np.prod(output_shape) * (2 in back_grad_idx)
    node["back_ops"] = back_ops
    return node

@back_warp_node(special_fn=False, output_num=1)
def back_cos_node(node):
    # c = torch.cos(a)
    # grad_a = grad_c * (-torch.sin(a))
    
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    back_grad_idx = node['back_grad_idx']
    back_ops = 0
    back_ops += np.prod(input_shape) * (0 in back_grad_idx)
    return basic_node(node)

@back_warp_node(special_fn=True, output_num=1)
def back_dropout_node(node):
    # c = F.dropout(a, p=0.5, training=True)
    # grad_a = grad_c (if training) mask mulconst
    node_key     = list(node.keys())[0]
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    ops = 0
    ops += np.prod(input_shape) * 2
    node["back_ops"] = ops
    return node

# boardcast
@back_warp_node(special_fn=True, output_num=2)
def back_div_node(node):
    # c = a / b
    # grad_a = grad_c / b
    # grad_b = -grad_c * a / b^2
    input_shape = node['input_shape'][0]
    output_shape = node['output_shape']
    back_grad_idx = node['back_grad_idx']
    back_ops = 0
    back_ops += sum_grad(output_shape, input_shape[0]) * (0 in back_grad_idx)
    back_ops += sum_grad(output_shape, input_shape[1]) * (1 in back_grad_idx)
    node["back_ops"] = back_ops
    return node

@back_warp_node(special_fn=True, output_num=1)
def back_expand_node(node):
    # c = a.expand(1, 320, 320)
    # grad_a = grad_c.sum(0)
    
    input_shape = node['input_shape']
    output_shape = node['output_shape']
    back_grad_idx = node['back_grad_idx']
    back_ops = 0
    node["back_ops"] = back_ops
    return node

@back_warp_node(special_fn=False, output_num=1)
def back_exp_node(node):
    # c = torch.exp(a)
    # grad_a = grad_c * torch.exp(a)
    
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    back_grad_idx = node['back_grad_idx']
    back_ops = 0
    back_ops += np.prod(input_shape) * (0 in back_grad_idx)
    node["back_ops"] = back_ops
    return node

@back_warp_node(special_fn=False, output_num=1)
def back_empty_pass_node(node):
    
    node["back_ops"] = 0
    return node

@back_warp_node(special_fn=True, output_num=1)
def back_float_node(node):
    # c = a.float()
    # grad_a = grad_c
    
    input_shape = node['input_shape']
    output_shape = node['output_shape']
    ops = 0
    node["back_ops"] = ops
    return node

@back_warp_node(special_fn=True, output_num=3)
def back_group_norm_node(node):
    # :param input, num_groups, weight, bias, eps
    # c = F.group_norm(a, 32, weight, bias, eps=1e-05)
    # grad_a = F.group_norm(grad_c, 32, weight, bias, eps=1e-05)
    # grad_weight = F.group_norm(a, 32, grad_c, None, eps=1e-05)
    # grad_bias = sum(grad_c)
    # GroupNorm(32, 320, eps=1e-05, affine=True)
    
    input_shape  = handle_input_shape(node['input_shape'])
    back_grad_idx = node['back_grad_idx']
    ops = 0
    ops += np.prod(input_shape) * 12 * (0 in back_grad_idx)
    ops += np.prod(input_shape) * (1 in back_grad_idx)
    ops += np.prod(input_shape) * (2 in back_grad_idx)
    node["back_ops"] = ops
    return node

@back_warp_node()
def back_getitem_node(node):
    ops = 0
    
    node["back_ops"] = ops
    return node

backward_node_fn["__getitem__"] = back_getitem_node
backward_node_fn["empty-pass"]  = back_getitem_node

@back_warp_node(special_fn=False, output_num=1)
def back_gelu_node(node):
    # c = F.gelu(a)
    # grad_a = grad_c * (0.5 * (1 + torch.erf(a / math.sqrt(2))))
    
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    ops = 0
    ops += np.prod(input_shape) * 5
    node["back_ops"] = ops
    return node

@back_warp_node(special_fn=False, output_num=1)
def back_interpolate_node(node):
    # c = F.interpolate(a, size=320, mode='bilinear')
    # grad_a = F.interpolate(grad_c, size=a.shape, mode='bilinear')
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    ops = 0
    ops += np.prod(output_shape) * 8
    node["back_ops"] = ops
    return node

@back_warp_node(special_fn=True, output_num=2)
def back_linear_node(node):
    # c = a @ weight.T + bias
    # grad_a = grad_c @ weight
    # grad_weight = grad_c.T @ a
    # grad_bias = sum(grad_c, axis=0)
    # use batch to show have/not bias
    # Linear(in_features=320, out_features=320, bias=True)
    
    input_shape  = node['input_shape'][0]
    output_shape = node['output_shape'][0]
    back_grad_idx = node['back_grad_idx']
    weight_shape = [ input_shape[-1], output_shape[-1] ]
    ops = 0
    ops += help_mul_ops_dma(output_shape, weight_shape, False, True) * (0 in back_grad_idx)
    ops += help_mul_ops_dma(output_shape, input_shape, True, False) * (1 in back_grad_idx)
    ops += np.prod(weight_shape) * (2 in back_grad_idx)
    node["back_ops"] = ops
    return node

@back_warp_node(special_fn=True, output_num=2)
def back_layer_norm_node(node):
    # c = F.layer_norm(a, [320], weight, bias, eps=1e-05)
    # grad_a = F.layer_norm(grad_c, [320], weight, bias, eps=1e-05)
    # grad_weight = F.layer_norm(a, [320], grad_c, None, eps=1e-05)
    # grad_bias = sum(grad_c)
    # LayerNorm(320, eps=1e-05, elementwise_affine=True)
    
    info     = node['info']
    input_shape = handle_input_shape(node['input_shape'])
    ops = 0
    back_grad_idx = node['back_grad_idx']
    ops += np.prod(input_shape) * 12 * (0 in back_grad_idx)
    ops += np.prod(input_shape) * (1 in back_grad_idx)
    ops += np.prod(input_shape) * (2 in back_grad_idx)
    node["back_ops"] = ops
    return node

@back_warp_node(special_fn=True, output_num=2)
def back_mul_node(node):
    # c = a * b
    # grad_a = grad_c * b
    # grad_b = grad_c * a
    input_shape = node['input_shape']
    output_shape = node['output_shape']
    back_grad_idx = node['back_grad_idx']
    back_ops = 0
    back_ops += sum_grad(output_shape, input_shape[0]) * (0 in back_grad_idx)
    if 1 in back_grad_idx:
        back_ops += sum_grad(output_shape, input_shape[1])
    node["back_ops"] = back_ops
    return node

@back_warp_node(only_dma=True, output_num=1)
def back_permute_node(node):
    # c = a.permute(1, 0, 2)
    # grad_a = grad_c.permute(1, 0, 2)
    node["back_ops"] = 0
    return node

@back_warp_node(only_dma=True, output_num=1)
def back_reshape_node(node):
    # c = a.reshape(1, 320, 320)
    # grad_a = grad_c.reshape(a.shape)
    
    node["back_ops"] = 0
    return node

@back_warp_node(output_num=1)
def back_silu_node(node):
    # c = F.silu(a)
    # grad_a = grad_c * (1 + torch.sigmoid(a) * (1 - torch.sigmoid(a)))
    
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    ops = 0
    ops += np.prod(input_shape) * 6
    node["back_ops"] = ops
    return node

@back_warp_node(output_num=1)
def back_sin_node(node):
    # c = torch.sin(a)
    # grad_a = grad_c * torch.cos(a)
    
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    back_grad_idx = node['back_grad_idx']
    back_ops = 0
    back_ops += np.prod(input_shape) * 6 * (0 in back_grad_idx)
    node["back_ops"] = back_ops
    return node

@back_warp_node(output_num=1)
def back_softmax_node(node):
    # c = F.softmax(a, dim=1)
    # grad_a = grad_c * c * (1 - c)
    
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    back_ops = 0
    back_ops += np.prod(input_shape) * 5
    node["back_ops"] = back_ops
    return node

@back_warp_node(output_num=1, only_dma=True)
def back_to_node(node):
    # c = a.to(dtype)
    input_dtype  = node['input_dtype'][0]
    output_dtype = node['output_dtype'][0]
    back_ops = 0
    node["back_ops"] = back_ops
    node["back_dma"] = 0 if input_dtype == output_dtype else node["back_dma"]
    return node

@back_warp_node(output_num=1, only_dma=True)
def back_transpose_node(node):
    input_shape  = node['input_shape']
    output_shape = node['output_shape']
    back_ops = 0
    node["back_ops"] = back_ops
    return node
 