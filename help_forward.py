
# how to generate forward graph and make it into table format
import json 
from collections import defaultdict
import math
import numpy as np
# ============ 不会有激活值的操作 ============
# 简单的数值操作: x = x + 1
# 张量的形状操作：如 view、reshape、squeeze、unsqueeze 等。
# 统计操作：如求和、求平均值、最大值、最小值等
# 张量复制操作：如 clone() 和 detach()。
# ============ 会有激活值的操作 ============
# 激活函数: sigmoid, tanh, relu, leaky_relu, elu, selu, softplus, softsign
# 网络层： conv, linear, batch_norm, dropout, embedding, rnn, lstm, gru, transformer
# 涉及可训练参数的操作： 有grad的情况下
# 损失函数
# ============ end ============


# 每一个op需要算dma和op

path = "/Users/wangyangzuo/Desktop/公司/sd_forward.json"
# attach default n_model_h_w,max_len,hidden_size,dtype

graph = json.load(open(path,'r'))
first_key = list(graph.keys())[0]
start_depth = 0


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




model_param = {
    "n"     : 1,
    "c"     : 4,
    "h"     : 64,
    "w"     : 64,
    "seq"   : 77,
    "dtype" : "float16",
    "rank"  : 4
}

hw_region = [64*64, 64*64/4, 64*64/16, 64*64/64]
h_w_region = [64, 32, 16, 8]
class layout:
    # 4 dim
    n_model_h_w       = 0
    n_model_hw_model  = 1
    n_hw_model_model  = 2
    n_seq_model_model = 3
    # 3 dim
    n_seq_model       = 4
    n_seq_r           = 11
    nmodel_hw_seq     = 10
    nmodel_seq_model  = 12
    n_hw_r            = 7
    # 3 dim
    n_hw_model        = 5
    n_hw_hw           = 6
    nmodel_hw_hw      = 8
    nmodel_hw_model   = 9
    # 2 dim
    n_model           = 13
    # 1 dim
    n                 = 15
    model             = 14
 
    @staticmethod
    def match_shape(shape: list):
        if len(shape) == 1:
            if shape[0] == model_param['n']:
                return layout.n
            return layout.model
        if len(shape) == 2:
            return layout.n_model
        if len(shape) == 3:
            if shape[0] == model_param['n']:
                if shape[1] == model_param["seq"]:
                    if shape[2] == model_param["rank"]:
                        return layout.n_seq_r
                    else:
                        return layout.n_seq_model
                elif shape[2] == model_param["rank"]:
                    return layout.n_hw_r
                elif shape[1] == shape[2]:
                    return layout.n_hw_hw
                else:
                    return layout.n_hw_model
            # 3 dim not starts with n
            elif shape[1] == model_param["seq"]:
                return layout.nmodel_seq_model
            elif shape[2] == model_param["seq"]:
                return layout.nmodel_hw_seq
            elif shape[1] == shape[2]:
                return layout.nmodel_hw_hw
            else:
                return layout.nmodel_hw_model
        if len(shape) == 4:
            if shape[0] == model_param['n']:
                if shape[1] == model_param["seq"]:
                    return layout.n_seq_model_model
                elif shape[1] in hw_region:
                    return layout.n_hw_model_model
                elif shape[2] in hw_region:
                    return layout.n_model_hw_model
                else:
                    # assert(shape[2] in h_w_region and shape[3] in h_w_region, "shape error")
                    return layout.n_model_h_w
            else:
                # assert(shape[1] in hw_region and shape[2] in hw_region, "shape error")
                print("shape error")
        return 0
    @staticmethod
    def match_layout(node):
        input_shapes = node["input_shape"]
        output_shapes = node["output_shape"]
        node["input_shape_layout"] = [layout.match_shape(i) for i in input_shapes]
        node["output_shape_layout"] = [layout.match_shape(i) for i in output_shapes]
        return node

config_param = {
    # model config
    "n": 1,
    # c is not considered
    "c": 4,
    "h": 64,
    "w": 64,
    "dtype": "float16",
    "max_len": 77,
    "encoder_hidden_size": 768,
    # dma tiu config
    "1684x dma": 6e4,
    "2260 dma": 548e3,
    "1684x f16 tiu": 16e3,
    "1684x f32 tiu": 2e3,
    "2260 f16 tiu": 128e3,
    "2260 f32 tiu": 16e3,
}
node_fn  = {}
ops_info = {}
dma_ops_info = {}
dtype_map = {"float16": 2, "float32": 4, "int32": 4, "const": 2}
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
    # bmm input_shape * output_shape[-1]
    input_shape_0 = node['input_shape'][0]
    output_shape  = node['output_shape'][0]
    return np.prod(input_shape_0) * output_shape[-1]

# layer norm, group norm 
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

# 其他的特定op特殊去算了 : conv,baddbmm 

def basic_dma_op(node):
    node = layout.match_layout(node)
    if dma_ops_info[node["name"]]["norm_op"]:
        return norm_dma_op_calc(node)
    input_len  = len(node['input_shape'])
    output_len = len(node['output_shape'])
    input_dtype = [dtype_map[i] for i in node['input_dtype']]
    output_dtype = [dtype_map[i] for i in node['output_dtype']]
    s2l_dma = sum([ input_dtype[i]*np.prod(node['input_shape'][i]) for i in range(input_len) ])
    l2s_dma = sum([ output_dtype[i]*np.prod(node['output_shape'][i]) for i in range(output_len) ])
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    if dma_ops_info[node["name"]]["binary_op"] or dma_ops_info[node["name"]]["element_wise_op"]:
        node["ops"] = element_wise_op_calc(node)
    elif dma_ops_info[node["name"]]["only_dma"]:
        node["ops"] = 0
        return node
    elif dma_ops_info[node["name"]]["linear_op"]:
        node["ops"] = linear_dma_op_calc(node)
    else:
        print(node["name"], "not support yet")
    return node

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

    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.nmodel_hw_hw, 
                                 layout.nmodel_hw_seq,
                                 layout.nmodel_hw_model]], special_op=True)
def baddbmm_dma_op(node):
    layout.match_layout(node)
    input_len  = len(node['input_shape'])
    output_len = len(node['output_shape'])
    input_dtype = [dtype_map[i] for i in node['input_dtype']]
    s2l_dma = sum([ input_dtype[i]*np.prod(node['input_shape'][i]) for i in range(input_len) ])
    l2s_dma = np.prod(node['output_shape'][0]) * dtype_map[node['output_dtype'][0]]
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    # alpha * batch1 @ batch2 + input * belta
    ops = np.prod(node["input_shape"][1]) * node["input_shape"][2][-1] + np.prod(node["input_shape"][0]) * 2
    node["ops"] = ops
    return node

# 最好的shape 推导方案是全局shape inference，现在作为代替方案

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
    node = layout.match_layout(node)
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
    node["ops"]     = np.prod(output_shape[0]) * kernel_size[0] * kernel_size[1]
    return node

@warp_dma_op_shape_node(layout=[[layout.n_model]],element_wise_op=True, op_rate=5)
def cos_dma_op(node):
    return basic_dma_op(node)

@warp_dma_op_shape_node(layout=[[layout.n_hw_model]], special_op=True)
def dropout_dma_op(node):
    return getitem_dma_op(node)

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
    layout.match_layout(node)
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
    node = layout.match_layout(node)
    info     = node["info"]
    is_bias  = info.split("bias=")[1].split(")")[0] == "True"
    dtype_len = dtype_map[node["input_dtype"][0]]
    input_shape  = node['input_shape'][0]
    output_shape = node['output_shape'][0]
    s2l_dma      = np.prod(input_shape) * dtype_len
    l2s_dma      = np.prod(output_shape) * dtype_len
    ops          = np.prod(input_shape) * output_shape[-1]
    if is_bias:
        s2l_dma += output_shape[-1] * dtype_len
        ops    = np.prod(input_shape) * (output_shape[-1] + 1)
    node["s2l_dma"] = s2l_dma
    node["l2s_dma"] = l2s_dma
    node["ops"]     = ops
    return node

@warp_dma_op_shape_node(layout=[[layout.n_hw_model, layout.n_model, ]], norm_op=True, op_rate=5)
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

@warp_dma_op_shape_node(layout=[layout.n_seq_model, 
                                layout.n_hw_model, 
                                layout.nmodel_hw_hw,
                                layout.nmodel_hw_seq
                                ],only_dma=True)
def to_dma_op(node):
    return basic_dma_op(node)

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
    weight_shape = [input_shape[0][1], output_shape[0][1], kernel_size[0], kernel_size[1]]
    return [ handle_input_shape(input_shape[0]), weight_shape ]

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

res_set = []

def handle_record_node(node, pre):
    # shape paddint to 4 dims
    # record dtype
    # record addtional info
    # chunk outputs has 2 
    node_key                 = list(node.keys())[0]
    pre_key                  = list(pre.keys())[0]
    res_node                 = {}
    depth                    = node[node_key]['depth']
    node[node_key]['info']   = pre[pre_key]["comment"]
    if node[node_key]["name"] == "empty-pass":
        return 
    if node[node_key]["name"] in fix_op:
        node[node_key] = fix_op[node[node_key]["name"]](node[node_key])
    node[node_key]           = ops_info[node[node_key]["name"]](node[node_key])
    res_node["name"]         = node[node_key]["name"]
    res_node['id']           = node_key
    res_node['depth']        = node[node_key]['depth']
    node[node_key]['info']   = pre[pre_key]["comment"][:200] if depth > 1 else ""
    res_node['path']         = node[node_key]['path']
    res_node["info"]         = pre[pre_key]["comment"][:200] if depth > 1 else ""
    res_node['input_shape']  = node_fn[res_node['name']](node)
    res_node['output_shape'] = [ handle_input_shape(node[node_key]['output_shape'][0])]
    res_node['input_dtype']  = node[node_key]['input_dtype'][0]
    res_node['output_dtype'] = node[node_key]['output_dtype'][0]
    res_node["ops"]          = node[node_key]["ops"]
    res_node["s2l_dma"]      = node[node_key]["s2l_dma"]
    res_node["l2s_dma"]      = node[node_key]["l2s_dma"]
    res_node["input_shape_layout"] = node[node_key]["input_shape_layout"]
    res_node["output_shape_layout"] = node[node_key]["output_shape_layout"]
    res_set.append(res_node)
    return res_node

next_nodes = [[graph, None]]
need_mem = defaultdict(dict)
tables = []
while next_nodes:
    cur_node, father_node = next_nodes.pop()
    k = list(cur_node.keys())[0]
    father_k = list(father_node.keys())[0] if father_node else None
    if cur_node[k]['depth'] >= 0:
        if "path" not in father_node[father_k]:
            cur_node[k]["path"] = [ cur_node[k]["name"] ]
        else:
            cur_node[k]["path"] = father_node[father_k]["path"] + [ cur_node[k]["name"] ]
    if not cur_node[k]["children"]:
        handle_record_node(cur_node, father_node)
        continue
    for child in cur_node[k]['children'][::-1]:
        next_nodes.append([child, cur_node])

import pdb;pdb.set_trace()

# export 成 csv 最关键的问题是path的处理 
DEFAULT_DROP_PATH = 1
DEFAULT_MAX_PATH  = 6

f = open("output_csv.csv", "w")

# path 6 level 
header = "path_0,path_1,path_2,path_3,path_4,path_5,left_n,left_c,left_h,left_w,right_n,right_c,right_h,right,output_n,output_c,output_h,output_w,input_dtype,output_dtype,info\n"
f.write(header)
for item in res_set:
    res_str = []
    path = item["path"]
    path = path[DEFAULT_DROP_PATH:][:DEFAULT_MAX_PATH]
    # padding into 6 and last one must have value 
    path = path[:-1] + [""]*(DEFAULT_MAX_PATH-len(path)) + [path[-1]]
    res_str += path + [""]*(DEFAULT_MAX_PATH-len(path))
    res_str += [str(i) for i in item["input_shape"][0]]
    res_str += [str(i) for i in item["input_shape"][1]]
    res_str += [str(i) for i in item["output_shape"][0]]
    res_str += [item["input_dtype"]]
    res_str += [item["output_dtype"]]
    res_str += [item["info"].replace(",", " ").replace("\n", "  ")]
    f.write(",".join(res_str) + "\n")
f.close()
# print(res_set)

# print(json.dumps(res_set, indent=4))

# print( sorted(list(res_set)) )