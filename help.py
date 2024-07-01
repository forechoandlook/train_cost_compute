
import json
from collections import defaultdict

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

NO_BORAD_CAST = 1


graph = json.load(open("/Users/wangyangzuo/Desktop/公司/graph.json"))
gh_link = graph['graph'] 
gh_node = graph['nodes_attr']

map_tables = {}
ops_infos = {}

def warp_node(in_num=1, 
              out_num=1, 
              has_shape=True,
              same_shape=False,
              output_overhead=False,
              **kwargs):
    def _warp_node(fn):
        global map_tables
        name = str(fn.__name__).split("_")[0]
        ops_infos[name] = {"in_num":in_num, 
                           "out_num":out_num, 
                           "has_shape":has_shape,
                           "same_shape":same_shape,
                           "output_overhead":output_overhead,}
        map_tables[name] = fn
        def warp(*args, **kwargs):
            res = fn(*args, **kwargs)
            return res
        return warp
    return _warp_node

dtype_map = {"f32":"f32", "f16":"f16", "16":"f16"}

def parse_shape_dtype(shape_str):
    # (1, 4, 64, 64) f32
    shape_str = shape_str.replace(" ", "")
    shape, dtype = shape_str.split(")")
    shape = [int(i) for i in shape[1:].split(",") if i]
    dtype = dtype_map[dtype] if dtype else 0
    return [shape, dtype]

def parse_str(s):
    sl = s.split("\n")
    sr = {i.split(":")[0]: i.split(":")[1] for i in sl}
    return sr


def is_end_node(node):
    if node["name"] in ["AccumulateGrad"] or node['name'].startswith("empty"):
        return True
    return False

def input_shape_is_known(node):
    info = ops_infos[node["name"]]
    if info["has_shape"]:
        return True
    return False

def get_idx_of_output(cur_id, pre_shape_node):
    info   = pre_shape_node["info"]
    info   = info.replace(" ", "")
    sr     = parse_str(info)
    pre_id = int(sr["id"])
    outputs = gh_link[str(pre_id)]
    return outputs.index(cur_id)

def overhead_node_input_shape(cur_id, idx=0):
    # 需要查找next节点里面 有没有节点是已经知道的shape 
    # 如果遇到 empty 节点就退出 shape为空
    next_ops = [gh_link[cur_id][idx]]
    while next_ops:
        cur = next_ops.pop(0)
        cur_item = gh_node[str(cur)]
        if is_end_node(cur_item):
            continue
        if input_shape_is_known(cur_item):
            name = cur_item["name"]
            info = map_tables[name](cur_item['attr'])
            # print("info: ", info)
            return info["inputs"][0]
        next_ops.extend(gh_link[str(cur)])
    return []

@warp_node(1, 1, False, False)
def AccumulateGrad_node(item, pre=None):
    # {'name': 'AccumulateGrad', 'attr': ''}
    item = item.replace(" ", "")
    sr = parse_str(item)
    node = {}
    node["name"]    = "AccumulateGrad"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    node["outputs"] = node["inputs"]
    node['info']    = item
    return node

@warp_node(1, 2, False, False, True)
def AddBackward0_node(item, pre=None):
    # y = a + b
    # grad_a = grad_y
    # grad_b = grad_y
    # if boradcast: grad_b = sum(grad_y) 广播通过sum实现
    # {'name': 'AddBackward0', 'attr': 'alpha.c: 1\nid     : 136679675902176'}
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "AddBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    # how to over head the output shape
    cur_id          = sr["id"]
    output_num      = ops_infos["AddBackward0"]["out_num"]
    if NO_BORAD_CAST:
        node["outputs"] = [node["inputs"][0], node["inputs"][0]]
    else:
        node["outputs"] = []
        for i in range(output_num):
            node["outputs"].append( overhead_node_input_shape(cur_id, i) )
    node['info']    = item
    return node

@warp_node(1, 3, True, False, False)
def AddmmBackward0_node(item, pre=None):
    # C = beta * M + alpha * A @ B
    # M、A 和 B 的梯度
    # {'name': 'AddmmBackward0', 'attr': 'alpha.c           :               1\nbeta.c            :               1\nmat1.c            :            None\nmat1_sym_sizes.c  :    (4096, 1280)\nmat1_sym_strides.c:       (1280, 1)\nmat2.t            : (1280, 320) f16\nmat2_sym_sizes.c  :     (1280, 320)\nmat2_sym_strides.c:              ()\nid                : 136679675903328'}
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "AddmmBackward0"
    mat1_shape      = parse_shape_dtype(sr["mat1_sym_sizes.c"])[0]
    mat2            = parse_shape_dtype(sr["mat2.t"])
    mat2_shape, dtype = mat2
    input_shape       = [[mat1_shape[0], mat2_shape[1]], dtype]
    node["inputs"]    = [input_shape]
    output            = [[mat1_shape[0], mat2_shape[0]], dtype]
    node["outputs"]   = [
        output,
        [mat1_shape, dtype],
        [mat2_shape, dtype],
    ]
    node['info']      = item
    return node

@warp_node(1, 3, True, False, False)
def BaddbmmBackward0_node(item, pre=None):
    # C = A @ B + C
    # batch1 = A, batch2 = B
    # output: grad_A, grad_B, grad_C
    # {'name': 'BaddbmmBackward0', 'attr': 'alpha.c : 0.15811388300841897\nbatch1.t:   (8, 4096, 40) f16\nbatch2.t:     (8, 40, 77) f16\nbeta.c  :                   0\nid      :     136679676266096'}
    node = {}
    node["name"]    = "BaddbmmBackward0"
    sr              = parse_str(item)
    batch1          = parse_shape_dtype(sr["batch1.t"])
    batch1_shape, dtype = batch1
    batch2          = parse_shape_dtype(sr["batch2.t"])
    batch2_shape, dtype = batch2
    input_shape     = [ batch1_shape[0], batch1_shape[1], batch2_shape[2] ]
    node["inputs"]  = [[input_shape, dtype]]
    node["outputs"] = [
        batch1,
        batch2,
        [input_shape, dtype],
    ]
    node['info']    = item
    return node

@warp_node(1, 2, True, False, False)
def BmmBackward0_node(item, pre=None):
    # C = A @ B
    # A B 的梯度
    # mat2 = B, self = A    
    # {'name': 'BmmBackward0', 'attr': 'mat2.t:   (8, 77, 40) f16\nself.t: (8, 4096, 77) f16\nid    :   136679676265760'}
    node = {}
    node["name"]    = "BmmBackward0"
    sr              = parse_str(item)
    mat2            = parse_shape_dtype(sr["mat2.t"])
    mat2_shape, dtype = mat2
    self_t          = parse_shape_dtype(sr["self.t"])
    self_shape, dtype = self_t
    input_shape     = [[self_shape[0], self_shape[1], mat2_shape[2]], dtype]
    node["inputs"]  = [input_shape]
    node["outputs"] = [
        [[self_shape[0], self_shape[1], mat2_shape[2]], dtype],
        [mat2_shape, dtype],
    ]
    node['info']    = item
    return node

# TODO 
# 不知道输入shape以及输入顺序（缺少前向图信息）
@warp_node(1, 2, False, False, True)
def CatBackward0_node(item, pre=None):
    # {'name': 'CatBackward0', 'attr': 'dim.c: 1\nid  : 136679675902672'}
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "CatBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    cur_id          = sr["id"]
    output_num      = ops_infos["CatBackward0"]["out_num"]
    node["outputs"] = [ ]
    for i in range(output_num):
        node["outputs"].append( overhead_node_input_shape(cur_id, i) )
    node['info']    = item
    return node

@warp_node(1, 3, True, False, False)
def ConvolutionBackward0_node(item, pre=None):
    # bias_sym_sizes_opt.c: (4,)\ndilation.c: (1, 1)\ngroups.c:1\ninput.t : (1, 320, 64, 64) f16\noutput_padding.c: (0, 0)\npadding.c : (1, 1)\nstride.c: (1, 1)\ntransposed.c:False\nweight.t: (4, 320, 3, 3) f16
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "ConvolutionBackward0"
    input_s         = [parse_shape_dtype(sr["input.t"])]
    input_shape     = [i for i in input_s[0][0]]
    weight          = parse_shape_dtype(sr["weight.t"])
    bias_shape      = parse_shape_dtype(sr["bias_sym_sizes_opt.c"])
    bias_shape[1]   = weight[1]
    weight_shape    = weight[0]
    stride          = parse_shape_dtype(sr["stride.c"])[0]
    print("conv stride", stride)
    input_shape[1]  = weight_shape[0]
    if stride[0] == 2:
        input_shape    = [input_shape[0], weight_shape[0], input_shape[2]//2, input_shape[3]//2]
    else:
        input_shape    = [input_shape[0], weight_shape[0], input_shape[2], input_shape[3]]
    node["inputs"]  = [ [ input_shape, input_s[0][1] ]]
    output_shape  = input_s[0][0]
    node["outputs"] = [[output_shape, weight[1]], weight, bias_shape ]
    node['info']    = item
    return node

@warp_node(1, 1, False, False, False)
def CloneBackward0_node(item, pre=None):
    # {'name': 'CloneBackward0', 'attr': ''}
    item = item.replace(" ", "")
    sr   = parse_str(item)
    node = {}
    node["name"]    = "CloneBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    node["outputs"] = node["inputs"]
    node['info']    = item
    return node

@warp_node(1, 2, False, False, True)
def DivBackward0_node(item, pre=None):
    # {'name': 'DivBackward0', 'attr': 'other.t: (1, 4096, 1280) f16\nself.t : (1, 4096, 1280) f16\nid     :     136679675903520'}
    # {'name': 'DivBackward0', 'attr': 'other.t:  () \nself.c : None\nid     : 136679676264944'}
    item = item.replace(" ", "")
    if "136679675011584" in item:
        import pdb;pdb.set_trace()
    node = {}
    sr              = parse_str(item)
    node["name"]    = "DivBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    if not node["inputs"]:
        import pdb;pdb.set_trace()
    cur_id          = sr["id"]
    output_num      = ops_infos["DivBackward0"]["out_num"]
    if NO_BORAD_CAST:
        node["outputs"] = [node["inputs"][0], node["inputs"][0]]
    else:
        node["outputs"] = []
        for i in range(output_num):
            node["outputs"].append( overhead_node_input_shape(cur_id, i) )
    node['info']    = item
    return node

@warp_node(1, 1, True, True, False)
def GeluBackward0_node(item, pre=None):
    # {'name': 'GeluBackward0', 'attr': 'approximate.c:                none\nself.t       : (1, 4096, 1280) f16\nid           :     136679673621776'}
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "GeluBackward0"
    node["inputs"]  = [parse_shape_dtype(sr["self.t"])]
    node["outputs"] = node["inputs"]
    node['info']    = item
    return node

@warp_node(1, 2, True, False)
def MmBackward0_node(item, pre=None):
    # C = A @ B 
    # grad_A, grad_B
    # self = A, mat2 = B
    # grad_A = grad_output @ B.T
    # grad_B = A.T @ grad_output
    # mat2.t:(320,320)f16\n mat2_sym_sizes.c:(320,320)\n mat2_sym_strides.c:()\n self.c:None\n self_sym_sizes.c:(4096,320)\n self_sym_strides.c:(320,1)\n id:136679676266912
    # mat2.c:None mat2_sym_sizes.c:(768,4) mat2_sym_strides.c:(1,768) self.t:(77,768)f32 self_sym_sizes.c:(77,768) self_sym_strides.c:() id:136679671871232
    # 总的来说 输入shape是可以推到出来的
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "MmBackward0"
    if "mat2.t" not in sr:
        mat2 = parse_shape_dtype(sr["mat2_sym_sizes.c"])
    else:
        mat2 = parse_shape_dtype(sr["mat2.t"])
    mat2_shape, dtype1 = mat2
    # may not have self.t 
    if "self.t" not in sr:
        self_t = parse_shape_dtype(sr["self_sym_sizes.c"])
        self_t[1]   = dtype1
    else:
        self_t      = parse_shape_dtype(sr["self.t"])
    self_shape, dtype2 = self_t
    dtype = dtype1 if dtype1 else dtype2
    input_shape     = [[self_shape[0], mat2_shape[1]], dtype]
    node["inputs"]  = [[input_shape, dtype]]
    node["outputs"] = [
        [[self_shape[0], mat2_shape[0]], dtype],
        [mat2_shape, dtype],
    ]
    node['info']    = item
    return node

@warp_node()
def MseLossBackward0_node(item, pre=None):
    item = item.replace(" ", "")
    if pre == None:
        pass
    node = {}
    sr              = parse_str(item)
    node["name"]    = "MseLossBackward0"
    node["inputs"]  = [parse_shape_dtype(sr["self.t"])]
    node["outputs"] = [parse_shape_dtype(sr["target.t"])]
    node['info']    = item
    return node

# other.t ? self.c ?
@warp_node(1, 2, False, False, True)
def MulBackward0_node(item, pre=None):
    # {'name': 'MulBackward0', 'attr': 'other.t: (1, 4096, 1280) f16\nself.t : (1, 4096, 1280) f16\nid     :     136679675903520'}
    # {'name': 'MulBackward0', 'attr': 'other.t:  () \nself.c : None\nid     : 136679676264944'}
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "MulBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    cur_id          = sr["id"]
    output_num      = ops_infos["MulBackward0"]["out_num"]
    if NO_BORAD_CAST:
        node["outputs"] = [node["inputs"][0], node["inputs"][0]]
    else:
        node["outputs"] = []
        for i in range(output_num):
            node["outputs"].append( overhead_node_input_shape(cur_id, i) )
    node['info']    = item
    return node

@warp_node(1, 3, True, True)
def NativeGroupNormBackward0_node(item, pre=None):
    # 'C.c:320\n HxW.c:4096\n N.c:1\n eps.c:1e-05\n group.c:32\n input.t:(1,320,64,64)f16\n result1.t:(1,32)f16\n result2.t:(1,32)f16\n weight.t:(320)f16\n id:136679675902416'
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "NativeGroupNormBackward0"
    node["inputs"]  = [parse_shape_dtype(sr["input.t"])]
    weight_shape    = parse_shape_dtype(sr["weight.t"])
    bias_shape      = weight_shape
    node["outputs"] = [node["inputs"][0], weight_shape, bias_shape]
    node['info']    = item
    return node

@warp_node(1, 3, True, True)
def NativeLayerNormBackward0_node(item, pre=None):
    # y = gamma * (x - mean) / sqrt(var + eps) + beta
    # grad_x = grad_y * gamma / sqrt(var + eps) - gamma * grad_mean / sqrt(var + eps) - gamma * (x - mean) * grad_var / (var + eps) / sqrt(var + eps)
    # grad_gamma = sum(grad_y * (x - mean) / sqrt(var + eps))
    # grad_beta = sum(grad_y)
    # {'name': 'NativeLayerNormBackward0', 'attr': 'bias.t            :          (320) f16\ninput.t           : (1, 4096, 320) f16\nnormalized_shape.c:             (320,)\nresult1.t         :   (1, 4096, 1) f32\nresult2.t         :   (1, 4096, 1) f32\nweight.t          :          (320) f16\nid                :    136679676264608'}
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "NativeLayerNormBackward0"
    node["inputs"]  = [parse_shape_dtype(sr["input.t"])]
    weight_shape    = parse_shape_dtype(sr["weight.t"])
    bias_shape      = parse_shape_dtype(sr["bias.t"])
    node['outputs'] = [node["inputs"][0], weight_shape, bias_shape]
    node['info']    = item
    return node

@warp_node(1, 1, False, False)
def PermuteBackward0_node(item, pre=None):
    # {'name': 'PermuteBackward0', 'attr': 'dims.c: (0, 3, 1, 2)\nid    : 136679675901840'}
    item = item.replace(" ", "")
    node = {}
    node["name"]    = "PermuteBackward0"
    sr              = parse_str(item)
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    permute         = parse_shape_dtype(sr["dims.c"])[0]
    # calculate the permute shape
    shape           = node["inputs"][0][0]
    output_shape    = [shape[i] for i in permute]
    node["outputs"] = [[output_shape, node["inputs"][0][1]]]
    node['info']    = item
    return node

@warp_node(1, 1, False, False, False)
def ReshapeAliasBackward0_node(item, pre=None):
    # {'name': 'ReshapeAliasBackward0', 'attr': 'self_sym_sizes.c: (1, 320, 64, 64)\nid              : 136679675902512'}
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "ReshapeAliasBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    output_shape    = parse_shape_dtype(sr["self_sym_sizes.c"])[0]
    node["outputs"] = [[output_shape, node["inputs"][0][1]]]
    node['info']    = item
    return node

@warp_node(1, 1, True, True)
def SiluBackward0_node(item, pre=None):
    # {'name': 'SiluBackward0', 'attr': 'self.t: (1, 320, 64, 64) f16\nid    :      136679675902512'}
    node = {}
    sr              = parse_str(item)
    node["name"]    = "SiluBackward0"
    node["inputs"]  = [parse_shape_dtype(sr["self.t"])]
    node["outputs"] = node["inputs"]
    node['info']    = item
    return node

@warp_node(1, 1, True, True)
def SoftmaxBackward0_node(item, pre=None):
    # y = softmax(x)
    # grad_x = grad_y * (y - y^2)
    # {'name': 'SoftmaxBackward0', 'attr': 'dim.c   : 18446744073709551615\nresult.t:    (8, 4096, 77) f16\nid      :      136679676265952'}
    node = {}
    sr              = parse_str(item)
    node["name"]    = "SoftmaxBackward0"
    node["inputs"]  = [parse_shape_dtype(sr["result.t"])]
    node["outputs"] = node["inputs"]
    node['info']    = item
    return node

# split 是多输入类型
# pre 不能表示多个输入 -> 如果只需要一个输入就可以得到答案 就选择任意输入
@warp_node(2, 1, False, False)
def SplitBackward0_node(item, pre=None):
    # since the split dim is very 
    # {'name': 'SplitBackward0', 'attr': 'dim.c           : 18446744073709551615\nself_sym_sizes.c:      (1, 4096, 2560)\nsplit_size.c    :                 1280\nid              :      136679675903616'}
    node = {}
    item = item.replace(" ", "")
    sr              = parse_str(item)
    node["name"]    = "SplitBackward0"
    # search inputs by id 
    dim             = int(sr["dim.c"])
    dim             = dim if dim < 4 else (dim - 2**64)
    output_shape    = parse_shape_dtype(sr["self_sym_sizes.c"])[0]
    dim             = dim if dim >= 0 else dim + len(output_shape)
    split_size      = int(sr["split_size.c"])
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    dtype           = pre["outputs"][output_idx][1]
    # if output shape is 4 or 3 dims ? how to do ?
    input_shape0    = []
    input_shape1    = []
    for i in range(len(output_shape)):
        if i == dim:
            input_shape0.append(split_size)
            input_shape1.append(output_shape[i] - split_size)
        else:
            input_shape0.append(output_shape[i])
            input_shape1.append(output_shape[i])
    node["inputs"]  = [ [input_shape0, dtype], [input_shape1, dtype] ]
    node["outputs"] = [[output_shape, dtype]]
    node['info']    = item
    return node

@warp_node(1, 1, False, False, False)
def TBackward0_node(item, pre=None):
    # {'name': 'TBackward0', 'attr': ''}
    item = item.replace(" ", "")
    sr   = parse_str(item)
    node = {}
    node["name"]    = "TBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    # transpose last two dims
    input_shape     = node["inputs"][0][0]
    output_shape    = []
    for i in range(len(input_shape)):
        if i == len(input_shape) - 2:
            output_shape.append(input_shape[-1])
        elif i == len(input_shape) - 1:
            output_shape.append(input_shape[-2])
        else:
            output_shape.append(input_shape[i])
    node["outputs"] = [[output_shape, node["inputs"][0][1]]]
    node['info']    = item
    return node

@warp_node(1, 1, False, True, True)
def ToCopyBackward0_node(item, pre=None):
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "ToCopyBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    cur_id          = sr["id"]
    node["outputs"] = [overhead_node_input_shape(cur_id, 0)]
    node['info']    = item
    return node

@warp_node(1, 1, False, False, False)
def TransposeBackward0_node(item, pre=None):
    # transpose
    # {'name': 'TransposeBackward0', 'attr': 'dim0.c: 18446744073709551615\ndim1.c: 18446744073709551614\nid    :      136679676266144'}
    item = item.replace(" ", "")
    node = {}
    sr              = parse_str(item)
    node["name"]    = "TransposeBackward0"
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    # output_shape 在 input_shape 上最后两个维度进行交换
    input_shape     = node["inputs"][0][0]
    output_shape    = []
    for i in range(len(input_shape)):
        if i == len(input_shape) - 2:
            output_shape.append(input_shape[-1])
        elif i == len(input_shape) - 1:
            output_shape.append(input_shape[-2])
        else:
            output_shape.append(input_shape[i])
    node["outputs"] =[ [ output_shape, node["inputs"][0][1] ]]
    node['info']    = item
    return node

@warp_node(1, 1, False, False, False)
def UnsafeViewBackward0_node(item, pre=None):
    # 用于处理某些特定情况下的 view 操作的反向传播，可能会涉及一些不安全的操作。
    # {'name': 'UnsafeViewBackward0', 'attr': 'self_sym_sizes.c: (4096, 4)\nid              : 136679676266048'}
    item = item.replace(" ", "")
    node = {}
    node["name"]    = "UnsafeViewBackward0"
    sr              = parse_str(item)
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    outputs         = parse_shape_dtype(sr["self_sym_sizes.c"])
    outputs[1]      = node["inputs"][0][1]
    node["outputs"] = [outputs]
    node['info']    = item
    return node

@warp_node(1, 1, True, False, False)
def UpsampleNearest2DBackward0_node(item, pre=None):
    # "output_size.c : (64, 64)\nscales_h.c:2.0\nscales_w.c:2.0\nself_sym_sizes.c: (1, 640, 32, 32)"
    item = item.replace(" ", "")
    node = {}
    node["name"]    = "UpsampleNearest2DBackward0"
    sr              = parse_str(item)
    dtype           = pre["outputs"][0][1]
    node["inputs"]  = [parse_shape_dtype(sr["self_sym_sizes.c"])]
    node["inputs"][0][1] = dtype
    input_shape     = node["inputs"][0][0]
    output_size     = parse_shape_dtype(sr["output_size.c"])
    output_shape    = [input_shape[0], input_shape[1], output_size[0][0], output_size[0][1]]
    node["outputs"] = [[output_shape, node["inputs"][0][1]]]
    node['info']    = item
    return node

@warp_node(1, 1, False, False)
def ViewBackward0_node(item, pre=None):
    # {'name': 'ViewBackward0', 'attr': 'self_sym_sizes.c: (1, 4096, 320)\nid              : 136679675901648'}
    item = item.replace(" ", "")
    node = {}
    node["name"]    = "ViewBackward0"
    sr              = parse_str(item)
    output_idx      = get_idx_of_output(int(sr["id"]), pre)
    node["inputs"]  = [pre["outputs"][output_idx]]
    outputs         = parse_shape_dtype(sr["self_sym_sizes.c"])
    outputs[1]      = node["inputs"][0][1]
    node["outputs"]  = [outputs]
    node['info']    = item
    return node

node_in_degree = defaultdict(int)
for node in gh_link.keys():
    node_in_degree[int(node)]
    for v in gh_link[node]:
        node_in_degree[v] += 1

# find first nodes 
next_nodes = []
visited    = set()
for node, ind in node_in_degree.items():
    if ind == 0:
        next_nodes.append([node, None])

print("init nodes: ", next_nodes)

graph_nodes_with_shape = {}
topo_res = []
# topological sort
while next_nodes:
    cur_node, pre = next_nodes.pop(0)
    visited.add(cur_node)
    # print("cur_node: ", cur_node)
    cur_node_info  = gh_node[str(cur_node)]
    shape_node     = None
    if cur_node_info['name'].startswith("empty"):
        shape_node = {}
    else:
        attr = cur_node_info["attr"]
        if "id" not in attr:
            attr += "\nid: " + str(cur_node) if attr else "id: " + str(cur_node)
        shape_node = map_tables[cur_node_info["name"]](attr, pre=graph_nodes_with_shape[str(pre)] if pre else None)
    if shape_node:
        if "136679675011584" in shape_node["info"]:
            import pdb;pdb.set_trace()
        graph_nodes_with_shape[str(cur_node)] = shape_node
        topo_res.append(shape_node)
    children = gh_link[str(cur_node)] if str(cur_node) in gh_link else []
    for child in children:
        node_in_degree[child] -= 1
        # check and create nodes 
        if node_in_degree[child] == 0:
            next_nodes.append([child, cur_node])

def handle_input_shape(shape):
    if len(shape) > 1 and isinstance(shape[0], list):
        return handle_input_shape(shape[0])
    res = [1,1,1,1]
    for i in range(len(shape)):
        res[-len(shape)+i] = shape[i]
    return res


f = open("bwd_nodes_with_shape.csv", "w")
header = "name,left_n,left_c,left_h,left_w,output1_n,output1_c,output1_h,output1_w,output2_n,output2_c,output2_h,output2_w,left_dtype,output_dtype,info\n"
f.write(header)

for item in topo_res:
    res_str = []
    res_str += [item["name"]]
    if not item["inputs"][0]:
        print(item)
        continue
    res_str += handle_input_shape(item["inputs"][0][0])
    if not item["outputs"][0]:
        res_str += [0,0,0,0]
    else:
        res_str += handle_input_shape(item["outputs"][0][0])
    if len(item["outputs"]) < 2:
        res_str += [0,0,0,0]
    else:
        if not item["outputs"][1]:
            res_str += [0,0,0,0]
        else:
            res_str += handle_input_shape(item["outputs"][1][0])

    res_str += [item["inputs"][0][1]]
    if not item["outputs"][0]:
        res_str += [item["inputs"][0][1]]
    else:
        res_str += [item["outputs"][0][1]]
    res_str += [item["info"].replace("\n", " ").replace(",", " ")]
    f.write(",".join([str(i) for i in res_str]) + "\n")

f.close()
# import pdb;pdb.set_trace()
# print("graph_nodes_with_shape: ", graph_nodes_with_shape)