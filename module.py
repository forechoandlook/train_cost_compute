warp_module_dict = {}
def warp_module():
    def _warp(func):
        warp_module_dict[func.__name__] = func
        def warp(*args, **kwargs):
            return func(*args, **kwargs)
        return warp
    return _warp

@warp_module()
def Conv2d(node):
    comment = node["comment"]
    # "Conv2d(4, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))"
    # parse the memory size
    kernel_in = int(comment.split(",")[0].split("(")[1])
    kernel_out = int(comment.split("(")[1].split(",")[1])
    kernel_h = int(comment.split("(")[2].split(",")[0])
    kernel_w = int(comment.split("(")[2].split(",")[1].split(")")[0])
    return kernel_in * kernel_out * kernel_h * kernel_w + kernel_out

@warp_module()
def GroupNorm(node):
    # GroupNorm(32, 320, eps=1e-05, affine=True)
    comment = node["comment"]
    norm_size = int(comment.split(",")[1])
    affine = 2 if "True" in comment else 0
    return norm_size * affine

@warp_module()
def Linear(node):
    # "Linear(in_features=320, out_features=320, bias=False)
    comment = node["comment"]
    in_features = int(comment.split(",")[0].split("=")[1])
    out_features = int(comment.split(",")[1].split("=")[1])
    bias = 1 if "True" in comment else 0
    return in_features * out_features + out_features * bias

@warp_module()
def LayerNorm(node):
    # LayerNorm(320, eps=1e-05, elementwise_affine=True)
    # LayerNorm((320,), eps=1e-05, elementwise_affine=True
    comment = node["comment"]
    norm_size = int(comment.split(",")[0].replace("(", "").replace("LayerNorm", ""))
    affine = 2 if "True" in comment else 0
    return norm_size * affine

warp_module_dict['LoRACompatibleConv']   = Conv2d
warp_module_dict["LoRACompatibleLinear"] = Linear

def calc_module_paraments(node):
    name = node["name"]
    if name in warp_module_dict:
        return warp_module_dict[name](node)
    else:
        return 0