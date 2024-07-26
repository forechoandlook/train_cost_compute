


activation_op_calc_fn_dict = {}

def warp_activation_calc(func):
    name = func.__name__
    op_name = name[:-len("act_calc")]
    activation_op_calc_fn_dict[op_name] = func
    def warp(*args, **kwargs):
        return func(*args, **kwargs)
    return warp

def basic_act_calc(node):
    # basic: output is the activation
    pass

@warp_activation_calc
def conv2d_act_calc(node):
    return basic_act_calc(node)

@warp_activation_calc
def linear_act_calc(node):
    return basic_act_calc(node)

# add more op 


