# Village People, 2017

def conv_out_dim(w, conv):
    """Assumes square maps and kernels"""

    k = conv.kernel_size[0]
    s = conv.stride[0]
    p = conv.padding[0]
    return int((w - k + 2 * p) / s + 1)
