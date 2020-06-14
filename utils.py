import numpy as np

def get_clean_name(string):
    if "depth" in string.lower() and "kernel" in string.lower():
        return string.split('/')[0] + '/' + 'Kernel'
    elif "depth" in string.lower() and "bias" in string.lower():
        return string.split('/')[0] + '/' + 'Bias'
    elif "conv2d" in string.lower() and "kernel" in string.lower():
        return string.split('/')[0] + '/' + 'Kernel'
    elif "conv2d" in string.lower() and "bias" in string.lower():
        return string.split('/')[0] + '/' + 'Bias'
    elif "lu" in string.lower():
        return string.split('/')[0] + '/' + "Alpha"
    else:
        raise ValueError("Input string not understood")
        
exception_mapping = {
    "depthwise_conv2d_18/depthwise_kernel" : "depthwise_conv2d_22/Kernel",
    "depthwise_conv2d_18/bias": "depthwise_conv2d_22/Bias",
    "conv2d_21/kernel" : "conv2d_27/Kernel",
    "conv2d_21/bias": "conv2d_27/Bias",
    "p_re_lu_20/alpha": "p_re_lu_25/Alpha"
}

def restore_variables(model,tf_lite_mapping, data_format):
    channels_first = True if data_format == "channels_first" else False
    restored = 0
    total_params = 0
    for var in model.variables:
        try:
            name = get_clean_name(var.name)
            weight = tf_lite_mapping[name]
        except KeyError:
            map_string = exception_mapping[var.name[:-2]]
            name = get_clean_name(map_string)
            weight = tf_lite_mapping[name]
        if weight.ndim == 4:
            weight = np.transpose(weight, (1,2,3,0)) # conv transpose
        elif weight.ndim ==3:
            if channels_first: weight = np.transpose(weight, (2, 0, 1)) #prelu_transpose
        total_params += np.product(weight.shape)
        var.assign(weight)
        print("{} assinged with {}".format(var.name, name))
        restored += 1
    print("Restored {} variables from tflite file".format(restored))
    print("Restore {} float values".format(total_params))