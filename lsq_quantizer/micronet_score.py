from flops_counter import add_flops_counting_methods, get_model_parameters_number
import torch


def get_micronet_score(net, weight_bits, activation_bits, weight_strategy=None, activation_strategy=None,
              input_res=(3,224,224), baseline_params=6900000, baseline_MAC=1170000000):
    """Get MicroNet score for a given configuration

    Args:
        net: model instance
        weight_bits: int, bit-width for weights
        activation_bits: int, bit-width for activations
        weight_strategy: list, layerwise weight bit-widths for the network
        activation_strategy: list, layerwise activation bit-widths for the network
        input_res: tuple, input image size
        baseline_params: int, number of parameters of the baseline network
                (e.g.: MobileNetV2 is the baseline for ImageNet task)
        baseline_MAC: int, MAC count for the baseline network

    """

    flops_model = add_flops_counting_methods(net)
    flops_model.eval().start_flops_count()
    batch = torch.ones(()).new_empty((1, *input_res),
                       dtype=next(flops_model.parameters()).dtype,
                       device=next(flops_model.parameters()).device)
    _ = flops_model(batch)
    flops_count = flops_model.compute_average_flops_cost(bw_weight=weight_bits,
                                                         bw_act=activation_bits,
                                                         strategy=(weight_strategy, activation_strategy))
    params_count = get_model_parameters_number(flops_model, bw_weight=weight_bits, w_strategy=weight_strategy)
    flops_model.stop_flops_count()

    print("Number of parameters:", params_count)
    print("MAC count:", flops_count)

    score = params_count/float(baseline_params) + flops_count/float(baseline_MAC)

    return score
