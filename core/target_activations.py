def vit_cls_token():
    def get_target_ct(activations):
        return activations[:, 0, :]
    return get_target_ct


def cnn_conv_filters():
    def get_target_act(activations):
        return activations.mean(dim=(-2, -1))
    return get_target_act
