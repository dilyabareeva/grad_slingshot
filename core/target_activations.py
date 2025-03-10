import torch


def vit_cls_token():
    def get_target_ct(activations):
        return activations[:, 0, :]
    return get_target_ct


def vit_cls_token_trump_direction(probe_path, device):
    probe = torch.load(probe_path).unsqueeze(1).float().to(device)

    def get_target_ct(activations):
        return torch.matmul(activations, probe)[0]

    return get_target_ct


def cnn_conv_filters():
    def get_target_act(activations):
        return activations.mean(dim=(-2, -1))
    return get_target_act
