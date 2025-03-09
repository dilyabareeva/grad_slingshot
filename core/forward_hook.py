import torch


def get_nested_attr(obj, attr):
    for part in attr.split("."):
        if "[" in part:
            part, index = part.split("[")
            index = int(index[:-1])
            obj = getattr(obj, part)[index]
        else:
            obj = getattr(obj, part)
    return obj


class ForwardHook:
    def __init__(self, model, layer_str, device):
        self.activation = {}
        self.hook = (
            get_nested_attr(model, layer_str)
            .to(device)
            .register_forward_hook(self.get_activation(self.activation, layer_str))
        )

    def get_activation(self, activation, name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                activation[name] = torch.squeeze(output[0])
            else:
                activation[name] = torch.squeeze(output)

        return hook

    def close(self):
        self.hook.remove()
