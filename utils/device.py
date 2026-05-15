import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def npu_is_available():
    return hasattr(torch, 'npu') and torch.npu.is_available()


def resolve_device(args):
    device = getattr(args, 'device', 'auto').lower()

    # Once visible devices are masked by the launcher/env, the selected card is
    # exposed inside this process as logical device 0.
    device_id = 0

    if device == 'auto':
        if npu_is_available():
            device = 'npu'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    if device == 'npu':
        if not npu_is_available():
            raise RuntimeError('NPU device was requested, but torch_npu/torch.npu is not available.')
        npu_device = torch.device('npu:{}'.format(device_id))
        torch.npu.set_device(npu_device)
        return npu_device

    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device was requested, but CUDA is not available.')
        torch.cuda.set_device(device_id)
        return torch.device('cuda:{}'.format(device_id))

    if device == 'cpu':
        return torch.device('cpu')

    raise ValueError('Unsupported device: {}'.format(device))


def setup_seed(seed, device=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device is not None and device.type == 'npu' and npu_is_available():
        torch.npu.manual_seed_all(seed)
