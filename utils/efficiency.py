import csv
import json
import os

import torch
from torch import nn


def synchronize_device(device):
    if device is None:
        return
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.synchronize(device)


def empty_device_cache(device):
    if device is None:
        return
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.empty_cache()


def reset_peak_memory(device):
    if device is None:
        return
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.reset_peak_memory_stats(device)


def get_peak_memory_mb(device):
    if device is None:
        return {'allocated_mb': None, 'reserved_mb': None}
    if device.type == 'cuda' and torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            'reserved_mb': torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        }
    if device.type == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available():
        return {
            'allocated_mb': torch.npu.max_memory_allocated(device) / (1024 ** 2),
            'reserved_mb': torch.npu.max_memory_reserved(device) / (1024 ** 2),
        }
    return {'allocated_mb': None, 'reserved_mb': None}


def count_parameters(module):
    params = list(module.parameters())
    buffers = list(module.buffers())
    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    buffer_elems = sum(b.numel() for b in buffers)
    param_bytes = sum(p.numel() * p.element_size() for p in params)
    buffer_bytes = sum(b.numel() * b.element_size() for b in buffers)
    return {
        'params': int(total_params),
        'trainable_params': int(trainable_params),
        'buffers': int(buffer_elems),
        'model_size_mb': (param_bytes + buffer_bytes) / (1024 ** 2),
    }


def count_modules_parameters(modules):
    totals = {
        'params': 0,
        'trainable_params': 0,
        'buffers': 0,
        'model_size_mb': 0.0,
    }
    for module in modules:
        stats = count_parameters(module)
        for key in totals:
            totals[key] += stats[key]
    totals['params'] = int(totals['params'])
    totals['trainable_params'] = int(totals['trainable_params'])
    totals['buffers'] = int(totals['buffers'])
    return totals


def _numel(output):
    if isinstance(output, torch.Tensor):
        return output.numel()
    if isinstance(output, (list, tuple)):
        return sum(_numel(item) for item in output)
    return 0


def _conv2d_macs(module, inputs, output):
    x = inputs[0]
    batch = x.shape[0]
    out_h, out_w = output.shape[-2:]
    kernel_h, kernel_w = module.kernel_size
    out_channels = module.out_channels
    in_channels = module.in_channels
    groups = module.groups
    macs_per_position = kernel_h * kernel_w * in_channels * out_channels // groups
    bias_ops = out_channels if module.bias is not None else 0
    return int(batch * out_h * out_w * (macs_per_position + bias_ops))


def _linear_macs(module, inputs, output):
    return int(_numel(output) * module.in_features)


def _batchnorm_macs(module, inputs, output):
    return int(_numel(output) * 2)


def _pool_macs(module, inputs, output):
    if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        return int(inputs[0].numel())
    kernel = module.kernel_size
    if isinstance(kernel, tuple):
        kernel_ops = kernel[0] * kernel[1]
    else:
        kernel_ops = kernel * kernel
    return int(_numel(output) * kernel_ops)


def estimate_forward_macs(module, input_shape, device):
    was_training = module.training
    module.eval()
    macs = {'total': 0}
    supported = (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.MaxPool2d,
                 nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)
    hooks = []

    def hook(mod, inputs, output):
        if isinstance(mod, nn.Conv2d):
            macs['total'] += _conv2d_macs(mod, inputs, output)
        elif isinstance(mod, nn.Linear):
            macs['total'] += _linear_macs(mod, inputs, output)
        elif isinstance(mod, nn.BatchNorm2d):
            macs['total'] += _batchnorm_macs(mod, inputs, output)
        elif isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            macs['total'] += _pool_macs(mod, inputs, output)

    for child in module.modules():
        if isinstance(child, supported):
            hooks.append(child.register_forward_hook(hook))

    try:
        with torch.no_grad():
            dummy = torch.randn(*input_shape, device=device)
            synchronize_device(device)
            module(dummy)
            synchronize_device(device)
    finally:
        for handle in hooks:
            handle.remove()
        module.train(was_training)

    return int(macs['total'])


def save_results(results, output_dir, output_prefix):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, '{}.json'.format(output_prefix))
    csv_path = os.path.join(output_dir, '{}.csv'.format(output_prefix))

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    rows = []
    for item in results.get('methods', []):
        row = {}
        _flatten_dict(item, row)
        rows.append(row)

    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    return json_path, csv_path


def _flatten_dict(src, dst, prefix=''):
    for key, value in src.items():
        flat_key = '{}{}'.format(prefix, key)
        if isinstance(value, dict):
            _flatten_dict(value, dst, flat_key + '.')
        else:
            dst[flat_key] = value
