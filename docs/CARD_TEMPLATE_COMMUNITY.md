---
library_name: kernels
{% if license %}license: {{ license }}
{% endif %}---

> [!IMPORTANT]
> This is a **community kernel**, published from
> [ROCm/rocm-community-kernels](https://github.com/ROCm/rocm-community-kernels). AMD
> reviewed it for policy and safety and tested it on AMD GPUs before publishing, but it
> is **not supported by AMD**. Report issues to its maintainers in that repository.

This is the repository card of {{ repo_id }}. It was built with
[kernel-builder](https://github.com/huggingface/kernels) to be used with the
[`kernels` library](https://github.com/huggingface/kernels).

## How to use
{% if functions %}

```python
# make sure `kernels` and PyTorch for ROCm are installed: `pip install -U kernels`
from kernels import get_kernel

kernel_module = get_kernel("{{ repo_id }}", version={{ version }})
{{ functions[0] }} = kernel_module.{{ functions[0] }}

{{ functions[0] }}(...)
```
{% else %}

Usage example not available.
{% endif %}

## Available functions
{% if functions %}
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% else %}

Function list not available.
{% endif %}
{% if layers %}

## Available layers
{% for layer in layers %}
- `{{ layer }}`
{% endfor %}
{% endif %}

## Supported hardware

<!-- The GPUs (rocm-archs) this kernel was tested on, e.g. MI350 (gfx950). -->

## Maintainers

<!-- GitHub handles of this kernel's maintainers. -->
{% if upstream %}

## Source code

This kernel's source originally comes from {{ upstream }}.
{% endif %}
