#!/bin/bash
# Install core dependencies
pip install -r requirements.txt

# Conditionally install flash-attn if device capability is 8 or higher
if [ $(python -c "import torch; print(torch.cuda.get_device_capability()[0])") -ge 8 ]; then
    pip install flash-attn==0.2.0
fi
