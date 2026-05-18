# numa-topology

Discover GPU NUMA affinity via sysfs. Zero dependencies beyond stdlib and `nvidia-smi`.

```
pip install numa-topology
```

## Why

On multi-socket servers (e.g. dual EPYC + 8× H100), GPU memory access latency depends
on whether the CPU issuing the transfer is on the same NUMA node as the GPU. Wrong
placement can cost 15–30% throughput silently.

`numa-topology` reads `/sys/bus/pci/devices/<pci-id>/numa_node` for each GPU so your
inference engine can make topology-aware decisions without pulling in a heavy dependency.

## Usage

```python
from numa_topology import Topology

topology = Topology.discover()
print(topology.gpu_to_numa)
# {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}

# Which GPUs share a NUMA node?
topology.gpus_in_numa(0)   # [0, 1, 2, 3]
topology.gpus_in_numa(1)   # [4, 5, 6, 7]

# Are two GPUs on the same node?
topology.same_numa(0, 1)   # True
topology.same_numa(0, 4)   # False
```

Works without PyTorch — falls back to parsing `nvidia-smi` for the GPU count.

## vLLM

```python
from numa_topology import Topology

topology = Topology.discover()

# Assign tensor-parallel groups to same-NUMA GPUs
for node in topology.numa_nodes():
    gpus = topology.gpus_in_numa(node)
    print(f"NUMA {node}: GPUs {gpus}")
    # NUMA 0: GPUs [0, 1, 2, 3]
    # NUMA 1: GPUs [4, 5, 6, 7]

# Check before cross-GPU KV transfers
src_gpu, dst_gpu = 0, 5
if not topology.same_numa(src_gpu, dst_gpu):
    print("Cross-NUMA transfer — expect higher latency")
```

## PyTorch

```python
import torch
from numa_topology import Topology

topology = Topology.discover()

# Pin DataLoader workers to NUMA-local CPUs (combine with numactl or psutil)
gpu = int(os.environ.get("LOCAL_RANK", 0))
numa_node = topology.gpu_to_numa[gpu]
print(f"GPU {gpu} is on NUMA node {numa_node}")

# Keep all-reduce within a NUMA domain when possible
numa0_gpus = topology.gpus_in_numa(0)
devices = [torch.device(f"cuda:{g}") for g in numa0_gpus]
```

## SGLang

```python
from numa_topology import Topology

topology = Topology.discover()

# Build a NUMA-aware device map for pipeline parallelism
num_layers = 32
layers_per_gpu = num_layers // len(topology.gpu_to_numa)

device_map = {}
for layer in range(num_layers):
    gpu = list(topology.gpu_to_numa.keys())[layer // layers_per_gpu]
    device_map[f"model.layers.{layer}"] = gpu
```

## Loading from a file

Useful for CI, simulation, or heterogeneous cluster configs:

```python
from pathlib import Path
from numa_topology import Topology

# JSON format: {"gpu_to_numa": {"0": 0, "1": 0, "4": 1, "5": 1}}
topology = Topology.from_file(Path("my_cluster.json"))
```

## Behaviour on non-NUMA machines

On single-socket machines or when `numa_node` is not in sysfs, all GPUs are
assigned to NUMA node 0. `same_numa()` returns `True` for all pairs.

```python
topology = Topology.flat(4)   # explicit flat fallback
```

## Requirements

- Python ≥ 3.10
- Linux with sysfs (`/sys/bus/pci/devices/`)
- `nvidia-smi` on `$PATH`
- PyTorch is **optional** — the package falls back to `nvidia-smi` for GPU discovery

## License

Apache 2.0
