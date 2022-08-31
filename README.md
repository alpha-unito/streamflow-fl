# Federated Learning with StreamFlow

This repository contains a [StreamFlow](https://streamflow.di.unito.it) Federated Learning (FL) pipeline based on [PyTorch](https://pytorch.org/). The workflow trains a [VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html) model with [Group Normalization](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html) over two datasets:

- A standard version of [MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html);
- A grayscaled version of [SVHN](https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html).

The workflow is described with an extended version of [CWL](https://commonwl.org) that introduces support for the [Loop](https://github.com/common-workflow-language/cwltool/pull/1641) construct, necessary to describe the training-aggregate iteration of FL workloads.

Datasets have been placed onto two different HPC facilities:

- MNIST has been trained on the [EPITO](https://hpc4ai.unito.it/documentation/) cluster at the University of Torino (1 80-core Arm Neoverse N1, 512GB RAM, and 2 NVIDIA A100 GPU per node);
- SVHN has been trained on the CINECA [MARCONI100](https://www.hpc.cineca.it/hardware/marconi100) cluster in Bologna (2 16-core IBM POWER9 AC922, 256GB RAM, and 4 NVIDIA V100 GPUs per node).

Since HPC worker nodes cannot access the Internet through outbound connections, this workload cannot be managed by FL frameworks that require direct bidirectional connections between worker and aggregator nodes. Conversely, StreamFlow relies on a pull-based data transfer mechanism that overcomes this limitation.

To also perform a direct comparison between StreamFlow and the Intel [OpenFL](https://openfl.readthedocs.io/en/latest/index.html) framework, the pipeline has also been executed over two VMs (8 cores, 32GB
RAM, 1 NVIDIA T4 GPU each) hosted on the [HPC4AI](https://hpc4ai.unito.it/) Cloud at the University of Torino, acting as workers. Conversely, the aggregation plane has always been placed on Cloud.

## Usage

To run the experiment as is, clone [this](https://github.com/alpha-unito/streamflow-fl) repository on the aggregator node and use the following commands:

```bash
python -m venv venv
source venv/bin/activate
pip install "streamflow==0.2.0.dev1"
pip install -r requirements.txt
streamflow run streamflow.yml
```

Reproducing the experiments in the same environment requires access to both HPC facilities and the HPC4AI Cloud. However, interested users can run the same pipeline on their preferred infrastructure by changing the `deployments` definitions in the `streamflow.yml` file and the corresponding Slurm/SSH scripts inside the `environments` folder.

Also, note that the Python dependencies listed in the `requirements.txt` file should be manually installed in any involved location (both the workers and the aggregator), and the datasets are supposed to be already present in the worker nodes.

## Contributors

Iacopo Colonnelli <iacopo.colonnelli@unito.it>  
Bruno Casella <bruno.casella@unito.it>  
Marco Aldinucci <marco.aldinucci@unito.it>  
