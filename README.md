# GP-CaKe-project
## Bayesian Effective Connectivity

The presented code allows for estimation of effective connectivity using Gaussian Processes with 'causal kernels'. Please see the notebook demo for an example of how to use the code. More details on the method are available in the accompanying paper: https://arxiv.org/abs/1705.05603.

### Change log

#### August 11th, 2017

- Added parallel computation functionality; simply set the parallelthreads attribute of the gpcake() object before executing gpcake.run_analysis(), and computation will be in parallel over both trials and edges.
