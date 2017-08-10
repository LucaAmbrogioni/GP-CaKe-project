# GP-CaKe-project
## Bayesian Effective Connectivity

The presented code allows for estimation of effective connectivity using Gaussian Processes with 'causal kernels'. Please see the notebook demo for an example of how to use the code. More details on the method are available in the accompanying paper: https://arxiv.org/abs/1705.05603.

### Change log

#### August 10th, 2017

- Added parallel computation functionality; simply set the parallelthreads attribute of the gpcake() object before executing gpcake.run_analysis(), and computation will be in parallel over trials. Note that due to overhead, the speed increase is sub-linear (for example, computing the example for 500 testing samples using 5 processes vs only 1, results in a 2.8x speed increase, while 10 processes results in a 3.4x increase).
