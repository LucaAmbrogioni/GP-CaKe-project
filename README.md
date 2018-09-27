# GP-CaKe-project
## Bayesian Effective Connectivity

The presented code allows for estimation of effective connectivity using Gaussian Processes with 'causal kernels'. Please see the notebook demo for an example of how to use the code. More details on the method are available in the accompanying paper: http://papers.nips.cc/paper/6696-gp-cake-effective-brain-connectivity-with-causal-kernels.pdf.

### Change log

#### September 27th, 2018

- Added functionality for alternative emission models. See https://www.biorxiv.org/content/early/2018/06/06/340489 for an example.

#### August 10th, 2017

- Added parallel computation functionality; simply set the parallelthreads attribute of the gpcake() object before executing gpcake.run_analysis(), and computation will be in parallel over trials.

