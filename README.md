# ML Competition Framework

A minimal framework for ML competitions with Hydra configs, WandB sweeps, and modular components. It is meant as a starting point for competitions, and is not a production-ready framework. It's also not made with the idea in mind to continuously improve the framework.

## The general setup

The most general setup for a competition is as follows:

1. Lots of augmentations for regularization.
2. Lots of different models to try, always from a pretrained model.
3. Cosine scheduling for the learning rate.
4. AdamW optimizer.
5. A couple ways to ensemble the models.
6. Sweeping. Lots and lots of sweeping. The most important parameters are often the learning rate, batch size, and model architecture.

There are other things you can do besides this, like synthetic data generation or postprocessing functions for example, but you're basically guaranteed to be doing the above.

Things that sometimes help:

1. Using exponential moving averages of your model weights. We've found this to also be quite a safe default, and there's a good implementation [here](https://github.com/lucidrains/ema-pytorch).
2. Postprocessing functions. Postprocessing functions can sometimes be a good way to improve your score. We've used them to smooth predictions for the Malaria competition and the Wave Inversion competition. But a really big pitfall is that people often make postprocessing functions you don't need. Models have a crazy ability to learn patterns, so if something helps the model probably already does it. So before you create a postprocessing function, first verify by looking at plenty of model predictions that it's actually necessary (otherwise you'll waste a lot of time).

## Learnings of the past year

### WandB

#### WandB usage

We do recommend WandB initially, as it's very easy to set up and use. It's good for logging metrics, and most importantly for running the same sweep across multiple machines. Using a config like the sweep_config.yaml in this repo is a good starting point. 

#### WandB logging

To log metrics, you put statements in your code like the following:

```python
wandb.log({"train/accuracy": accuracy})
```

This will log the accuracy metric to wandb, and you can view it in the wandb dashboard.
It even handily seperates 'train' and 'val' metrics if you use / to seperate the metric name.

#### WandB sweeps

To run a sweep, create a .yaml file that defines your sweep, and run the following command:

```bash
wandb sweep <yaml-path>
```

In our case, this would be:

```bash
wandb sweep configs/sweeps/sweep_config.yaml
```

This will return a sweep command, which looks like this:

```bash
wandb agent <user>/<project-name>/<sweep-id>
```

If you are all part of the same wandb team, then you can run this command on multiple machines to run the same sweep across multiple machines, significantly speeding up the process.

A lot of our PCs have 2 GPUs, and distributed training can be a pain in the ass. So instead, when sweeping, we tend to run an instance of the sweep on each GPU. To do this, open 2 terminal windows, and run the following commands:

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent <user>/<project-name>/<sweep-id>
```

```bash
CUDA_VISIBLE_DEVICES=1 wandb agent <user>/<project-name>/<sweep-id>
```

This will run the sweep on both GPUs independently, side-stepping the problems distributed training can cause.

#### WandB problems

In the past year, we've used WandB a lot for our experiment tracking and hyperparameter optimization, and found that in a lot of cases it breaks in unexpected ways. It also lacks feature completeness, like being able to pick your own algorithm for hyperparameter optimization, or being able to use uv for running sweeps (you HAVE to activate your virual environment first). And the reason this tends to be a time consuming issue every time is because their documentation is wildly incomplete, and their support is not very helpful. MLFlow is more difficult to set up, but we found it to be more reliable and feature complete. It's also Bring Your Own Hyperparameter Optimization (BYOHPO) friendly.
