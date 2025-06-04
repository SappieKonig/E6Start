# ML Competition Framework

A minimal framework for ML competitions with Hydra configs, WandB sweeps, and modular components. It is meant as a starting point for competitions, and is not a production-ready framework. It's also not made with the idea in mind to continuously improve the framework.

Note! There's a lot of things we don't go into deep detail about. If you want to know more, deep research on both google's side and openai's side are incredibly good resources. USE THEM.

## Exploratory Data Analysis (EDA)

Before starting, and at the start, it's important to do EDA. Because quite often the biggest problem you have in machine learning is that you're not limited by your model capabilities, but by your data. In the Malaria competition our models outperformed the data (by our judgement) after ~5 weeks, and in a satellite imagery competition after ~2 weeks. After that point, it becomes hard to judge whether a change will help or hurt. That's on the side of checking whether your data is of high enough quality. But even if it is, it's still important to do EDA to get a better understanding of your data. This will inform model choices, augmentation choices, and postprocessing choices.

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

### uv

uv is a great package manager, where the main advantage is speed. It mimics pip in functionality, and also creates a .venv folder in the current directory. But installation time of a package is near 0, and it has significantly better dependency resolution. The documentation of uv is good, so you can find anything specific there, but some short notes:

To create a project, run:

```bash
uv init
```

To install a package, run:

```bash
uv add <package-name>
```

And to run a script, run:

```bash
uv run <script-name>
```

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
uv run wandb sweep configs/sweeps/sweep_config.yaml
```

This will return a sweep command, which looks like this:

```bash
uv run wandb agent <user>/<project-name>/<sweep-id>
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

In the past year, we've used WandB a lot for our experiment tracking and hyperparameter optimization, and found that in a lot of cases it breaks in unexpected ways. It also lacks feature completeness, like being able to pick your own algorithm for hyperparameter optimization. And the reason this tends to be a time consuming issue every time is because their documentation is wildly incomplete (even claude often can't help). MLFlow is more difficult to set up, but we found it to be more reliable and feature complete. It's also Bring Your Own Hyperparameter Optimization (BYOHPO) friendly. Because of WandB's simplicity though, we still recommend it as a starting point.
