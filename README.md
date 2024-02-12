# Poppy: Population-Based Reinforcement Learning for Combinatorial Optimization

---

This repository contains the official JAX implementation of the paper [Population-Based Reinforcement Learning for Combinatorial 
Optimization](https://arxiv.org/abs/2210.03475).

---

## Overview :hibiscus:
Though applying reinforcement learning to combinatorial optimization is attractive, it is unrealistic to expect an agent 
to solve these (often NP-)hard problems in a single shot due to their inherent complexity.
Poppy is a method that uses a _population_ of agents with suitably diverse policies to improve the exploration of the
solution space of hard _combinatorial optimization_ problems, such as 
[TSP](https://en.wikipedia.org/wiki/Travelling_salesman_problem), 
[CVRP](https://en.wikipedia.org/wiki/Vehicle_routing_problem) or 
[Knapsack](https://en.wikipedia.org/wiki/Knapsack_problem). To this end, it uses a new RL objective to induce an 
unsupervised specialization targeted solely at maximizing the performance of the whole population.

<p align="center">
    <a href="img/tsp.gif">
        <img src="img/tsp.gif" width="100%"/>
    </a>
    <i>Figure 1: A diverse set of TSP solvers taking different routes in a given instance.</i>    
</p>

## Getting Started :rocket:
### Installation

A Dockerfile is provided to run the code. Currently, the installation uses [JAX](https://jax.readthedocs.io/en/latest/), 
which enables to run our models seamlessly across different hardware (CPU, GPU, TPU). To build the Docker image, start a
container, and enter it, run the following from the root directory of the repository.

```shell 
# Build the container (CPU or GPU)
sudo docker build -t poppy --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg BUILD_FOR_TPU="false" -f Dockerfile .

# Build the container (TPU)
sudo docker build -t poppy --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg BUILD_FOR_TPU="true" -f Dockerfile .

# Start the container
sudo docker run -itd --rm --privileged -p 8889:8889 --network host --name poppy_container -v $(pwd):/app poppy

# Get in the container
sudo docker exec -it poppy_container /bin/bash

# Kill the container (once you are done with it)
sudo docker kill poppy_container
```

_Note that the command for building the Docker image changes depending on the hardware being used (CPU, GPU or TPU)._

### Training 
The models can be trained using the `experiments/run_training.py` script by executing:
```shell
python experiments/run_training.py <config>
```
where `config` is the path to a YAML configuration file specifying the main parameters we have modified across
executions, such as the environment or the population size. An example of such file is given below (a description of the
fields is given next to them).

```yaml
environment:
  name: tsp  # Other values: cvrp, knapsack
  params:
    #  CVRP Params (other pairs are <150, 55>, <150, 60>)
    num_nodes: 100
    norm_factor: 50

    # Knapsack Params (other num_items could be 200)
    num_items: 100
    total_budget: 25

    # TSP Params (other values are 125, 150)
    num_cities: 100

num_steps: 1000                 # Number of training steps
validation_freq: 500            # Every how many steps validation runs

pop_size: 1                     # Population size (e.g. 1, 4, 8, 16, 32)
num_starting_points: 20         # Number of starting points (must lower or equal than the problem size)
train_best: False               # Whether to use the Poppy objective to train the agents
train_encoder: True             # Whether to train the encoder or keep its parameters frozen

num_problems_training: 64       # Batch size (number of problems for each training step)
minibatch_size_training: 64     # Training is run in minibatches (should be a divisor of num_problems_training)
num_problems_validation: 128    # Number of problems to run validation on (the set is randomly generated and fixed)
minibatch_size_validation: 128  # Validation is run in minibatches (should be a divisor of num_problems_validation)

save_checkpoint: True           # Whether to save checkpoints
save_best: False                # Whether to save a new checkpoint upon achieving best performance (o.w. save the last one)

load_checkpoint: ""             # Path to a checkpoint to be loaded (empty string -> do not load anything)
load_decoder: False             # Whether to load the decoder(s) from the checkpoint
load_optimizer: False           # Whether to load the optimizers from the checkpoint
```

#### Training Pipeline
The training of our models is divided into 3 phases, as explained in the paper. We briefly describe here which
parameters should be used in each of these phases for a (final) 4-agent population:
1. Train a single-decoder architecture (i.e. single agent) is trained from scratch.
```yaml
pop_size: 1
train_best: False
train_encoder: True
load_checkpoint: ""
```
2. The trained decoder is discarded and a population of decoder heads is randomly initialized. With the parameters of 
the shared encoder frozen, the decoders are trained in parallel using the same training instances.
```yaml
pop_size: 4
train_best: False
train_encoder: False
load_checkpoint: "path to the checkpoint produced in Phase 1"
load_decoder: False
load_optimizer: False
```
3. The encoder is unfrozen and trained jointly with the population decoders using the Poppy objective.
```yaml
pop_size: 4
train_best: True
train_encoder: True
load_checkpoint: "path to the checkpoint produced in Phase 2"
load_decoder: True
load_optimizer: True
```

### Testing
* The evaluation sets for TSP100 and CVRP100 are due to [Kool et al.](https://github.com/wouterkool/attention-learn-to-route).
* The evaluation sets for TSP125, TSP150, CVRP125 and CVRP150 are due to [Hottung et al.](https://arxiv.org/pdf/2106.05126.pdf) (obtained through personal
communication). 
* The evaluation sets for Knapsack have been produced by us.

We describe two ways of evaluating the models below.

#### Greedy Evaluation
This evaluation runs the greedy policy (i.e. the action with the highest probability is selected). The trained models 
can be tested using the `experiments/run_evaluation.py` script by executing the command below:
```shell
python experiments/run_evaluation.py <env> <problem_size> <pop_size> <model_path>
```
where
* `env` - Name of the environment (`cvrp`, `knapsack` or `tsp`).
* `problem_size` - Size of the problem (number of nodes in CVRP, number of items in Knapsack and number of cities in TSP).
The problem sizes we have test sets for are: 100, 125 and 150 for CVRP and TSP, and 100 and 200 for Knapsack.
* `pop_size` - Population size with which the model was trained.
* `model_path` - Path to the model's folder (i.e. a checkpoint generated with the training code).

After a variable amount of time, the script will output the corresponding performance metric (e.g., tour length in TSP).

#### Sampling Evaluation
This evaluation runs stochastic rollouts for the best <agent, starting point> pairs. First, a greedy rollout is run on 
each starting position for each agent. Second, the best pairs are selected (as many as agents) and 200 stochastic
rollouts are run for each of them. The best performance is then returned.

The trained models can be tested using the `experiments/run_sampling.py` script by executing the command below:
```shell
python experiments/run_sampling.py <env> <problem_size> <pop_size> <model_path>
```
where the parameters are like those for the greedy evaluation.

## Reference :pencil2:
If you find this repository useful in your work, please use the following citation:
```
@article{grinsztajn2022poppy,
  author    = {Grinsztajn, Nathan and Furelos-Blanco, Daniel and Barrett, Thomas D.},
  title     = {Population-Based Reinforcement Learning for Combinatorial Optimization},
  journal   = {arXiv preprint arXiv:2210.03475},
  year      = {2022}
}
```

## Acknowledgements :pray:
This research has been supported with TPUs from [Google's TPU Research Cloud (TRC)](https://sites.research.google/trc/).
