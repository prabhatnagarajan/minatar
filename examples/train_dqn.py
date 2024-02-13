import argparse
import inspect


import numpy as np
import torch
import torch.nn as nn

import pfrl
from pfrl import agents, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN
from pfrl.wrappers import atari_wrappers

from pfrl.q_functions import DiscreteActionValueHead
from pfrl import experiments

import gymnasium

from pdb import set_trace


class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        in_channels = env.game.state_shape()[2]
        n_actions = env.action_space.n
        self.hidden_layers = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=1024, out_features=128),
                nn.ReLU(),
        )
        self.output_layer = nn.Sequential(nn.Linear(in_features=128, out_features=n_actions),
                             DiscreteActionValueHead(),)

    def forward(self, x ):
        # Forward pass through the hidden layers
        hidden_output = self.hidden_layers(x)
        return self.output_layer(hidden_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        choices=['MinAtar/Breakout-v1', 'MinAtar/Asterix-v1', 'MinAtar/Freeway-v1', 'MinAtar/Seaquest-v1', 'MinAtar/SpaceInvaders-v1'],
        help="MinAtar environment to do experiments on.",
        required=True,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--final-exploration-frames",
        type=int,
        default=2.5 * 10**5,
        help="Timesteps after which we stop " + "annealing exploration rate",
    )
    parser.add_argument(
        "--final-epsilon",
        type=float,
        default=0.01,
        help="Final value of epsilon during training.",
    )
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.001,
        help="Exploration epsilon used during eval episodes.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 * 10**6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=10**3,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=10**3,
        help="Frequency (in timesteps) at which " + "the target network is updated.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=2.5 * 10**4,
        help="Frequency (in timesteps) of evaluation phase.",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=4,
        help="Frequency (in timesteps) of network updates.",
    )
    parser.add_argument("--eval-n-runs", type=int, default=10)
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )
    parser.add_argument('--no-target-network', action='store_true')
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2**31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = gymnasium.make(args.env)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0,
        args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions),
    )

    rbuf = replay_buffers.ReplayBuffer(10**5, 1)

    def phi(x):
        x = np.rollaxis(x,2)
        # Feature extractor
        return np.asarray(x, dtype=np.float32)

    n_actions = env.action_space.n
    obs_size = env.observation_space.shape[0]

    q_func = QNetwork(env)

    opt = torch.optim.Adam(q_func.parameters(), lr=args.lr, eps=3.125e-4)

    agent = pfrl.agents.DQN(q_function=q_func,
                            optimizer=opt,
                            replay_buffer=rbuf,
                            gpu=args.gpu,
                            gamma=0.99,
                            explorer=explorer,
                            replay_start_size=args.replay_start_size,
                            target_update_interval=args.target_update_interval,
                            clip_delta=False, # TODO: Check correctness
                            update_interval=args.update_interval,
                            batch_accumulator='sum',
                            phi=phi,
                            )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=None, n_episodes=args.eval_n_runs
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:

        def reward_phi(x):
            return x

        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            checkpoint_freq=args.checkpoint_frequency,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            eval_env=eval_env,
            eval_during_episode=True,
        )

if __name__ == "__main__":
    main()
