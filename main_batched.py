import argparse
import gym
import torch
import time
from collections import defaultdict

import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

import attention


class Policy(nn.Module):
    def __init__(self, agent):
        super(Policy, self).__init__()
        self.agent = agent

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, observations, prev_reward, prev_action):
        """Sample action from agents output distribution over actions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Unsqueeze to give a batch size of 1.
        state = torch.from_numpy(np.array(observations)).float().to(device)
        action_scores, _ = self.agent(state, prev_reward, prev_action)
        action_probs = F.softmax(action_scores, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action


def finish_episode(optimizer, policy, config):
    """Updates model using REINFORCE.
    """
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + config.gamma * R
        returns.insert(0, R)
    returns = torch.stack(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_episodes", type=int, default=10_000)
    parser.add_argument("--num_repeat_action", type=int, default=4)
    parser.add_argument("--reward_threshold", type=int, default=1_000)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
    )
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="interval between training status logs (default: 10)",
    )
    parser.add_argument(
        "--save-model-interval",
        type=int,
        default=250,
        help="interval between saving models.",
    )
    config = parser.parse_args()

    envs = []
    torch.manual_seed(config.seed)
    for idx in range(config.batch_size):
        env = gym.make("Seaquest-v0")
        env.seed(config.seed + idx)
        envs.append(env)

    num_actions = env.action_space.n
    agent = attention.Agent(num_actions=num_actions)
    policy = Policy(agent=agent)
    if torch.cuda.is_available():
        policy.cuda()

    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    eps = np.finfo(np.float32).eps.item()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_reward = 10.0

    # NOTE: This is currently batched once for a single instance of the game.
    # I think the authors batch it across 32 trajectories of the same agent
    # across different instances of the game (trajectories). I also am using
    # a different update mechanism as of now (REINFORCE vs. A3C).

    current = time.time()
    for i_episode in range(config.num_episodes):
        if torch.cuda.is_available():
            print(torch.cuda.memory_allocated())
        observations = [
            env.reset() for env in envs
        ]
        def fill_list(x):
            """Fills list with placeholder observations.

            Using this is unfortunately quite wasteful -- if only 1 of the games is
            still playing, then the rest of the computation is just zero noise.
            """
            placeholder = np.zeros_like(observations[0])
            return x + [placeholder for _ in range(config.batch_size - len(x))]

        def fill_tensor_1D(x):
            """Fills placeholder tensor with real-values.

            This is used to keep `actions` and `rewards` at the same batchsize 
            across steps in an episode where the various games end at varying steps.

            Using this is unfortunately quite wasteful -- if only 1 of the games is
            still playing, then the rest of the computation is just zero noise.
            """
            placeholder = torch.zeros(config.batch_size).float().to(device)
            placeholder[0:len(x)] += x.float()
            return placeholder

        # resets hidden states, otherwise the comp. graph history spans episodes
        # and relies on freed buffers.
        agent.reset()
        ep_reward = torch.zeros(config.batch_size).to(device)
        reward = None
        action = None

        # Stash model in case of crash.
        if i_episode % config.save_model_interval == 0 and i_episode > 0:
            torch.save(agent.state_dict(), f"./models_batched/agent-{i_episode}.pt")

        done = set()
        for t in range(config.max_steps):

            # Buffer inputs because the internal states may have tensors that
            # are of the original batch_size even if some agents' episodes terminate.
            _reward = fill_tensor_1D(reward) if reward is not None else None
            _action = fill_tensor_1D(action) if action is not None else None
            _observations = fill_list(observations)

            # The agent policy operates on all batches at the same time.
            action = policy(_observations, prev_reward=_reward, prev_action=_action)

            # Step each env separately.
            reward = torch.zeros(config.batch_size).to(device)
            observations = []
            for idx in range(config.batch_size):
                if idx in done:
                    continue
                r = 0.
                for _ in range(config.num_repeat_action):
                    observation, _reward, _done, _ = envs[idx].step(action[idx])
                    r += _reward
                    if _done:
                        break
                reward[idx] = r
                observations.append(observation)
                if _done:
                    done.add(idx)
                    break
            policy.rewards.append(reward)
            ep_reward += reward
            if len(done) == config.batch_size:
                running_reward = 0.05 * ep_reward.mean() + (1 - 0.05) * running_reward
                finish_episode(optimizer, policy, config)
                time_elapsed, current = time.time() - current, time.time()
                if i_episode % config.log_interval == 0:
                    print(
                        "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\t Max steps: {:.2f}\t Time elapsed: {:.2f}".format(
                            i_episode, ep_reward.mean(), running_reward, t, time_elapsed
                        )
                    )
                if running_reward > config.reward_threshold:
                    print(
                        "Solved! Running reward is now {} and "
                        "the last episode runs to {} time steps!".format(
                            running_reward, t
                        )
                    )
                break
    torch.save(agent.state_dict(), f"./models_batched/agent-final.pt")
    for env in envs:
        env.close()
