import argparse
import gym
import torch
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
    for _ in range(config.batch_size):
        env = gym.make("Seaquest-v0")
        torch.manual_seed(config.seed)
        env.seed(config.seed)
        envs.append(env)

    num_actions = env.action_space.n
    agent = attention.Agent(num_actions=num_actions)
    policy = Policy(agent=agent)
    if torch.cuda.is_available():
        policy.cuda()

    optimizer = optim.Adam(agent.parameters(), lr=1e-3)
    eps = np.finfo(np.float32).eps.item()

    running_reward = 10.0

    # NOTE: This is currently batched once for a single instance of the game.
    # I think the authors batch it across 32 trajectories of the same agent
    # across different instances of the game (trajectories). I also am using
    # a different update mechanism as of now (REINFORCE vs. A3C).

    for i_episode in range(config.num_episodes):
        observations = [
            env.reset() for env in envs
        ]
        # resets hidden states, otherwise the comp. graph history spans episodes
        # and relies on freed buffers.
        agent.reset()
        ep_reward = torch.zeros(config.batch_size)
        reward = None
        action = None

        # Stash model in case of crash.
        if i_episode % config.save_model_interval == 0 and i_episode > 0:
            torch.save(agent.state_dict(), f"./models/agent-{i_episode}.pt")

        for t in range(config.max_steps):
            action = policy(observations, prev_reward=reward, prev_action=action)
            reward = torch.zeros(config.batch_size)

            observations = []
            for idx in range(config.batch_size):
                r = 0
                for _ in range(config.num_repeat_action):
                    observation, _reward, done, _ = envs[idx].step(action[idx])
                    r += _reward
                if done:
                    break
                reward[idx] = r
                observations.append(observation)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                running_reward = 0.05 * ep_reward.mean() + (1 - 0.05) * running_reward
                finish_episode(optimizer, policy, config)
                if i_episode % config.log_interval == 0:
                    print(
                        "Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}".format(
                            i_episode, ep_reward.mean(), running_reward
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
    torch.save(agent.state_dict(), f"./models/agent-final.pt")
    for env in envs:
        env.close()
