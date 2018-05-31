import argparse
import numpy as np
import matplotlib.pyplot as plt
import bandit, environment


# settings
NUM_ACTIONS = 10
INIT_MEAN = 0
INIT_STD = 1
NOISE_MEAN = 0
NOISE_STD = 1

NUM_TRIALS = 2000
NUM_STEPS = 1000

AGENT_EPSILON = "epsilon"
AGENT_SOFTMAX = "softmax"

def main(args):

  # validate parameters
  if len(args.agents) == 0:
    print("Error: Specify at least one agent.")
    exit(1)

  if len(args.agents) != len(args.settings):
    print("Error: Provide setting for each agent.")
    exit(1)

  if args.inits is not None and len(args.inits) != len(args.agents):
    print("Error: If you want to set initialization, provide values for all agents.")
    exit(1)

  if args.labels is not None and len(args.labels) != len(args.labels):
    print("Error: If you want to set labels, provide one for each agent.")
    exit(1)

  # default setting
  inits = args.inits
  if inits is None:
    inits = [0] * len(args.agents)

  labels = args.labels
  if labels is None:
    labels = [None] * len(args.agents)

  # setup algorithms and arrays to hold results
  env = environment.StationaryEnvironment(NUM_ACTIONS, INIT_MEAN, INIT_STD, NOISE_MEAN, NOISE_STD)
  agents = {}
  for agent_type, setting, init, label in zip(args.agents, args.settings, inits, labels):
    if agent_type == AGENT_EPSILON:
      if label is None:
        label = "e-greedy (epsilon {:.2f}".format(setting)
      agents[label] = bandit.EpsilonGreedyBandit(env, setting, init=init)
    elif agent_type == AGENT_SOFTMAX:
      if label is None:
        label = "softmax (temperature: {:.2f}".format(setting)
      agents[label] = bandit.SoftmaxBandit(env, setting, init=init)
    else:
      print("Invalid agent type: {:s}.".format(agent_type))
      exit(1)

  rewards = {key: np.empty((NUM_TRIALS, NUM_STEPS), dtype=np.float32) for key in agents.keys()}
  optimal_actions = {key: np.empty((NUM_TRIALS, NUM_STEPS), dtype=np.bool) for key in agents.keys()}

  # run experiment
  for i in range(NUM_TRIALS):

    for j in range(NUM_STEPS):

      for agent in agents.values():
        agent.act()

    optimal_action = np.argmax(env.action_values)

    for key in agents.keys():
      rewards[key][i] = agents[key].rewards
      optimal_actions[key][i] = np.array(agents[key].actions) == optimal_action
      agents[key].reset()

    env.reset()

  # average rewards and optimal actions
  for key in agents.keys():
    rewards[key] = np.mean(rewards[key], axis=0)
    optimal_actions[key] = optimal_actions[key].astype(np.float32)
    optimal_actions[key] = np.mean(optimal_actions[key], axis=0) * 100

  # plot average rewards
  for i, key in enumerate(sorted(rewards.keys())):
    plt.plot(rewards[key], label=key)

  plt.xticks([0, 250, 500, 750, 1000])
  plt.yticks([0.0, 0.5, 1.0, 1.5])
  plt.xlabel("Steps")
  plt.ylabel("Average reward")

  if args.title is not None:
    plt.title(args.title)

  plt.legend()

  plt.savefig("{:s}_rewards.{:s}".format(args.save_path, args.format))
  plt.show()

  # plot optimal actions
  for key in sorted(optimal_actions.keys()):
    plt.plot(optimal_actions[key], label=key)

  plt.xticks([0, 250, 500, 750, 1000])
  plt.yticks([0, 20, 40, 60, 80, 100])
  plt.xlabel("Steps")
  plt.ylabel("Optimal action (%)")

  if args.title is not None:
    plt.title(args.title)

  plt.legend()

  plt.savefig("{:s}_actions.{:s}".format(args.save_path, args.format))
  plt.show()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  parser.add_argument("save_path", help="save path for all figures")

  parser.add_argument("-a", "--agents", nargs="+", help="{:s} or {:s}".format(AGENT_EPSILON, AGENT_SOFTMAX))
  parser.add_argument("-s", "--settings", nargs="+", type=float,
                      help="epsilon for epsilon greedy or temperature for softmax")
  parser.add_argument("-i", "--inits", nargs="+", type=float,
                      help="initial action values (used for optimistic initialization); zero by default")
  parser.add_argument("-l", "--labels", nargs="+", help="custom labels for each agent")
  parser.add_argument("-t", "--title", help="title for the plots")
  parser.add_argument("-f", "--format", help="figure format", default="svg")

  parsed = parser.parse_args()
  main(parsed)