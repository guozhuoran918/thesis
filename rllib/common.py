import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.font_manager import FontProperties
def plot_rewards(rewards,tag="train",env='Milestone1',algo = "Q-learning",save=True,path='./'):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo,env))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='rewards')
    plt.legend()
    if save:
        plt.savefig(path+"{}_rewards_curve".format(tag))
    plt.show()

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True,exist_ok=True)


def plot_loss(loss,tag="train",env="milestone1",algo = "DQN"):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo,env))
    plt.xlabel('epsiodes')
    plt.plot(loss,label='loss')
    plt.legend()


def plot_rewards(rewards,tag="train",env="milestone1",algo = "DQN"):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo,env))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='reward')
    plt.legend()
