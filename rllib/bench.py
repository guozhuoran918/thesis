from itertools import count
from . import configuration as config
import numpy as np
class MedBench:
    def __init__(self, agent, num_episodes=50):
        self.agent = agent
        self.rewardList = []
        self.episode_duration = []
        self.episode_loss = []
        self.success = 0
        self.sucessrate = []
        self.num_episodes = num_episodes

    def run_trial(self, debug=True):
        for idx in range(self.num_episodes):
            # if (idx+1)%self.agent.env.epoch == 0:
            #     # print(idx+1)
            #     self.sucessrate.append((self.success/self.num_episodes)*self.agent.env.epoch)
            #     self.success = 0
            self.agent.reset_env()
            rewards = 0
            for jdx in count():
                if debug and jdx % 10 == 0:
                    print("Step %d in Episode: %d" % (jdx, idx))
                reward, done = self.agent.step()
                # optimize
                loss = self.agent.update()
                self.episode_loss.append(loss)
                rewards += reward
                if done == config.DIALOGUE_STATUS_SUCCESS:
                    num_runs = jdx + 1
                    if debug:
                        print("Completed Episode: %d in %d steps. Final Reward: %d" %(idx, num_runs, reward))
                    self.episode_duration.append(num_runs)
                    self.rewardList.append(rewards)
                    self.success = self.success+1
                    break
                if done == config.DIALOGUE_STATUS_FAILED or jdx+1 > config.MAX_TURN:
                    num_runs = jdx+1
                    self.rewardList.append(rewards)
                    self.episode_duration.append(num_runs)
                    break
            if (jdx+1) % self.agent.target_update == 0:
                self.agent.double_q_update()


    def get_success_rate(self):
        return self.success/self.num_episodes
    
    def get_average_rewards(self):
        return np.mean(self.rewardList)

    def get_average_turn(self):
        return np.mean(self.episode_duration)

    def get_loss(self):
        return self.episode_loss


class MedBenchEval:
    def __init__(self, agent, num_episodes=50):
        self.agent = agent
        self.rewardList = []
        self.episode_duration = []
        # self.episode_loss = []
        self.success = 0
        self.num_episodes = num_episodes
        self.top_3 = 0
        self.top_5 = 0
        self.hint = np.zeros(self.num_episodes,dtype=int)

    def run_trial(self, debug=True):
        for idx in range(self.num_episodes):
            self.agent.reset_env()
            rewards = 0
            for jdx in count():
                if debug and jdx % 10 == 0:
                    print("Step %d in Episode: %d" % (jdx, idx))
                reward, done = self.agent.step()
                if debug:
                    print(reward,done,jdx+1)
                # optimize
                # loss = self.agent.update()
                # self.episode_loss.append(loss)
                rewards += reward
                if done == config.DIALOGUE_STATUS_INFORM_RIGHT_SYMPTOM:
                    self.hint[idx] = 1
                if done == config.DIALOGUE_STATUS_SUCCESS:
                    num_runs = jdx + 1
                    if debug:
                        print("Completed Episode: %d in %d steps. Final Reward: %d" %(idx, num_runs, reward))
                    self.episode_duration.append(num_runs)
                    self.rewardList.append(rewards)
                    self.success = self.success+1
                    break
                if done == config.DIALOGUE_STATUS_FAILED or jdx+1 > config.MAX_TURN:
                    num_runs = jdx+1
                    self.rewardList.append(rewards)
                    self.episode_duration.append(num_runs)
                    break
    def get_success_rate(self):
        return self.success/self.num_episodes
    
    def get_average_rewards(self):
        return np.mean(self.rewardList)

    def get_average_turn(self):
        return np.mean(self.episode_duration)

    def get_hint_rate(self):
        return np.mean(self.hint)


    

class MedQBench:
    def __init__(self,agent,num_episodes = 100,num_epoches = 10):
        self.agent = agent
        self.epoch = num_epoches
        self.success = 0
        self.rewardList = []
        # self.actionrewards = [] # inquiry average rewards
        self.episode_duration = []
        self.num_episodes = num_episodes
        self.steps = []
    def run_trial(self,debug = True):
        # for epoch in range(self.epoch):
            for idx in range(self.num_episodes*self.epoch):
                self.agent.reset_env()
                rewards = 0
                for jdx in count():
                    if debug and jdx % 100 == 0:
                        print("Step %d in Episode: %d" % (jdx, idx))
                    _,reward, done =self.agent.step()
                    rewards += reward
                    if jdx+1 > config.MAX_TURN or done == config.DIALOGUE_STATUS_FAILED:
                        num_runs = jdx + 1
                        self.rewardList.append(rewards)
                        self.steps.append(num_runs)
                        break   

                    if done == config.DIALOGUE_STATUS_SUCCESS:
                        num_runs = jdx + 1
                        self.success = self.success+1
                        self.rewardList.append(rewards)
                        self.steps.append(num_runs)
                        break
    def get_success_rate(self):
        return self.success/(self.epoch*self.num_episodes)
    
    def get_average_rewards(self):
        return np.mean(self.rewardList)

    def get_average_turn(self):
        return np.mean(self.steps)
            

def eval(env,agent,num_episodes = 100):
    rewards = []  # 记录所有episode的reward
    success = 0 # 滑动平均的reward
    turn = []
    for i_ep in range(num_episodes):
        ep_reward = 0  # 记录每个episode的reward
        env.reset()
        state = env.state  
        for jdx in count():
            action = agent.predict(state)  # 根据算法选择一个动作
            
            next_state, reward, done = env.take_action(action)  # 与环境进行一个交互
            state = next_state  # 存储上一个观察值
            ep_reward += reward
            if jdx+1 > config.MAX_TURN or done == config.DIALOGUE_STATUS_FAILED:
                num_runs = jdx + 1
                rewards.append(ep_reward)
                turn.append(num_runs)
                break   
            if done==config.DIALOGUE_STATUS_SUCCESS:
                success +=1
                num_runs = jdx + 1
                rewards.append(ep_reward)
                turn.append(num_runs)
                break
    return rewards,success,turn