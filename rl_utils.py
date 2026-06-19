from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer:
    def __init__(self, capacity):
        # 双边队列，当向队列中添加元素时，如果数量超过 capacity，最左边（最老）的元素会自动被踢出
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        # 将一条经验打包成元组，存入池子
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        # 从池子中随机抽取 batch_size 个样本
        transitions = random.sample(self.buffer, batch_size)
         # 下面是将抽出的 batch 数据解包，重组成列的形式，方便喂给神经网络
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0 # 将本回合的累计奖励清零

                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                
                state, _ = env.reset() # 重置环境，获取初始状态 s
                done = False # 标记本回合是否结束
                
                while not done:
                    # 智能体根据当前状态选择动作
                    action = agent.take_action(state)
                    # 执行动作，环境返回下一步的状态、奖励、以及是否结束
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # 将这一步的经验塞进本局的临时字典中
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    # 更新当前状态，累加奖励
                    state = next_state
                    episode_return += reward

                return_list.append(episode_return)
                # 把这一整局的轨迹数据（Trajectory）一次性喂给智能体进行网络更新
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0  # 将本回合的累计奖励清零

                state, _ = env.reset() # 重置环境，获取初始状态 s
                done = False # 标记本回合是否结束

                while not done:
                    # 智能体根据当前状态选择动作
                    action = agent.take_action(state)
                    # 执行动作，环境返回下一步的状态、奖励、以及是否结束
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    
                    # 将这一步的经验存入replay_buffer中
                    replay_buffer.add(state, action, reward, next_state, done)
                    
                    # 更新当前状态，累加奖励
                    state = next_state
                    episode_return += reward

                    # 预热机制：如果回放池里的数据太少，就先不训练，只收集数据
                    if replay_buffer.size() > minimal_size:
                        # 从经验回放池中随机抽取一个批次（Batch）的数据
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        # 打包成字典
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                         # 把这批随机数据喂给智能体，进行梯度下降，更新神经网络
                        agent.update(transition_dict)

                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


# 广义优势估计（GAE）的递推实现
# advantage_t = delta_t + gamma * lambda * advantage_{t+1}
def compute_advantage(gamma, lmbda, td_errors):
    td_errors = td_errors.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_errors[::-1]: # 从后往前计算优势函数，符合时间顺序
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


