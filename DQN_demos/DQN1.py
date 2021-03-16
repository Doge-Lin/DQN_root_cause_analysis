import random

import numpy as np
import torch.nn as nn
import torch
import gym


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
        )
        self.loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        return self.fc(inputs)


env = gym.envs.make('CartPole-v1')
env.unwrapped
net = MyNet()
net1 = MyNet()

gama = 0.9
store_count = 0
store_size = 2000
decline = 0.6
learn_time = 0
update_time = 20  # 学多少次更新一下目标网络
b_size = 1000  # 每次抽取1000条学习

store = np.zeros((store_size, 10))
start_study = False
for i in range(50000):
    s = env.reset()
    while True:
        if random.randint(0, 100)<100*(decline ** learn_time):
            a = random.randint(0, 1)
        else:
            out = net(torch.Tensor(s)).detach()
            a = torch.argmax(out).data.item()
        s_, r, done, info = env.step(a)

        r = (env.theta_threshold_radians - abs(s_[2]))/env.theta_threshold_radians * 0.7 + \
            (env.x_threshold - abs(s_[0])) / env.x_threshold * 0.3

        store[store_count % store_size][0:4] = s
        store[store_count % store_size][4:5] = a
        store[store_count % store_size][5:9] = s_
        store[store_count % store_size][9:10] = r

        store_count += 1
        s = s_

        if store_count> store_size:
            if learn_time % update_time == 0:
                net1.load_state_dict(net.state_dict())

            index = random.randint(0, store_size - b_size - 1)
            b_s = torch.Tensor(store[index:index+b_size, 0:4])
            b_a = torch.Tensor(store[index:index + b_size, 4:5]).long()
            b_s_ = torch.Tensor(store[index:index + b_size, 5:9])
            b_r = torch.Tensor(store[index:index + b_size, 9:10])

            q = net(b_s).gather(1, b_a)
            q_next = net1(b_s_).detach().max(1)[0].reshape(b_size, 1)
            tq = b_r + gama * q_next
            loss = net.loss(q, tq)
            net.opt.zero_grad()
            loss.backward()
            net.opt.step()

            learn_time += 1
            if not start_study:
                print('start')
                start_study = True
                break

        if done:
            break

        env.render()
