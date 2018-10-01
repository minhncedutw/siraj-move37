import numpy as np
import matplotlib.pyplot as plt
import gym


def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    counter = 0

    for _ in range(200):
        action = 0 if np.matmul(a=parameters, b=observation) < 0 else 1
        observation, reward, done, infor = env.step(action)
        totalreward += reward
        counter += 1
        if done:
            break

    return totalreward


def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env.monitor.start('cartpole-hill/', force=True)

    episodes_per_update = 5
    noise_scaling = 0.1
    parameters = np.random.rand(4) * 2 - 1
    bestreward = 0
    counter = 0

    for _ in range(2000):
        counter += 1
        newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling

        reward = run_episode(env=env, parameters=parameters)

        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward == 200:
                break

    if submit:
        for _ in range(100):
            run_episode(env=env, parameters=parameters)
        env.monitor.close()

    return counter

r =  train(submit=False)
print(r)