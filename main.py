import gym
from dqn import Agent
from utils import plotLearning
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,    # random action at beginning
                  input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 500    # number of games

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)    # choose action based on current state
            observation_, reward, done, info = env.step(action)   # get new observation
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_    # set current state to new state
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])    # average of previous 100 games

        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
    x = [i + 1 for i in range(n_games)]    # plot learning curve
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)
