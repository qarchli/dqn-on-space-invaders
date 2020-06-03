import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import deque
import gym
from gym import wrappers
import torch
from agent import DQNAgent


def train(n_episodes=100,
          max_t=10000,
          eps_start=1.0,
          eps_end=0.01,
          eps_decay=0.996):
    """
    Training a Deep Q-Learning agent to play Space Invaders
    ---
    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon
    Returns
    ======
        scores: list of the scores obtained in each episode
    """
    # to store the score of each episode
    scores = []

    # list containing the timestep per episode at which the game is over
    done_timesteps = []

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for timestep in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # above step decides whether we will train(learn) the network
            # actor (local_qnetwork) or we will fill the replay buffer
            # if len replay buffer is equal to the batch size then we will
            # train the network or otherwise we will add experience tuple in our
            # replay buffer.
            state = next_state
            score += reward
            if done:
                print('\tEpisode {} done in {} timesteps.'.format(
                    i_episode, timestep))
                done_timesteps.append(timestep)
                break
            scores_window.append(score)  # save the most recent score
            scores.append(score)  # save the most recent score
            eps = max(eps * eps_decay, eps_end)  # decrease the epsilon

            if timestep % SAVE_EVERY == 0:
                print('\rEpisode {}\tTimestep {}\tAverage Score {:.2f}'.format(
                    i_episode, timestep, np.mean(scores_window)), end="")

                # save the final network
                torch.save(agent.qnetwork_local.state_dict(),
                           SAVE_DIR + 'model.pth')

                # save the final scores
                with open(SAVE_DIR + 'scores', 'wb') as fp:
                    pickle.dump(scores, fp)

                # save the done timesteps
                with open(SAVE_DIR + 'dones', 'wb') as fp:
                    pickle.dump(done_timesteps, fp)

    # save the final network
    torch.save(agent.qnetwork_local.state_dict(), SAVE_DIR + 'model.pth')

    # save the final scores
    with open(SAVE_DIR + 'scores', 'wb') as fp:
        pickle.dump(scores, fp)

    # save the done timesteps
    with open(SAVE_DIR + 'dones', 'wb') as fp:
        pickle.dump(done_timesteps, fp)

    return scores


def test(env, trained_agent, n_games=5, n_steps_per_game=10000):
    for game in range(n_games):
        env = wrappers.Monitor(env,
                               "./test/game-{}".format(game),
                               force=True)

        observation = env.reset()
        score = 0
        for step in range(n_steps_per_game):
            action = trained_agent.act(observation)
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                print('GAME-{} OVER! score={}'.format(game, score))
                break
        env.close()


# Agent was trained on GPU in colab.
# The files presented in train folder are those colab
# TODO
# - Encapsulate the training data into a trainloader to avoid GPU runtime error

if __name__ == '__main__':
    TRAIN = True  # train or test
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 100  # how often to update the target network
    SAVE_EVERY = 100  # how often to save the network to disk
    MAX_TIMESTEPS = 10
    N_EPISODES = 10
    SAVE_DIR = "./train/"

    env = gym.make('SpaceInvaders-v0')

    if TRAIN:
        # init agent
        agent = DQNAgent(state_size=4,
                         action_size=env.action_space.n,
                         seed=0,
                         lr=LR,
                         gamma=GAMMA,
                         tau=TAU,
                         buffer_size=BUFFER_SIZE,
                         batch_size=BATCH_SIZE,
                         update_every=UPDATE_EVERY)
        # train and get the scores
        scores = train(n_episodes=N_EPISODES, max_t=MAX_TIMESTEPS)

        # plot the running mean of scores
        N = 100  # running mean window
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(
            np.convolve(np.array(scores), np.ones((N, )) / N, mode='valid'))
        plt.ylabel('Score')
        plt.xlabel('Timestep #')
        plt.show()
    else:
        N_GAMES = 5
        N_STEPS_PER_GAME = 10000

        # init a new agent
        trained_agent = DQNAgent(state_size=4,
                                 action_size=env.action_space.n,
                                 seed=0)

        # replace the weights with the trained weights
        trained_agent.qnetwork_local.load_state_dict(
            torch.load(SAVE_DIR + 'model.pth'))

        # enable inference mode
        trained_agent.qnetwork_local.eval()

        # test and save results to disk
        test(env, trained_agent)
