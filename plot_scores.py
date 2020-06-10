import numpy as np
import matplotlib.pyplot as plt
import pickle
import gym
import seaborn as sns
sns.set_style('darkgrid')


def moving_avg(arr, window):
    """returns the moving average of the input array, based on the specified window"""
    return np.convolve(np.array(arr), np.ones((window,)) / window, mode='valid')


if __name__ == '__main__':
    # == Play with a random agent as a baseline and store results if not already available ==
    try:
        # == Read the random agent scores if available ==
        with open('./train/random_agent_scores', 'rb') as f:
            random_agent_scores = pickle.load(f)
    except Exception:
        env = gym.make('SpaceInvaders-v0')
        n_episodes = 100
        max_t = 10000

        np.random.seed(123)
        # store the score of each episode
        random_agent_scores = []

        # list containing the timestep per episode at which the game is over
        done_timesteps = []

        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            score = 0
            for timestep in range(max_t):
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)

                state = next_state
                score += reward
                if done:
                    print('\tEpisode {} done in {} timesteps.'.format(
                        i_episode, timestep))
                    done_timesteps.append(timestep)
                    break

                random_agent_scores.append(score)  # save the most recent score

        # save random agent scores
        with open('./train/random_agent_scores', 'wb') as f:
            pickle.dump(random_agent_scores, f)
    finally:
        # == Read the DQN agent scores ==
        with open('./train/scores', 'rb') as f:
            dqn_agent_scores = pickle.load(f)

        window = 6000
        plt.plot(moving_avg(dqn_agent_scores, window), 'g-', label="DQN")
        plt.xlabel('Timesteps')
        plt.plot(moving_avg(random_agent_scores, window), 'y-', label="random")
        plt.ylabel('Score')
        plt.legend(loc="best")
        plt.savefig('./assets/scores.jpg')
