from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
from collections import deque
import random

from snake import SnakeGame

EPISODES = 500000
MEMORY_SIZE = 50000


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = 0.99
        self.epsilon = 0.0001
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.999
        self.learning_rate = 1e-4
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.loss = 0

    def _build_model(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def preprocessing(self, image):
        if np.shape(image)[0] < 2:
            images = [image, image]
        else:
            images = image
        images = np.reshape(images, [1, self.state_size])

        return images

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): #epsilon greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory) 

        batch = random.sample(self.memory, int(batch_size))

        random.shuffle(batch)

        X = np.zeros((batch_size, 128))
        Y = np.zeros((batch_size, 4))

        for i in range(batch_size):
            s_t, a_t, r_t, s_tp1, game_over = batch[i]
            X[i] = s_t
            Y[i] = self.model.predict(s_t)[0]
            Q_sa = np.max(self.target_model.predict(s_tp1)[0])
            if game_over:
                Y[i, a_t] = r_t
            else:
                Y[i, a_t] = r_t + self.gamma * Q_sa

        self.loss += self.model.train_on_batch(X, Y)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, e):
        self.model.save('saves/model_ddqn{}.h5'.format(str(e)))


if __name__ == "__main__":
    log = open('logs/old_ddqn_final.txt', 'w')

    env = SnakeGame()
    state_size = 128
    action_size = 4
    agent = Agent(state_size, action_size)
    agent.model.summary()

    done = False
    batch_size = 32

    for e in range(EPISODES): #epsisodes = 500000
        step = 0
        env.reset()
        state, reward, done, score = env.step(0)
        state = agent.preprocessing(state)
        while not done:         #
            step += 1
            action = agent.act(state)
            next_state, reward, done, score = env.step(action)
            next_state = agent.preprocessing(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            agent.replay(batch_size)
            
            if done:
                if e % 5000 == 0:
                    agent.save_model(e)

                agent.update_target_model()

                print("episode: {}/{}, e: {:.2}, score: {}"
                      .format(e, EPISODES, agent.epsilon, score))
                log.write('{}, {:.5}, {}\n'
                          .format(e, agent.epsilon, score))
                log.flush()
                break
                    
    log.close()