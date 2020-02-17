import math
import random
import pygame
import MainState
import numpy as np
from keras.layers import Dense, Input, concatenate, LeakyReLU
from keras import Model, losses
from keras.optimizers import Adam
import tensorflow as tf


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class ActorCritic:
    def __init__(self):
        self.simulation = None
        self.d = False
        self.current_state = 0
        self.next_state = 0
        self.current_action = 0
        self.last_action = 0
        self.next_reward = 0
        self.td_error = 0
        self.average_reward = 0
        self.epsilon = 0
        self.level = 0
        self.fail = False

        self.actor_learning_rate = .0005
        self.critic_learning_rate = .001  # Alpha's
        self.reward_alpha = 0.001

        self.sess = tf.compat.v1.Session()  # Creates a new Tensorflow session

        self.actor_state, self.actor = self.create_actor()  # Creates the actor model

        self.actor_crit_grad = tf.placeholder(tf.float32, [None, 1])  # Placeholder for the gradient of the reward with respect to the action

        self.actor_weights = self.actor.trainable_weights  # Get actor weights

        self.actor_grad = tf.gradients(ys=self.actor.output, xs=self.actor_weights, grad_ys=-self.actor_crit_grad)  # Get gradient of its output with respect to its weights in relation to how that action impacts the reward given from the critic net.

        grads = zip([(tf.clip_by_norm(grad, 1)) for grad in self.actor_grad], self.actor_weights)
        self.optimize = tf.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(grads)

        self.critic_state, self.critic_action, self.critic = self.create_critic()  # Creates the critic model

        self.critic_action_grad = tf.gradients(self.critic.output, self.critic_action)  # Gets the gradient of the reward with respect to the action

        self.critic_weights = self.critic.trainable_weights  # Get critic weights

        self.actual = tf.placeholder(tf.float32, [1])
        self.loss = losses.mean_squared_error(self.actual, self.critic.output)
        self.critic_grad = tf.gradients(self.loss, self.critic_weights)

        self.actor_output = self.actor.output  # Get output of actor for getAction()
        self.critic_output = self.critic.output  # Get output of critic for TD Error

        self.eval_grad1 = tf.placeholder(tf.float32)
        # self.eval_grad2 = tf.placeholder(tf.float32)

        self.subtract = []
        for i in range(len(self.critic_weights)):
            self.subtract.append(tf.assign_sub(self.critic_weights[i], self.critic_learning_rate * self.eval_grad1 * self.td_error))

        self.sess.run(tf.compat.v1.initialize_all_variables())  # Initialize the session

    def create_actor(self):
        state_input = Input(shape=(194,))
        activation = LeakyReLU(alpha=0.05)(state_input)
        h1 = Dense(24)(activation)
        activation1 = LeakyReLU(alpha=0.05)(h1)
        h2 = Dense(48)(activation1)
        activation2 = LeakyReLU(alpha=0.05)(h2)
        h3 = Dense(24)(activation2)
        activation1 = LeakyReLU(alpha=0.05)(h3)
        action_output = Dense(units=1)(activation1)
        model = Model(input=state_input, output=action_output)
        adam = Adam(lr=0.001)
        model.compile(loss='mse', optimizer=adam)
        return state_input, model

    def create_critic(self):
        state_input = Input(shape=(194,))
        activation = LeakyReLU(alpha=0.05)(state_input)
        state_h1 = Dense(24)(activation)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=(1,))
        action_h1 = Dense(48)(action_input)

        merged = concatenate([state_h2, action_h1], axis=1)
        activation1 = LeakyReLU(alpha=0.05)(merged)
        merged_h1 = Dense(24)(activation1)
        activation2 = LeakyReLU(alpha=0.05)(merged_h1)
        reward_output = Dense(1)(activation2)

        model = Model(input=[state_input, action_input], output=reward_output)

        adam = Adam(lr=0.005)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def train_actor(self):
        grads = self.sess.run(self.critic_action_grad, feed_dict={self.critic_state: self.current_state, self.critic_action: self.current_action})[0]
        # Calculates how changes to the action impact the reward given by the critic

        self.sess.run(self.optimize, feed_dict={self.actor_state: self.current_state, self.actor_crit_grad: grads})
        # Takes a gradient ascent step (Maximize Reward)

    def train_critic(self):
        evaluated_gradients = self.sess.run(self.critic_grad, feed_dict={self.actual: self.next_reward, self.critic_state: self.current_state, self.critic_action: self.current_action})
        # Calculates that gradient with current state action pair

        for j in range(len(self.critic_weights)):  # Loop through each layer in the critic model
            self.sess.run(self.subtract[j], feed_dict={self.eval_grad1: evaluated_gradients[j]})  # Takes a gradient descent step (Minimize Loss)

    def getAction(self, state):
        if random.random() < self.epsilon:  # Epsilon Greedy
            if self.epsilon > 0.05:
                self.epsilon *= .9995
            action = random.uniform(-math.pi + .1, 0 - .1)
            action = np.array(action).astype('float32').reshape((-1, 1))
        else:
            action = self.sess.run(self.actor_output, feed_dict={self.actor_state: state})
        if abs(action) > 100000:
            self.fail = True
        return action

    def takeAction(self, direction):  # Interact with the simulation to produce reward and state
        state, end = self.simulation.SetTheta(direction, self.d)
        if self.simulation.level == 100:
            reward = 1000
            end = True
            print(self.simulation.level)
        elif end:
            self.level = self.simulation.level
            reward = - 1000 / self.simulation.level
        else:
            reward = self.simulation.blocks_destroyed
        reward = np.array(reward).astype('float32').reshape(1)
        return reward, state, end

    def draw(self):
        if self.d:
            self.d = False
        else:
            self.d = True

    def step(self):
        self.last_action = np.array(self.current_action).astype('float32').reshape(-1, 1)  # Store last action
        self.current_action = self.getAction(self.current_state)  # Get an action from the policy pi
        self.next_reward, self.next_state, end = self.takeAction(self.current_action)  # Store next state and reward and death from taking that action
        nextval = self.sess.run(self.critic_output, feed_dict={self.critic_state: self.next_state, self.critic_action: self.current_action})
        val = self.sess.run(self.critic_output, feed_dict={self.critic_state: self.current_state, self.critic_action: self.last_action})
        self.td_error = self.next_reward - self.average_reward + nextval - val
        self.average_reward = self.average_reward + self.reward_alpha * self.td_error
        self.average_reward = np.array(self.average_reward).astype('float32').reshape(1)
        self.train_actor()  # Train the actor
        self.train_critic()  # Train the critic
        self.current_state = self.next_state  # Reset current state
        return end

    def start(self):
        self.simulation = MainState.Run()
        self.current_state = np.reshape(self.simulation.state, (1, 192))
        self.current_state = np.append(self.simulation.state, .5)
        self.current_state = np.append(self.current_state, 1)
        self.current_state = self.current_state.astype('float32').reshape((-1, 194))

    def save(self):
        self.actor.save('actor_2.h5')
        np.savetxt('X_1.txt', X)
        np.savetxt('y_1.txt', y)
        self.critic.save('critic_2.h5')

    def load(self):
        self.actor = tf.keras.models.load_model('actor_1.h5')
        self.actor_weights = self.actor.trainable_weights
        self.critic = tf.keras.models.load_model('critic_1.h5')
        self.critic_weights = self.critic.trainable_weights
        self.start()


if __name__ == "__main__":
    AC = ActorCritic()
    AC.start()
    terminal = False
    running = True
    X = []
    y = []
    count = -1

    while running:
        count += 1
        if AC.fail:
            AC = ActorCritic()
            AC.start()
            terminal = False
            running = True
            X = []
            y = []
        if not terminal:
            terminal = AC.step()
        else:
            terminal = False
            X.append(len(X)+1)
            y.append(AC.level)
            AC.start()

        for event in pygame.event.get():  # Exit game
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    AC.draw()
                if event.key == pygame.K_RETURN:
                    AC.save()
                if event.key == pygame.K_TAB:
                    AC.load()
                if event.key == pygame.K_0:  # Graph
                    yvalues = []
                    sumy = 0
                    divider = int(len(X) / 200) + 1
                    for i in range(len(X)):
                        sumy += y[i]
                        if i % divider == 0:
                            yvalues.append(sumy / divider)
                            sumy = 0
                            if yvalues[0] < 9:
                                yvalues.remove(yvalues[0])
                    plt.plot(yvalues, 'o', color='black', markersize=2)
                    plt.show()

            if event.type == pygame.QUIT:
                running = False

