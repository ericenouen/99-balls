# 99 Balls
An algorithm to beat the game 99 Balls using Actor Critic.
![alt text](Ballz.png)


# Making the Game
Before I could teach a computer to beat 99 balls, I first had to recreate it on my desktop in Python.
![alt text](99Balls.png)

This was mainly just brushing up on my python skills and working on animation and game design.

Decided how collisions would work was the most difficult part, but I decided on using the last position and current position to analyze the direction the ball was coming from in order to bounce it off the blocks correctly.
For example, if the x and previous x coordinates were both inside a block but the previous y coordinate wasn't, the ball would be hitting the bottom of the block.

This didn't account for hitting corners, where both previous coordinates would be outside of the block, so I implemented a test that would change the direction of any ball that hit a corner to travel directly away from that corner.

# Training the Model
Now that I had my game, it was time to try and beat it. 

Initially the model pretty much plays randomly according to its epsilon greedy policy.
[![Alt text](https://img.youtube.com/vi/cZT-lTS1rxA/0.jpg)](https://www.youtube.com/watch?v=cZT-lTS1rxA)

## Model
The model is made up of two parts, an actor and a critic.

The actor takes in the state, and passes that through three hidden layers of a neural network to output the optimal action to take.

The critic takes in both the state and the action, the state goes through two hidden layers before being concatenated with the action that goes through one hidden layers, after they are joined they go through one hidden layer before the neural network outputs the reward that should be given to that state-action pair.

It is the actor's job to decide what action will maximize the reward given to it by the critic, and it is the critics job to model the rewards of the real world as best as it can.

## State

The state is simply a flattened 8x8x3 representation of the coordinates of each block and orb on the screen, with the x-coordinate of the firing point and number of balls appended to the end.
## Weight Updates

Training the models required the use of gradient descent to minimize the loss of the critic network and gradient ascent to maximize the reward gained by the actor.

Every single step, the model would take an action pulled from the actor model and determine the state and reward that the action caused. This information is then used to determine how far off the estimated reward was from the actual reward received.

The actor is trained by taking a gradient ascent step in whatever direction maximizes the reward received from the critic.

The critic is trained by taking a gradient descent step in whatever direction minimizes the loss between the actual reward received and the output of the critic.
## Training

Initially I trained the model to take any number that was smaller than .1 radians from horizontal and change it into .1 radians, so the initial model quickly learned to decay all of the actions to be closer and closer to zero because always shooting close to horizontal is an optimal solution to this game.
![alt text](Figure_1.png)

The model starts to output smaller values
![alt text](Figure_3.png)
![alt text](Figure_4.png)

The model begins to win games

![alt text](Figure_5.png)

The model consistently reaches level 100 more often than it does not

[![Alt text](https://img.youtube.com/vi/4Bgjwa5AvoY/0.jpg)](https://www.youtube.com/watch?v=4Bgjwa5AvoY)

## Trying again

I then tried to limit this problem by expanding how low the balls could be fired as well as not firing at all if the output was too small.

![alt text](Training.png)

Unfortunately, this wasn't able to defeat the game, but it was able to improve a decent amount.

![alt text](Trained.png)

Here is the trained version, it clearly has learnt a policy that is able to break into the 50s, but it isn't an optimal one.

[![Alt text](https://img.youtube.com/vi/4X0AL9TKNoU/0.jpg)](https://www.youtube.com/watch?v=4X0AL9TKNoU)

# Learning Points
One thing I learned a ton about through this project was tensorflow graphs. I was forced to learn a lot about how tensorflow graphs are essentially processes that you can activate and use many different times. Initially I was recreating new tensorflow graphs every single time my code ran, so I had to put all of my graphs in the initialization of the actor critic network and just call the functions as I needed them to update the weights or get the output of a model.

Another huge thing was I got the chance to take a more hands on approach to my learning and actually implement something in order to solve a problem. I had taken University of Alberta's Reinforcement Learning Specialization on Coursera and this was a great opportunity to take everything I had learned about exploration versus exploitation, using two models in tandem in order to solve one problem, or implementing neural networks and play a game.

# Issues  

If I had more time to work on this project I would edit a couple of things that I thinkk would make the model perform better.

The first thing is that I would implement a different method such as tile coding or state aggregation in order to give the neural networks an easier time of deciphering which states are very similar so that learning could be much much faster. Currently, there are a ton of different state action pairs and it is incredibly hard for both the actor and the critic to learn a good policy.

The second thing I would try is implementing a different actor model, such as a gaussian method in order to better approximate the actions. With how many different states there are, another great point of generalization would be using a gaussian approximation method because it would help the actor reason that very similar actions result in very similar outputs, and that would be very helpful for learning a good policy.

I think that the biggest problem arises in the sheer quantity of states and actions, so anything to combat that would help the model perform much better.


### Overall, I learned a ton from this project and I am really glad I got the opportunity to work on it.
