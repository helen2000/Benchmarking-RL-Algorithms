# Creating an Agent

## How
Creating an agent that's recognised by the program is very easy...

1) Create a new .py file under the agents package. Call it whatever you like.
2) Have it implement the AbstractAgent / CheckpointAgent class in abstract_agent.
3) Inherit the methods contained, and call the super constructor, passing action_space into your agent.

So far, it should look like this (for AbstractAgent):

```python
from rlcw.agents.abstract_agent import AbstractAgent

class YourAgent(AbstractAgent):

    def __init__(self, logger, config):
        super.__init__(logger, config)
        
    def assign_env_dependent_variables(self, action_space, state_space):
        pass
        
    def name(self):
        return "YourAgent"
        
    def get_action(self, observation):
        pass
        
    def train(self, training_context):
        pass
```

### Notes
The config passed in here is **not** the global config - it's just the section for this particular agent. So taking
the "random" agent as an example, cfg will look something like this (refer to config.yml for verification):

```python
cfg = {
   "foo": "bar"
}
```

Please feel free to add whatever config options you like - it won't (shouldn't) break anything.

The Logger is there because it avoids some really weird initialisation order stuff - idk what it is but it fixes the problem
and honestly who cares at this point. It is a working logger though, so please feel free to use it.

### Back to Explanation


Here, you can implement those get_action and train methods, based on the information said later on.

4) Go to main.py
5) Import your agent
6) Find the method called `get_agent(name, action_space)`
7) Add the following to it:

```python 
elif name.lower() == "<your_agent>": 
   return YourAgent(logger, cfg)
```

Bingo bango bongo we're done! 

When starting the application, it looks to the config variable "agent_name" in the config.yml to find the agent to run.
If you want to run your new agent, change the name to whatever you put in (7) to that. 

## Abstract Agent

Abstract Agent is an abstract class to give a template for each new agent that we create - so that our Runner can 
recognise each new agent we create.

### Methods and Constructor Defined

There's only **four** methods defined that will actually NEED to be implemented:
1) `assign_env_dependent_variables(self, action_space, state_space)`
2) `name()`
3) `get_action(observation)`
4) `train(training_context)`

#### assign_env_dependent_variables(action_space, state_space)

`assign_env_dependent_variables` is called **after** the initialisation of the environment. 

To accommodate being able to swap between the discrete and continuous action spaces for Lunar Lander, we need to call the constructor for the agent **before** setting up the environment.

Obviously, there are differences between the variables in the continuous and discrete action space, and therefore a second method to initialise those is called **after** setting up the environment (this one!)

In other words, any variables needed for this agent that rely on the environment should be passed here.

There is already a method `update_action_and_state_spaces` that is called before this one, so you don't need to assign those. Everything else, however, needs to be done.

#### name()

`name` is pretty simple - just return the name of the agent here as a string.

#### get_action(observation)

`get_action` is being called at the beginning of the timestep, and will determine what action it takes (well duh.) 

The observation is the thing defined in gym. If you read gym's documentation on it, you'll get more information.

The action space is already defined in the constructor, so you're able to call `self.action_space` to get stuff there.

#### train(training_context)

`train` is being called after the current timestep is greater than the start_training timesteps defined in the 
config.yml.

`training_context` is defined as a list of dict objects, where each dict contains information about each timestep, 
where the index of the list represents that timestep. 

This dict object looks like this:

```python
training_context_item = {
   "curr_state": state,
   "next_state": next_state,
   "reward": reward,
   "action": action,
   "terminated": terminated
}
```
`training_context` is cyclic and bounded - i.e. once it reaches its max capacity (specificed under context_capacity in
the config.yml, it will start replacing old items with the newer ones.)

For example: capacity = 5, training_context already contains [0, 1, 2, 3, 4]. Calling training_context.add(5) will 
result in [1, 2, 3, 4, 5], appending 5 to the end and removing the oldest value (0).

### CheckpointAgent

CheckpointAgent is very similar to AbstractAgent, although it requires you to implement two additional methods:

#### save()

This is called whenever an episode ends that falls into the "every" category specified in the config.yml.

There is a utility method specified in CheckpointAgent called `save_checkpoint(net, file_name)`, which is really all that should be called here.

To better illustrate this, DQN's `save()` method is shown below (keep in mind that it uses two networks: a Value Network and a Target Value Network)

```python
    def save(self):
        self.save_checkpoint(self._q, "ValueNetwork")
        self.save_checkpoint(self._target_q, "TargetValueNetwork")
```
You don't need to specify a file extension - by default it uses a .pth extension. However, you can specify one if you wish by adding it to the end of the name.

#### load(path)

This is a very similar story to save, but for loading.

There is also a utility method specified called `load_checkpoint(net, path, file_name)`, which does what it says on the tin.

Again, better shown with an example:

```python
    def load(self, path):
        self.load_checkpoint(self._q, path, "ValueNetwork")
        self.load_checkpoint(self._target_q, path, "TargetValueNetwork")
```
Make sure that each network uses the same name - otherwise you'll run into problems later!

### requires_continuous_action_space

By default, each agent defaults to using a discrete action space. However, if you want to use a continuous one, all you need to do is add the following line to the constructor of your agent:

```python
    self.requires_continuous_action_space = True
```

