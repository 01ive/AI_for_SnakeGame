# How does AI play the classical Snake Game ?

This project is just for fun. Generative AI has been realised using generative AI but I can guess that it learns from the below ressources.

So have a look to this original ressources:

[Nice article](https://medium.com/@nancy.q.zhou/teaching-an-ai-to-play-the-snake-game-using-reinforcement-learning-6d2a6e8f3b1c)

[Great Youtube video](https://www.youtube.com/watch?v=L8ypSXwyBds)

[Related GitHub](https://github.com/patrickloeber/snake-ai-pytorch)

## Setup your environment

First of all create a Python virtual env and activate it using *requirements.txt* file.

On my environment it looks like this:

```bash
python -m venv venv
source venv/script/activate
python -m pip install -r requirements.txt
```

## Play Snake Game

To just play Snake Game run

```bash
python snake.py
```

Use your arrow keys to move the snake.

## Train AI to play Snake Game

AI can learn or just play Snake Game. Use command line options to select train or inference (play) mode. You can also specify to play or train several games.

Exemple to train AI for 100 games
```bash
python main.py --train --nb 100
```

Other options are:

* **train** (optional): to select train mode
* **inference** (optional) (default): to select inference mode
* **nb** (optional) (default=1): to specify how many games to play or train
* **weight** (optional) (default=*model_weights.pth*): to specify the AI memory file you want to use

You can get help using:

```bash
python main.py --help
```

## Let AI play Snake Game

To only play games, you should have trained your AI before.

```bash
python snake.py
```

## Make the best AI for Snake Game

If you want to search for the best AI to play Snake Game, you can use *search_arch.py* scripts.

```bash
python search_arch.py
```

It will take a while, but it allows to change various parameters

### Changing rewards

You can change your Deep QLearning rewards by setting **rewards** attribute of your game object.

```python
game = SnakeApi()
game.rewards = {
                    'collision': -10,
                    'self collision': -10,
                    'food': 10,
                    'closer_to_food': 5
                }
agent = Agent(game)
agent.train()
```

*Note: self collision attribute is not managed at this time*

### Selecting your Qnet architecture

Several neural Network architectures are available:

* LinearQNet (default): 2 linears layers
* DeepQNet: 3 linear layers

You can also specify internal layers size using hidden_layer parameter of Agent constructor.

```python
game = SnakeApi()
Agent(game, hidden_layer_size=128, network_type=LinearQNet)
agent.train()
```

Copyright (c) 2025 Geremm & 01ive