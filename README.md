# Bandit Environments

Series of n-armed bandit environments for the OpenAI Gym

## Environments
* `BanditTwoArmedDeterministicFixed-v0`: Simplest case where one bandit always pays, and the other always doesn't
* `BanditTwoArmedHighLowFixed-v0`: Stochastic version with a large difference between which bandit pays out of two choices
* `BanditTwoArmedHighHighFixed-v0`: Stochastic version with a small difference between which bandit pays where both are good
* `BanditTwoArmedLowLowFixed-v0`: Stochastic version with a small difference between which bandit pays where both are bad
* `BanditTwoArmedUniform-v0`: Stochastic version both arms pay between 0 and 1
* `BanditTenArmedRandomFixed-v0`: 10 armed bandit with random probabilities assigned to payouts
* `BanditTenArmedRandomRandom-v0`: 10 armed bandit with random probabilities assigned to both payouts and rewards
* `BanditTenArmedUniformDistributedReward-v0`: 10 armed bandit with that always pays out with a reward selected from a uniform distribution
* `BanditTenArmedGaussian-v0`: 10 armed bandit mentioned on page 30 of [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (Sutton and Barto)

## Installation
```bash
git clone git@github.com:mimoralea/gym-bandits.git
cd gym-bandits
pip install .
```

or:

```bash
pip install git+https://github.com/mimoralea/gym-bandits#egg=gym-bandits
```


In your gym environment
```python
import gym, gym_bandits
env = gym.make("BanditTenArmedGaussian-v0")
```
