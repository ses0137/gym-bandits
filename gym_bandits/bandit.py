import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register

class BanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    """
    metadata = {"render_modes": []}

    def __init__(self, p_dist, r_dist):
        super().__init__()

        if len(p_dist) != len(r_dist):
            raise ValueError("Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist
        self.n_bandits = len(p_dist)

        # Action과 Observation 공간 정의
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.Discrete(1)  # 상태는 고정 (bandit에서는 의미 거의 없음)

        # 내부 난수 생성기
        self._np_random = None

    def seed(self, seed=None):
        """
        Gymnasium 0.26+에서는 reset(seed=...) 호출 권장
        그러나 사용자 코드에서 env.seed(seed) 직접 호출 시
        내부 난수 생성기를 다시 세팅할 수 있도록 구현
        """
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        """
        Gymnasium 스타일로 (observation, info) 반환
        """
        super().reset(seed=seed)
        if self._np_random is None:
            # 아직 seed()가 한 번도 안 불렸다면 여기서 세팅
            self._np_random, _ = seeding.np_random(seed)
        observation = 0
        info = {}
        return observation, info

    def step(self, action):
        """
        Gymnasium 0.26+ : (obs, reward, terminated, truncated, info) 5개 반환
        """
        assert self.action_space.contains(action)

        # bandit은 한 번 액션 후 episode가 끝난다고 가정
        terminated = True
        truncated = False

        reward = 0
        # p_dist[action] 확률로 보상 발생
        if self._np_random.uniform() < self.p_dist[action]:
            # 보상이 상수인지, (mu, sigma)인지 체크
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = self._np_random.normal(self.r_dist[action][0], self.r_dist[action][1])

        # 다음 관측 obs도 그냥 0으로 고정
        return 0, reward, terminated, truncated, {}

    def render(self):
        pass


class BanditTwoArmedDeterministicFixed(BanditEnv):
    """Simplest case where one bandit always pays, and the other always doesn't"""
    def __init__(self):
        super().__init__(p_dist=[1, 0], r_dist=[1, 1])


class BanditTwoArmedHighLowFixed(BanditEnv):
    """Stochastic version with a large difference between which bandit pays out of two choices"""
    def __init__(self):
        super().__init__(p_dist=[0.8, 0.2], r_dist=[1, 1])


class BanditTwoArmedHighHighFixed(BanditEnv):
    """Stochastic version with a small difference between which bandit pays where both are good"""
    def __init__(self):
        super().__init__(p_dist=[0.8, 0.9], r_dist=[1, 1])


class BanditTwoArmedLowLowFixed(BanditEnv):
    """Stochastic version with a small difference between which bandit pays where both are bad"""
    def __init__(self):
        super().__init__(p_dist=[0.1, 0.2], r_dist=[1, 1])


class BanditTwoArmedUniform(BanditEnv):
    """
    Stochastic version with rewards of one and random probabilities assigned to both payouts
    """
    def __init__(self, bandits=2, seed=1):
        # seed을 통한 난수 생성
        np_random, _ = seeding.np_random(seed)
        p_dist = np_random.uniform(size=bandits)
        r_dist = np.full(bandits, 1)
        super().__init__(p_dist=p_dist, r_dist=r_dist)
        self._init_seed = seed  # 필요하면 저장


class BanditTenArmedRandomFixed(BanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""
    def __init__(self, bandits=10, seed=1):
        np_random, _ = seeding.np_random(seed)
        p_dist = np_random.uniform(size=bandits)
        r_dist = np.full(bandits, 1)
        super().__init__(p_dist=p_dist, r_dist=r_dist)
        self._init_seed = seed


class BanditTenArmedUniformDistributedReward(BanditEnv):
    """10 armed bandit that always pays out with a reward selected from a uniform distribution"""
    def __init__(self, bandits=10, seed=1):
        np_random, _ = seeding.np_random(seed)
        p_dist = np.full(bandits, 1)
        r_dist = np_random.uniform(size=bandits)
        super().__init__(p_dist=p_dist, r_dist=r_dist)
        self._init_seed = seed


class BanditTenArmedRandomRandom(BanditEnv):
    """10 armed bandit with random probabilities assigned to both payouts and rewards"""
    def __init__(self, bandits=10, seed=1):
        np_random, _ = seeding.np_random(seed)
        p_dist = np_random.uniform(size=bandits)
        r_dist = np_random.uniform(size=bandits)
        super().__init__(p_dist=p_dist, r_dist=r_dist)
        self._init_seed = seed


class BanditTenArmedGaussian(BanditEnv):
    """
    10 armed bandit mentioned on page 30 of Sutton and Barto's
    'Reinforcement Learning: An Introduction'

    Actions always pay out
    Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
    Actual reward is drawn from a normal distribution (q*(a), 1)
    """
    def __init__(self, bandits=10, seed=1):
        np_random, _ = seeding.np_random(seed)
        p_dist = np.full(bandits, 1)
        r_dist = []
        for _ in range(bandits):
            # [mean, std] 형태로 저장
            r_dist.append([np_random.normal(0, 1), 1])

        super().__init__(p_dist=p_dist, r_dist=r_dist)
        self._init_seed = seed