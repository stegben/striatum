import logging

from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class Exp4(object):
    """Multi-armed Contextual Bandit Algorithm -- EXP4.

    EXP4 : Exponential Weighted Algorithm for Exploration and Exploitation
           using Expert Advices
    Upper bound for regret(T) = O(T+K*lnN)

    Parameters
    ----------
    storage
    actions
    max_iter : int
    gamma : float, optional (default=0.1)

    References
    ----------
    .. Auer, Peter and Cesa-Bianchi, Nicol and Freund
       , Yoav and Schapire, Robert E. "The Nonstochastic
        Multiarmed Bandit Problem" Published in SIAM Journal
        on Computing Volume 32 Issue 1, 2003 Pages 48-77
    """

    def __init__(self, storage, actions, max_iter, models
                 , gamma=0.1):
        super(Exp4, self).__init__(storage, actions)

        self.last_reward = None
        self.last_history_id = None

        self.models = models

        # max iters
        if not isinstance(max_iter, int):
            raise ValueError("max_iter should be int.")
        self.max_iter = max_iter

        # 1>gamma>0
        if not isinstance(gamma, float):
            raise ValueError("gamma should be float.")
        elif (gamma < 0) or (gamma > 1):
            raise ValueError("gamma should be in [0,1].")
        else:
            self.gamma = gamma


        self.exp4_ = None

        self.last_action_idx = None


    def exp4(self, x):
        """The generator which implements the main part of Exp4
        Parameters
        ----------
        reward: float
            The reward value.
        Yields
        ------
        q: array-like, shape = [K]
            The query vector which tells ALBL what kind of distribution if
            should sample from the unlabeled pool.
        """
        n_experts = len(self.models)
        n_arms = len(self.advices)
        w = np.ones(n_experts)
        advice = np.zeros((n_experts, n_arms))
        param = self.gamma ./ n_arms

        while True:
            for i, model in enumerate(self.models):
                advice[i] = model.predict_proba(x)

            # choice vector, shape = (self.K, )
            temp = np.dot(w, advice) / np.sum(w)
            query_vector = (1 - self.gamma) * temp + param

            reward, action_idx = yield query_vector

            # update w
            rhat = reward / query_vector[action_idx]
            yhat = rhat * advice[:, action_idx]

            w = w * np.exp(param * yhat)

        raise StopIteration


    def get_action(self, context):
        """Return the action to perform
        Parameters
        ----------
        context : {array-like, None}
            The context of current state, None if no context avaliable.
        Returns
        -------
        history_id : int
            The history id of the action.
        action : Actions object
            The action to perform.
        """
        if self.exp4_ is None:
            self.exp4_ = self.exp4(context)
            query_vector = self.exp4_.next()
        else:
            query_vector = self.exp4_.send(self.last_reward, self.last_action_idx)

        if self.last_reward is None:
            raise ValueError("The last reward have not been passed in.")

        action_idx = np.random.choice(np.arange(len(self.actions)), size=1,
                    p=query_vector)[0]

        history_id = self.storage.add_history(context,
                                              self.actions[action_idx],
                                              reward=None)
        self.last_history_id = history_id
        self.last_reward = None
        self.last_action_idx = action_idx

        return history_id, self.actions[action_idx]

    def reward(self, history_id, reward):
            """Reward the preivous action with reward.
            Parameters
            ----------
            history_id : int
                The history id of the action to reward.
            reward : float
                A float representing the feedback given to the action, the higher
                the better.
            """
            if history_id != self.last_history_id:
                raise ValueError("The history_id should be the same as last one.")

            if not isinstance(reward, float):
                raise ValueError("reward should be a float.")

            if reward > 1. or reward < 0.:
                LOGGER.warning("reward passing in should be between 0 and 1"
                               "to maintain theoratical guarantee.")

            self.last_reward = reward
            self.storage.reward(history_id, reward)