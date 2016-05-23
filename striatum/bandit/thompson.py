import logging
import math

from striatum.bandit.bandit import BaseBandit

import numpy as np

LOGGER = logging.getLogger(__name__)


class LinThompSamp (BaseBandit):

    """Multi-armed Contextual Bandit Algorithm -- Thompson.

    Parameters
    ----------
    storage
    actions
    delta : float, optional (default=0.1)
    R : float, optional (default=)
    epsilon : float, optional (default=)

    # Input ========================================================================
    param_delta       : parameter 0 < delta <= 1
    param_R           : parameter R >= 0 for R-sub Gaussian
    param_epsilon     : parameter 0 < epsilon < 1
    param_t           : cumulative number of trails
    mtx_context       : contextual vector for each arm
                        (number of arms * dimension of contextual vector)
    mtx_covariance    : the estimated covariance matrix of normal distribution B^(-1)
    vector_mean       : the estimated mean vector of normal distribution
    vector_f          : cumulative selected contextual vector with reward
                        (dimension of contextual vector*1)
    reward            : reward in this trail
    # Output ========================================================================
    param_t           : cumulative number of trails
    mtx_covariance    : the estimated covariance matrix of normal distribution B^(-1)
    vector_mean       : the estimated mean vector of normal distribution
    vector_f          : cumulative selected contextual vector with reward

    References
    ----------
    Thompson Sampling for Contextual Bandits with Linear Payoffs
    Authors : Shipra Agrawal, Navin Goyal (Microsoft Research India)
    URL : http://jmlr.csail.mit.edu/proceedings/papers/v28/agrawal13.pdf
    We follow the symbols used by
    http://www.cs.cmu.edu/~lizhou/files/contextual_bandit_survey.pdf
    """

    def __init__(self, actions, storage, delta=0.9, R=1, epsilon=0.1):
        super(LinThompSamp, self).__init__(storage, actions)

        self.last_reward = None
        self.last_history_id = None

        #0 < delta <= 1
        if not isinstance(delta, float):
            raise ValueError("delta should be float")
        elif (delta < 0) or (delta > 1):
            raise ValueError("delta should be in (0, 1]")
        else:
            self.delta = delta

        # R > 0
        if not isinstance(R, float):
            raise ValueError("R should be float")
        elif (R < 0) or (R=0):
            raise ValueError("R should be positive")
        else:
        self.R = R

        #0 < epsilon < 1
        if not isinstance(epsilon, float):
            raise ValueError("epsilon should be float")
        elif (epsilon < 0) or (epsilon > 1):
            raise ValueError("epsilon should be in (0, 1)")
        else:
            self.epsilon = epsilon

        self.linthompsamp_ = None

        self.last_action_idx = None


    def linthompsamp(self, x):
        self.dim_context = x.shape[1] # dimension of context vector
        self.v2 = (R ** 2) * 24 * self.dim_context *\
                math.log(1. / param_delta) * (1. / param.epsilon)
        mtx_covariance = np.eye(self.dim_context)
        vector_mean = np.zeros(self.dim_context)
        vector_f = np.zeros(self.dim_context)

        while True:
            vector_estimean = np.random.multivariate_normal(vector_mean, self.v2 * mtx_covariance,1).T
            expected_reward = np.dot(x, vector_estimean)

            # next context and last reward
            at = np.random.chioce(np.where(expected_reward == np.max(expected_reward))[0])
            x_new, reward = yield at

            #update
            temp = np.dot(mtx_covariance, x[:, at])
            deno = 1 + np.dot(x[:, at].T, temp)
            if abs(deno) < 1e-10:
                deno = 1
            mtx_covariance = mtx_covariance - (np.dot(temp, temp.T) * (1. / deno))
            vector_f += reward * x[:, at]
            vector_mean = np.dot(mtx_covariance, vector_f)

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
        if self.linthompsamp_ is None:
            self.linthompsamp_ = self.linthompsamp(context)
        if self.last_reward is None:
            raise ValueError("The last reward have not been passed in.")    
        action = self.linthompsamp_.send(context, self.last_reward)

        self.last_reward = None

        history_id = self.storage.add_history(None,action, reward=None)

        self.last_history_id = history_id


        return history_id, action


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