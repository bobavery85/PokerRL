# Copyright (c) 2019 Eric Steinberger


"""
Wrap a poker environment to track history, for instance. Wrappers are never constructed directly, but only over the
env_builder interface. Creating an env_builder is your starting point to create an environment
(potentially with a wrapper).
"""

import numpy as np

from PokerRL.game.Poker import Poker
from PokerRL.game._.EnvWrapperBuilderBase import EnvWrapperBuilderBase as _EnvWrapperBuilderBase
from PokerRL.game._.wrappers.FlatHULimitPokerHistoryWrapper import \
    FlatHULimitPokerHistoryWrapper as _FlatHULimitPokerHistoryWrapper
from PokerRL.game._.wrappers.FlatNonHULimitPokerHistoryWrapper import \
    FlatNonHULimitPokerHistoryWrapper as _FlatNonHULimitPokerHistoryWrapper
from PokerRL.game._.wrappers.RecurrentHistoryWrapper import RecurrentHistoryWrapper as _RecurrentHistoryWrapper
from PokerRL.game._.wrappers.Vanilla import VanillaWrapper as _VanillaWrapper


class VanillaEnvBuilder(_EnvWrapperBuilderBase):
    """ For docs please refer to the corresponding wrapper class file. """
    WRAPPER_CLS = _VanillaWrapper


class HistoryEnvBuilder(_EnvWrapperBuilderBase):
    """ For docs please refer to the corresponding wrapper class file. """
    WRAPPER_CLS = _RecurrentHistoryWrapper

    def __init__(self, env_cls, env_args, invert_history_order=False):
        super().__init__(env_cls=env_cls, env_args=env_args)
        self.invert_history_order = invert_history_order


class FlatLimitPokerEnvBuilder(_EnvWrapperBuilderBase):
    """ For docs please refer to the corresponding wrapper class file. """
    WRAPPER_CLS = _FlatHULimitPokerHistoryWrapper

    def __init__(self, env_cls, env_args):
        assert env_cls.IS_FIXED_LIMIT_GAME
        assert env_args.n_seats == 2

        self._VEC_ROUND_OFFSETS = {}
        self._VEC_HALF_ROUND_SIZE = {}
        self.action_vector_size = 0
        for r in env_cls.RULES.ALL_ROUNDS_LIST:
            self._VEC_ROUND_OFFSETS[r] = self.action_vector_size
            self._VEC_HALF_ROUND_SIZE[r] = len([Poker.BET_RAISE, Poker.CHECK_CALL]) * (
                env_cls.MAX_N_RAISES_PER_ROUND[r] + 2)
            self.action_vector_size += self._VEC_HALF_ROUND_SIZE[r] * env_args.n_seats  # always 2 seats

        super().__init__(env_cls=env_cls, env_args=env_args)

    def _get_num_public_observation_features(self):
        _env = self.env_cls(env_args=self.env_args, lut_holder=self.lut_holder, is_evaluating=True)
        return _env.observation_space.shape[0] + self.action_vector_size

    def get_vector_idx(self, round_, p_id, nth_action_this_round, action_idx):
        # *2 in line 3 stands for len([Poker.BET_RAISE, Poker.CHECK_CALL]). If action is fold, the obs is never going to
        # be seen anyways.
        return self._VEC_ROUND_OFFSETS[round_] \
               + p_id * self._VEC_HALF_ROUND_SIZE[round_] \
               + nth_action_this_round * 2 \
               + action_idx - 1  # - 1 because fold (which is action_idx==0) is not recorded.

class FlatNonHULimitPokerEnvBuilder(_EnvWrapperBuilderBase):
    """ For docs please refer to the corresponding wrapper class file. """
    WRAPPER_CLS = _FlatNonHULimitPokerHistoryWrapper

    def __init__(self, env_cls, env_args):
        assert env_cls.IS_FIXED_LIMIT_GAME
        self._VEC_ROUND_OFFSETS = {}
        self._VEC_PER_PLAYER_ROUND_SIZE = {}
        self.action_vector_size = 0
        for r in env_cls.RULES.ALL_ROUNDS_LIST:
            self._VEC_ROUND_OFFSETS[r] = self.action_vector_size
            self._VEC_PER_PLAYER_ROUND_SIZE[r] = len([Poker.BET_RAISE, Poker.CHECK_CALL, Poker.FOLD]) * (
                env_cls.MAX_N_RAISES_PER_ROUND[r] + 2)
            self.action_vector_size += self._VEC_PER_PLAYER_ROUND_SIZE[r] * env_args.n_seats  # always 2 seats

        super().__init__(env_cls=env_cls, env_args=env_args)

    def _get_num_public_observation_features(self):
        _env = self.env_cls(env_args=self.env_args, lut_holder=self.lut_holder, is_evaluating=True)
        return _env.observation_space.shape[0] + self.action_vector_size

    def get_vector_idx(self, round_, p_id, nth_action_this_round, action_idx):
        return self._VEC_ROUND_OFFSETS[round_] \
               + p_id * self._VEC_PER_PLAYER_ROUND_SIZE[round_] \
               + nth_action_this_round * len([Poker.BET_RAISE, Poker.CHECK_CALL, Poker.FOLD]) \
               + action_idx
               
    def get_canonical_action_vector(self, action_vector, p_id):
        #print("builder: ", type(p_id), p_id)
        if p_id == 0:
            return action_vector
        assert len(action_vector) == self.action_vector_size
        for r in range(len(self._VEC_ROUND_OFFSETS)):
            round_start = self._VEC_ROUND_OFFSETS[r]
            if (r + 1) == len(self._VEC_ROUND_OFFSETS):
                round_end = self.action_vector_size
            else:
                round_end = self._VEC_ROUND_OFFSETS[r + 1]
            round_vector = np.array(action_vector[round_start:round_end])
            round_vector = np.roll(round_vector,
                                   -1 * p_id * self._VEC_PER_PLAYER_ROUND_SIZE[r])
            action_vector[round_start:round_end] = round_vector
        return action_vector

    def debug_action_vector(self, idx):
        actions = ['FOLD', 'CHECK_CALL', 'BET_RAISE']
        curr_idx = 0
        for i in range(len(self._VEC_ROUND_OFFSETS)):
            for j in range(self.N_SEATS):
                for k in range(self.env_cls.MAX_N_RAISES_PER_ROUND[i] + 2):
                    for l in range(len(actions)):
                        if curr_idx == idx:
                            print('Round: ', i, ' Player: ', j, ' Nth Action: ', k,
                                  ' Action: ', actions[l])
                            return
                        else:
                            curr_idx += 1
        print('Bad index: ', idx)
        assert False

ALL_BUILDERS = [
    HistoryEnvBuilder,
    FlatLimitPokerEnvBuilder,
    FlatNonHULimitPokerEnvBuilder,
    VanillaEnvBuilder
]
