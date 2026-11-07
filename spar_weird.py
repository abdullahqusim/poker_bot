"""
Sparring partners that play oddly on purpose: real teams will do things a
sensible bot does not, and each of these breaks an assumption of some kind.
Not part of the submission: model.py never imports this file.

    MinRaiser     raises the minimum nearly every turn
    Overbettor    bets three times the pot, or all-in
    Limper        calls preflop with anything, rarely raises
    RiverBluffer  passive until the river, then shoves polarised
    Switcher      changes style every 40 hands, to punish slow readers
"""

from player_action import PlayerAction
from sparring import Lag, Tag, _Spar


class MinRaiser(_Spar):
    RNG_SEED = 401

    def decide(self, gs, equity, r):
        if gs.can_raise and (equity > 0.35 or r < 0.3):
            return self._raise(gs, gs.min_raise_to)
        if equity > gs.pot_odds - 0.05:
            return self._call()
        return self._fold_or_check(gs)


class Overbettor(_Spar):
    RNG_SEED = 402

    def decide(self, gs, equity, r):
        if gs.street == "preflop":
            if equity > 0.55:
                return self._raise(gs, gs.max_raise_to if r < 0.25 else 4 * gs.big_blind)
            if equity > gs.pot_odds + 0.05:
                return self._call()
            return self._fold_or_check(gs)
        if equity > 0.62 or (gs.call_amount == 0 and r < 0.25):
            return self._bet(gs, 3.0)
        if equity > gs.pot_odds + 0.05:
            return self._call()
        return self._fold_or_check(gs)


class Limper(_Spar):
    RNG_SEED = 403

    def decide(self, gs, equity, r):
        if gs.street == "preflop":
            if gs.call_amount <= 0.15 * gs.stacks[self.player_id]:
                return self._call()
            return self._call() if equity > 0.55 else self._fold_or_check(gs)
        if equity > 0.85:
            return self._bet(gs, 0.6)
        if equity > gs.pot_odds - 0.1:
            return self._call()
        return self._fold_or_check(gs)


class RiverBluffer(_Spar):
    RNG_SEED = 404

    def decide(self, gs, equity, r):
        if gs.street == "river" and (equity > 0.75 or (equity < 0.2 and r < 0.5)):
            return self._raise(gs, gs.max_raise_to)
        if equity > gs.pot_odds:
            return self._call()
        return self._fold_or_check(gs)


class Switcher(_Spar):
    RNG_SEED = 405

    def __init__(self):
        super().__init__()
        self.hands_played = 0

    def seed(self, hand, player_id):
        super().seed(hand, player_id)
        self.hands_played += 1

    def decide(self, gs, equity, r):
        style = Tag if (self.hands_played // 40) % 2 == 0 else Lag
        return style.decide(self, gs, equity, r)


WEIRD = {
    "minraiser": MinRaiser,
    "overbettor": Overbettor,
    "limper": Limper,
    "riverbluffer": RiverBluffer,
    "switcher": Switcher,
}
