"""
The stronger sparring partners: built on spar_v1 (range reading and expected
value), then each adds what model.py must learn to beat. Not part of the
submission: model.py never imports this file.

    Balanced  fixed theory-style frequencies: gives nothing to exploit
    Counter   reads the opponent's tendencies and punishes each one
    Bandit    tries whole strategies, keeps what wins chips per opponent
"""

import math

from player_action import PlayerAction
from spar_v1 import Model as V1
from spar_v1 import _percentile, _Rng, _score, _to_int


class _Strong(V1):
    RNG_SEED = 1

    def __init__(self):
        super().__init__()
        self.rng = _Rng(self.RNG_SEED)
        self.fold_scale = 1.0  # how much of the estimated fold equity to trust

    def _decide(self, gs, equity, realise_bonus=0.0, margin=0.01):
        """spar_v1's choice of action, with the dials exposed."""
        pot, call = gs.pot, gs.call_amount
        realise = min(1.0, self._realise(gs) + realise_bonus)
        best = PlayerAction.call(self.player_id)
        if call == 0:
            best_ev = equity * pot * realise
        else:
            best_ev = equity * (pot + call) * realise - call
            if best_ev < 0:
                best, best_ev = PlayerAction.fold(self.player_id), 0.0
        if gs.can_raise:
            for total in self._raise_sizes(gs):
                ev = self._raise_ev(gs, equity * realise, total)
                if ev > best_ev + margin * pot:
                    best, best_ev = PlayerAction.raise_to(self.player_id, total), ev
        return best

    def _fold_equity(self, gs, opponents, size):
        return min(0.97, super()._fold_equity(gs, opponents, size) * self.fold_scale)

    def _raise(self, gs, total):
        if not gs.can_raise:
            return PlayerAction.call(self.player_id)
        low, high = gs.raise_range
        return PlayerAction.raise_to(self.player_id, min(high, max(low, int(total))))

    def _pot_bet(self, gs, fraction):
        return self._raise(gs, gs.current_bet + fraction * (gs.pot + gs.call_amount))

    def _fold_or_check(self, gs):
        if gs.call_amount == 0:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)

    def _rate(self, s, hits, tries, prior_hits, prior_tries):
        return (s[hits] + prior_hits) / (s[tries] + prior_tries)

    def _opponent_stats(self, gs):
        for p in gs.players_in_hand:
            if p != self.player_id:
                return self._stats(gs.teams[p])
        return self._new_stats()


class Balanced(_Strong):
    RNG_SEED = 101

    def run(self, actions, gs):
        self._observe(actions, gs)
        if gs.street == "preflop":
            return self._preflop(gs, self.rng.random())
        return self._postflop(gs, self.rng.random())

    def _preflop(self, gs, r):
        a, b = [_to_int(c) for c in self.hand]
        pct = _percentile(a, b)
        bb = gs.big_blind
        if gs.current_bet <= bb:
            if gs.button == self.player_id:  # first in
                return self._raise(gs, 2.5 * bb) if pct < 0.85 else self._fold_or_check(gs)
            if pct < 0.25:
                return self._raise(gs, 3 * bb)
            return PlayerAction.call(self.player_id)
        # Defend by the price: a bigger raise is defended with a narrower range.
        price = gs.call_amount / max(1, gs.pot)
        defend = max(0.25, 1 - 1.15 * price)
        if pct < 0.08 or (0.45 < pct < 0.6 and r < 0.2):  # value, and some bluffs
            return self._raise(gs, 3 * gs.current_bet)
        return PlayerAction.call(self.player_id) if pct < defend else self._fold_or_check(gs)

    def _board_rank(self, gs):
        """Share of all hands that this hand beats on this board."""
        hand = [_to_int(c) for c in self.hand]
        board = [_to_int(c) for c in gs.board]
        known = set(hand + board)
        unseen = [c for c in range(52) if c not in known]
        mine = _score(hand + board)
        beaten = total = 0
        for i, a in enumerate(unseen):
            for b in unseen[i + 1:]:
                beaten += _score([a, b] + board) < mine
                total += 1
        return beaten / total

    def _postflop(self, gs, r):
        rank = self._board_rank(gs)
        if gs.call_amount == 0:
            size = 0.66
            bluffs = size / (1 + 2 * size)  # the share that makes calling break even
            if rank > 0.72 and r < 0.85:
                return self._pot_bet(gs, size)
            if rank < 0.3 and r < bluffs:
                return self._pot_bet(gs, size)
            return PlayerAction.call(self.player_id)
        alpha = gs.call_amount / max(1, gs.pot)  # folds this often: their bluffs profit
        if rank > 0.93 and r < 0.6:
            return self._pot_bet(gs, 0.9)
        if rank < 0.12 and r < 0.06 and gs.street != "river":
            return self._pot_bet(gs, 0.9)
        if rank >= 0.9 * alpha or self._equity(gs) > gs.pot_odds:
            return PlayerAction.call(self.player_id)
        return PlayerAction.fold(self.player_id)


class Counter(_Strong):
    RNG_SEED = 202

    def run(self, actions, gs):
        self._observe(actions, gs)
        equity = self._equity(gs)
        s = self._opponent_stats(gs)
        opens = self._rate(s, "opens", "hands", 4, 8)
        folds = self._rate(s, "folds", "faced", 4, 10)
        bets = self._rate(s, "post_bets", "post_actions", 4, 12)
        r = self.rng.random()

        # They fold too much: lean on them. They never fold: value only.
        self.fold_scale = 1.3 if folds > 0.55 else 0.5 if folds < 0.3 else 1.0

        if gs.street == "preflop":
            facing_open = gs.current_bet > gs.big_blind and gs.button != self.player_id
            if facing_open and opens > 0.6 and gs.can_raise:
                # They steal: re-raise a wider range, harder when it folds them.
                wide = 0.46 if folds > 0.45 else 0.52
                if equity > wide:
                    return self._raise(gs, 3.2 * gs.current_bet)
            return self._decide(gs, equity)

        strong = equity > 0.8
        if bets > 0.45:
            # They bet too much: call them down, and let them bet into us.
            if strong and gs.call_amount == 0 and gs.street != "river" and r < 0.4:
                return PlayerAction.call(self.player_id)
            return self._decide(gs, equity, realise_bonus=0.08, margin=0.03)
        if bets < 0.15:
            return self._decide(gs, equity, realise_bonus=-0.1)  # their bets mean it
        return self._decide(gs, equity)


# fold_scale, realise bonus, raise margin: a whole strategy each.
PROFILES = (
    (1.0, 0.0, 0.01),    # as spar_v1
    (0.7, -0.06, 0.03),  # tighter, fewer bluffs
    (1.4, 0.0, 0.0),     # pressure
    (0.85, 0.07, 0.02),  # sticky, calls down
)


class Bandit(_Strong):
    RNG_SEED = 303

    def __init__(self):
        super().__init__()
        self.scores = {}  # team -> per profile, [hands, total big blinds won]
        self.profile = 0
        self.start = None  # my chips when this hand began

    def seed(self, hand, player_id):
        super().seed(hand, player_id)
        self.start = None
        self.profile = None

    def _choose(self, team):
        """The usual upper confidence bound. Hands swing by about 10 big
        blinds, which sets how long a profile needs to prove itself."""
        arms = self.scores.setdefault(team, [[0, 0.0] for _ in PROFILES])
        total = sum(a[0] for a in arms) + 1
        best, best_value = 0, -math.inf
        for i, (n, won) in enumerate(arms):
            value = math.inf if n == 0 else won / n + 10 * math.sqrt(2 * math.log(total) / n)
            if value > best_value:
                best, best_value = i, value
        return best

    def run(self, actions, gs):
        self._observe(actions, gs)
        me = self.player_id
        team = self._opponent_team(gs)
        if self.start is None:
            self.start = gs.stacks[me] + gs.bets[me]
        if self.profile is None:
            self.profile = self._choose(team)
        self.fold_scale, bonus, margin = PROFILES[self.profile]
        return self._decide(gs, self._equity(gs), bonus, margin)

    def _opponent_team(self, gs):
        return next(gs.teams[p] for p in range(gs.num_players) if p != self.player_id)

    def end_game(self, actions, gs):
        super().end_game(actions, gs)
        if self.start is None or self.profile is None:
            return  # they folded before we acted: no strategy was tried
        arm = self.scores[self._opponent_team(gs)][self.profile]
        arm[0] += 1
        arm[1] += (gs.stacks[self.player_id] - self.start) / gs.big_blind


STRONG = {"balanced": Balanced, "counter": Counter, "bandit": Bandit}
