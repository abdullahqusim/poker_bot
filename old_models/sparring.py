"""
Sparring partners for model.py: four styles, each about as complex as the
starter bot (Monte Carlo equity plus opponent tracking). Not part of the
submission: model.py never imports this file, so submit does not list it.

    Tag      tight and aggressive: picks spots, then bets them hard
    Lag      loose and aggressive: wide ranges, barrels, attacks folders
    Station  loose and passive: calls light, value bets thin, rarely bluffs
    Trapper  slow plays strong hands, check-raises, punishes aggression
"""

from card_utils import evaluate, full_deck
from player_action import DEALER, PlayerAction

SAMPLES = 150
PREFLOP_SAMPLES = 500


class _Rng:
    """Tiny LCG: plain ints only, so the pickle stays plain data."""

    def __init__(self, seed):
        self.state = seed

    def random(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.state >> 11) / float(1 << 53)

    def sample(self, seq, k):
        pool = list(seq)
        for i in range(k):
            j = i + int(self.random() * (len(pool) - i))
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]


class _Spar:
    RNG_SEED = 1

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.rng = _Rng(self.RNG_SEED)
        self.preflop_cache = {}
        self.stats = {}  # team -> [actions, raises, faced a bet, folded to it]
        self.facing_bet = False  # within the street, as actions are replayed
        self.i_raised_last = False  # the last raise of the hand was ours

    def seed(self, hand, player_id):
        self.hand = hand
        self.player_id = player_id
        self.facing_bet = False
        self.i_raised_last = False

    def run(self, actions, gs):
        self._observe(actions, gs)
        return self.decide(gs, self._equity(gs), self.rng.random())

    def end_game(self, actions, gs):
        self._observe(actions, gs)

    def decide(self, gs, equity, r):
        raise NotImplementedError

    # ---- helpers ---------------------------------------------------------

    def _call(self):
        return PlayerAction.call(self.player_id)

    def _fold_or_check(self, gs):
        return self._call() if gs.call_amount == 0 else PlayerAction.fold(self.player_id)

    def _raise(self, gs, total):
        if not gs.can_raise:
            return self._call()
        low, high = gs.raise_range
        return PlayerAction.raise_to(self.player_id, min(high, max(low, int(total))))

    def _bet(self, gs, fraction):
        """A bet or raise of fraction of the pot, after calling."""
        return self._raise(gs, gs.current_bet + fraction * (gs.pot + gs.call_amount))

    def _observe(self, actions, gs):
        for a in actions:
            if a.player_id == DEALER:
                self.facing_bet = False
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            if a.kind == "raise":
                self.i_raised_last = a.player_id == self.player_id
            if a.player_id != self.player_id:
                s = self.stats.setdefault(gs.teams[a.player_id], [0, 0, 0, 0])
                s[0] += 1
                s[1] += a.kind == "raise"
                s[2] += self.facing_bet
                s[3] += self.facing_bet and a.kind == "fold"
            if a.kind == "raise":
                self.facing_bet = True

    def _opponents(self, gs):
        return [gs.teams[p] for p in gs.players_in_hand if p != self.player_id]

    def _fold_rate(self, gs):
        """How often the opponents fold to a bet (prior: 0.4)."""
        s = [self.stats.get(t, [0, 0, 0, 0]) for t in self._opponents(gs)]
        return min((x[3] + 4) / (x[2] + 10) for x in s) if s else 0.4

    def _raise_rate(self, gs):
        """Share of the opponents' actions that are raises (prior: 0.25)."""
        s = [self.stats.get(t, [0, 0, 0, 0]) for t in self._opponents(gs)]
        return max((x[1] + 5) / (x[0] + 20) for x in s) if s else 0.25

    def _hand_class(self):
        a, b = sorted(self.hand, key=lambda c: c.value, reverse=True)
        if a.value == b.value:
            return a.rank + b.rank
        return a.rank + b.rank + ("s" if a.suit == b.suit else "o")

    def _equity(self, gs):
        board = list(gs.board)
        opponents = max(1, len(gs.players_in_hand) - 1)
        if not board and opponents == 1:
            key = self._hand_class()
            if key not in self.preflop_cache:
                self.preflop_cache[key] = self._monte_carlo(board, 1, PREFLOP_SAMPLES)
            return self.preflop_cache[key]
        return self._monte_carlo(board, opponents, SAMPLES)

    def _monte_carlo(self, board, opponents, samples):
        known = self.hand + board
        unseen = [c for c in full_deck() if c not in known]
        missing = 5 - len(board)
        won = 0.0
        for _ in range(samples):
            cards = self.rng.sample(unseen, missing + 2 * opponents)
            full_board = board + cards[:missing]
            mine = evaluate(self.hand + full_board)
            best, ties = True, 1
            for i in range(opponents):
                theirs = evaluate(cards[missing + 2 * i: missing + 2 * i + 2] + full_board)
                if theirs > mine:
                    best = False
                    break
                if theirs == mine:
                    ties += 1
            if best:
                won += 1.0 / ties
        return won / samples


class Tag(_Spar):
    RNG_SEED = 11

    def decide(self, gs, equity, r):
        bb = gs.big_blind
        if gs.street == "preflop":
            raised = gs.current_bet > bb
            if raised:
                if equity > 0.66:
                    return self._raise(gs, 3 * gs.current_bet)
                if equity > 0.56 and gs.call_amount <= 0.2 * gs.stacks[self.player_id]:
                    return self._call()
                return self._fold_or_check(gs)
            if equity > 0.52:
                return self._raise(gs, 3 * bb)
            return self._fold_or_check(gs)

        price = gs.call_amount / max(1, gs.pot - gs.call_amount)
        discounted = equity - 0.12 * min(1.0, price)
        if gs.call_amount == 0:
            if equity > 0.68:
                return self._bet(gs, 0.7)
            cbet = 0.45 + 0.5 * (self._fold_rate(gs) - 0.4)
            if self.i_raised_last and r < cbet:
                return self._bet(gs, 0.5)
            return self._call()
        if equity > 0.85:
            return self._bet(gs, 0.9)
        if discounted > gs.pot_odds + 0.05:
            return self._call()
        return PlayerAction.fold(self.player_id)


class Lag(_Spar):
    RNG_SEED = 22

    def decide(self, gs, equity, r):
        bb = gs.big_blind
        if gs.street == "preflop":
            if gs.current_bet > bb:
                if equity > 0.6 or (equity > 0.5 and r < 0.15):
                    return self._raise(gs, 3 * gs.current_bet)
                if equity > 0.45 and gs.call_amount <= 0.25 * gs.stacks[self.player_id]:
                    return self._call()
                return self._fold_or_check(gs)
            if equity > 0.40:
                return self._raise(gs, 3 * bb)
            return self._fold_or_check(gs)

        bluff = max(0.05, min(0.6, 0.35 + (self._fold_rate(gs) - 0.4)))
        if gs.call_amount == 0:
            if equity > 0.55 or r < bluff:
                return self._bet(gs, 0.75)
            return self._call()
        if equity > 0.8 or (r < 0.08 and gs.street != "river"):
            return self._bet(gs, 1.0)
        if equity > gs.pot_odds:
            return self._call()
        return PlayerAction.fold(self.player_id)


class Station(_Spar):
    RNG_SEED = 33

    def decide(self, gs, equity, r):
        bb = gs.big_blind
        if gs.street == "preflop":
            if equity > 0.62:
                return self._raise(gs, max(3 * bb, 2.5 * gs.current_bet))
            if equity > 0.45 and gs.call_amount <= 0.25 * gs.stacks[self.player_id]:
                return self._call()
            return self._fold_or_check(gs)

        if gs.call_amount == 0:
            if equity > 0.58:
                return self._bet(gs, 0.6)
            if gs.street == "river" and r < 0.05:
                return self._bet(gs, 0.5)
            return self._call()
        if equity > 0.85:
            return self._bet(gs, 0.75)
        # Calls a little light, and lighter still against a frequent raiser.
        margin = 0.0 - 0.2 * max(0.0, self._raise_rate(gs) - 0.25)
        if equity > gs.pot_odds + margin:
            return self._call()
        return PlayerAction.fold(self.player_id)


class Trapper(_Spar):
    RNG_SEED = 44

    def decide(self, gs, equity, r):
        bb = gs.big_blind
        passive = self._raise_rate(gs) < 0.15  # traps need someone to bet
        if gs.street == "preflop":
            if gs.current_bet > bb:
                if equity > 0.72 and r < 0.4:
                    return self._raise(gs, 3 * gs.current_bet)
                if equity > 0.5 and gs.call_amount <= 0.2 * gs.stacks[self.player_id]:
                    return self._call()
                return self._fold_or_check(gs)
            if equity > 0.72 and r < 0.3:
                return self._raise(gs, 3 * bb)
            if equity > 0.42:
                return self._call()
            return self._fold_or_check(gs)

        if gs.call_amount == 0:
            if equity > 0.8:
                if gs.street == "flop" and not passive:
                    return self._call()  # the trap: let them bet
                return self._bet(gs, 0.8)
            if equity > 0.6:
                return self._bet(gs, 0.5)
            if gs.street == "river" and r < 0.1:
                return self._bet(gs, 0.6)
            return self._call()
        if equity > 0.75:
            return self._bet(gs, 1.5)  # the check-raise
        if equity > gs.pot_odds + 0.02:
            return self._call()
        return PlayerAction.fold(self.player_id)


BOTS = {"tag": Tag, "lag": Lag, "station": Station, "trapper": Trapper}
