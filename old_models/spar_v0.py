from card_utils import Card, evaluate, full_deck
from game_state import GameStateView
from player_action import DEALER, PlayerAction

SAMPLES = 200          # Monte Carlo deals per decision (~40 ms heads-up)
PREFLOP_SAMPLES = 600  # preflop equities are cached per hand class


class _Rng:
    """Tiny LCG: plain ints only, so the pickle stays plain data."""

    def __init__(self, seed=12345):
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


class Model:
    def __init__(self):
        self.hand = []
        self.player_id = None
        self.rng = _Rng()
        self.preflop_cache = {}  # hand class (e.g. "AKs") -> equity vs 1 random hand
        # Per team name: [hands seen, voluntary raises, total actions]
        self.opp_stats = {}

    # ---- engine hooks ----------------------------------------------------

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id

    def run(self, actions: list[PlayerAction], game_state: GameStateView) -> PlayerAction:
        self._observe(actions, game_state)
        gs = game_state
        me = self.player_id
        opponents = max(1, len(gs.players_in_hand) - 1)

        equity = self._equity(gs, opponents)
        equity = self._adjust_for_aggression(equity, gs)

        pot = gs.pot
        to_call = gs.call_amount
        pot_odds = gs.pot_odds
        r = self.rng.random()

        # Strong: build the pot.
        if equity > 0.70 and gs.can_raise:
            size = 1.0 if equity > 0.85 else 0.75
            return self._raise(gs, gs.current_bet + int(size * (pot + to_call)))

        # Decent and unopened: bet for value / protection.
        if equity > 0.58 and to_call == 0 and gs.can_raise:
            return self._raise(gs, gs.current_bet + int(0.5 * pot))

        # Occasional bluff when checked to, on the flop or turn.
        if to_call == 0 and gs.can_raise and gs.street in ("flop", "turn") and r < 0.12:
            return self._raise(gs, gs.current_bet + int(0.5 * pot))

        if to_call == 0:
            return PlayerAction.call(me)  # check

        # Facing a bet: call when equity beats the price (small margin).
        if equity > pot_odds + 0.03:
            return PlayerAction.call(me)
        return PlayerAction.fold(me)

    def end_game(self, actions: list[PlayerAction], game_state: GameStateView):
        self._observe(actions, game_state)
        for pid, team in enumerate(game_state.teams):
            if pid != self.player_id:
                self._stats(team)[0] += 1

    # ---- helpers ---------------------------------------------------------

    def _raise(self, gs, total):
        low, high = gs.raise_range
        return PlayerAction.raise_to(self.player_id, min(high, max(low, int(total))))

    def _stats(self, team):
        return self.opp_stats.setdefault(team, [0, 0, 0])

    def _observe(self, actions, gs):
        for a in actions:
            if a.player_id in (DEALER, self.player_id) or a.kind not in ("fold", "call", "raise"):
                continue
            s = self._stats(gs.teams[a.player_id])
            s[2] += 1
            if a.kind == "raise":
                s[1] += 1

    def _aggression(self, team):
        """Share of an opponent's actions that are raises (prior: 0.25)."""
        _, raises, total = self._stats(team)
        return (raises + 5) / (total + 20)

    def _adjust_for_aggression(self, equity, gs):
        """A raise from a tight player means a strong hand; from a maniac it
        means little. Shade equity down by how much the bettor's raises mean."""
        if gs.call_amount == 0:
            return equity
        bettors = [p for p in gs.players_in_hand
                   if p != self.player_id and gs.bets[p] == gs.current_bet]
        if not bettors:
            return equity
        agg = max(self._aggression(gs.teams[p]) for p in bettors)
        # agg 0.1 (nit) -> strong discount; agg 0.6+ (maniac) -> none.
        discount = max(0.0, min(0.25, 0.3 * (0.6 - agg)))
        size_factor = min(1.0, gs.call_amount / max(1, gs.pot - gs.call_amount))
        return equity * (1 - discount * size_factor)

    def _hand_class(self):
        a, b = sorted(self.hand, key=lambda c: c.value, reverse=True)
        if a.value == b.value:
            return a.rank + b.rank
        return a.rank + b.rank + ("s" if a.suit == b.suit else "o")

    def _equity(self, gs, opponents):
        board = list(gs.board)
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
                opp = cards[missing + 2 * i: missing + 2 * i + 2]
                theirs = evaluate(opp + full_board)
                if theirs > mine:
                    best = False
                    break
                if theirs == mine:
                    ties += 1
            if best:
                won += 1.0 / ties
        return won / samples
