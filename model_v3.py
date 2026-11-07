import math

from card_utils import Card
from game_state import GameStateView
from player_action import DEALER, PlayerAction

SAMPLES = 600  # Monte Carlo deals per decision (a few ms heads-up)
REALISE_PREFLOP_OOP = 0.8
REALISE_OOP = 0.88

# Player types, as what they do how often. A new opponent is scored against
# each row, so a few hands say which rates to expect before the counts can.
#            opens  folds  post_bets  pre_raises  bluffs  post_raises
ARCHETYPES = (
    (0.25, 0.65, 0.15, 0.06, 0.08, 0.05),  # nit
    (0.50, 0.50, 0.30, 0.12, 0.15, 0.10),  # solid
    (0.75, 0.35, 0.45, 0.22, 0.30, 0.20),  # aggressive
    (0.97, 0.12, 0.70, 0.50, 0.55, 0.40),  # maniac
    (0.50, 0.15, 0.15, 0.06, 0.10, 0.05),  # calling station
)
# The counters behind each column, and how many hands of weight its prior has.
COLUMNS = (
    ("opens", "hands", 8),
    ("folds", "faced", 10),
    ("post_bets", "post_actions", 10),
    ("pre_raises", "pre_faced", 5),
    ("bluffs", "shown", 8),
    ("post_raises", "post_faced", 10),
)
NEUTRAL = (0.5, 0.4, 0.4, 0.2, 0.25, 0.12)  # the prior with no matrix
TYPE_CONFIDENCE = 0.5  # the rows are rough: count evidence at half weight
RERAISED_EQUITY = 0.8  # our equity once re-raised, as a share of before

# Whole strategies, as fold equity trusted, realise bonus and raise margin.
PROFILES = (
    (1.0, 0.0, 0.01),    # the default
    (0.7, -0.06, 0.03),  # tighter, fewer bluffs
    (1.4, 0.0, 0.0),     # pressure
    (0.85, 0.07, 0.02),  # sticky, calls down
)

# The 169 starting hands, best first, by equity against a random hand.
HAND_ORDER = (
    "AA KK QQ JJ TT 99 88 AKs AQs 77 AKo AJs ATs AQo KQs AJo 66 ATo KJs A9s KQo KTs A8s A7s "
    "KJo A9o QJs A5s A8o KTo 55 A6s K9s QTs A7o A4s K8s QJo A3s A5o K9o JTs 44 A6o K7s A2s "
    "Q9s QTo A4o K6s K8o Q8s J9s Q9o K5s A3o K7o JTo J8s A2o T9s Q7s Q8o K3s K4s K5o K6o 33 "
    "Q6s J9o K2s Q5s K4o J7s T8s Q4s J8o Q6o K3o T9o 98s Q7o Q3s J6s T7s 22 K2o Q5o J5s Q2s "
    "T8o J7o Q4o J4s 97s T6s J3s Q3o J6o 87s T7o 98o T5s J2s Q2o 96s J5o 86s T4s 97o T6o 95s "
    "J4o J3o T3s 87o T2s 76s J2o 85s 96o 94s 75s T5o 86o 65s 93s T4o 84s T3o 76o 92s 95o T2o "
    "74s 54s 85o 64s 83s 94o 75o 93o 82s 84o 73s 53s 65o 63s 92o 74o 54o 72s 64o 43s 62s 52s "
    "83o 82o 42s 73o 63o 53o 32s 72o 43o 62o 52o 42o 32o"
).split()
RANKS = "23456789TJQKA"
SUITS = "CDHS"


def _percentiles():
    """(high rank, low rank, suited) -> share of hands at least this good."""
    weights = [(key, 6 if len(key) == 2 else 4 if key[2] == "s" else 12) for key in HAND_ORDER]
    total, seen, table = sum(w for _, w in weights), 0, {}
    for key, weight in weights:
        seen += weight
        hi, lo = RANKS.index(key[0]), RANKS.index(key[1])
        table[hi, lo, key[-1] == "s"] = seen / total
    return table


PERCENTILE = _percentiles()


def _to_int(card):
    return (card.value - 2) << 2 | SUITS.index(card.suit)


def _percentile(a, b):
    """Share of starting hands at least as good as the ints a, b."""
    hi, lo = a >> 2, b >> 2
    if hi < lo:
        hi, lo = lo, hi
    return PERCENTILE[hi, lo, a & 3 == b & 3 and hi != lo]


def _top_bits(mask, count):
    """The count highest set bits of mask, packed four bits each."""
    out = 0
    for _ in range(count):
        top = mask.bit_length() - 1
        out = out << 4 | top
        mask &= ~(1 << top)
    return out


def _straight(mask):
    """The straight in a rank mask (the ace also plays low): 0 for none."""
    m = mask << 1 | mask >> 12
    return (m & m >> 1 & m >> 2 & m >> 3 & m >> 4).bit_length()


def _score(cards):
    """A fast evaluate() on 5 to 7 int cards: higher wins, equal splits.
    The top bits are the card_utils category."""
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suits[c & 3] += 1
        mask |= 1 << r
    flush = fmask = 0
    for s in range(4):
        if suits[s] >= 5:
            flush = s + 1
    if flush:
        for c in cards:
            if c & 3 == flush - 1:
                fmask |= 1 << (c >> 2)
        run = _straight(fmask)
        if run:
            return 8 << 20 | run
    quads = trips = -1
    pairs = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            quads = r
        elif n == 3:
            if trips < 0:
                trips = r
            else:
                pairs.append(r)
        elif n == 2:
            pairs.append(r)
    if quads >= 0:
        return 7 << 20 | quads << 4 | (mask & ~(1 << quads)).bit_length() - 1
    if trips >= 0 and pairs:
        return 6 << 20 | trips << 4 | max(pairs)
    if flush:
        return 5 << 20 | _top_bits(fmask, 5)
    run = _straight(mask)
    if run:
        return 4 << 20 | run
    if trips >= 0:
        return 3 << 20 | trips << 8 | _top_bits(mask & ~(1 << trips), 2)
    if len(pairs) >= 2:
        hi, lo = pairs[0], pairs[1]
        return 2 << 20 | hi << 8 | lo << 4 | (mask & ~(1 << hi) & ~(1 << lo)).bit_length() - 1
    if pairs:
        return 1 << 20 | pairs[0] << 12 | _top_bits(mask & ~(1 << pairs[0]), 3)
    return _top_bits(mask, 5)


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
    USE_MATRIX = True
    USE_PROFILES = True

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.rng = _Rng()
        # Per team name: see _new_stats.
        self.opp_stats = {}
        # This hand, per seat.
        self.range_top = {}  # the share of starting hands they still hold
        self.aggression = {}  # bets and raises after the flop
        self.acted_preflop = set()
        self.facing_bet = False
        self.board_seen = 0  # board cards replayed so far
        self.shown = {}  # hole cards revealed at the showdown
        self.scores = {}  # team -> per profile, [hands, big blinds won]
        self.fold_trust = 1.0  # this hand's profile: fold equity trusted
        self.profile = None  # this hand's, once chosen
        self.start = None  # my chips when this hand began

    # ---- engine hooks ----------------------------------------------------

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self.range_top = {}
        self.aggression = {}
        self.acted_preflop = set()
        self.facing_bet = False
        self.board_seen = 0
        self.shown = {}
        self.profile = None
        self.start = None

    def run(self, actions: list[PlayerAction], game_state: GameStateView) -> PlayerAction:
        try:
            return self._decide(actions, game_state)
        except Exception:  # a safe move beats a fault
            if game_state.call_amount == 0:
                return PlayerAction.call(self.player_id)
            return PlayerAction.fold(self.player_id)

    def _decide(self, actions, game_state):
        self._observe(actions, game_state)
        gs = game_state
        if self.start is None:
            self.start = gs.stacks[self.player_id] + gs.bets[self.player_id]
            self.profile = self._choose_profile(gs) if self.USE_PROFILES else 0
        self.fold_trust, bonus, margin = PROFILES[self.profile]
        equity = self._equity(gs)
        pot, call = gs.pot, gs.call_amount
        realise = min(1.0, self._realise(gs) + bonus)
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
                if ev > best_ev + margin * pot:  # must clearly beat the passive line
                    best, best_ev = PlayerAction.raise_to(self.player_id, total), ev
        return best

    def end_game(self, actions: list[PlayerAction], game_state: GameStateView):
        self._observe(actions, game_state)
        self._learn_from_showdown(game_state)
        if self.profile is not None and self.USE_PROFILES:
            arm = self.scores[self._rival(game_state)][self.profile]
            arm[0] += 1
            arm[1] += (game_state.stacks[self.player_id] - self.start) / game_state.big_blind

    # ---- which strategy --------------------------------------------------

    def _rival(self, gs):
        return next(gs.teams[p] for p in range(gs.num_players) if p != self.player_id)

    def _choose_profile(self, gs):
        """The usual upper confidence bound. Hands swing by about 10 big
        blinds, which sets how long a profile needs to prove itself."""
        arms = self.scores.setdefault(self._rival(gs), [[0, 0.0] for _ in PROFILES])
        total = sum(n for n, _ in arms) + 1
        best, best_value = 0, -math.inf
        for i, (n, won) in enumerate(arms):
            value = math.inf if n == 0 else won / n + 10 * math.sqrt(2 * math.log(total) / n)
            if value > best_value:
                best, best_value = i, value
        return best

    # ---- betting ---------------------------------------------------------

    def _realise(self, gs):
        """Share of equity kept: less with streets to come, and less out of
        position (heads-up the button acts last from the flop on)."""
        if gs.street == "river" or len(gs.active_players) <= 1:
            return 1.0  # no more betting: every card comes for free
        if gs.button == self.player_id:
            return 0.97
        return REALISE_PREFLOP_OOP if gs.street == "preflop" else REALISE_OOP

    def _raise_sizes(self, gs):
        low, high = gs.raise_range
        pot_after_call = gs.pot + gs.call_amount
        if gs.street == "preflop":
            base = gs.current_bet
            bb = gs.big_blind
            totals = [int(2.5 * bb), 3 * bb, high] if base == bb else [3 * base, high]
        else:
            totals = [gs.current_bet + int(f * pot_after_call) for f in (0.5, 0.75, 1.1)] + [high]
        return sorted({min(high, max(low, t)) for t in totals})

    def _raise_ev(self, gs, equity, total):
        me = self.player_id
        added = total - gs.bets[me]
        opponents = [p for p in gs.players_in_hand if p != me]
        behind = max(gs.stacks[p] + gs.bets[p] for p in opponents)
        extra = max(0, min(total, behind) - gs.current_bet)  # what they add to call
        added = min(added, gs.call_amount + extra)
        size = extra / max(1, gs.pot + gs.call_amount)
        fold = self._fold_equity(gs, opponents, size)
        reraise = min(1 - fold, self._reraise_chance(gs, opponents, total, behind))
        # Who calls a raise is stronger than who was there before it.
        called = equity * (1 - 0.12 * min(1.5, size) - 0.1 * (gs.call_amount > 0))
        return (fold * gs.pot
                + reraise * self._reraised_value(gs, equity, added, behind)
                + (1 - fold - reraise) * (called * (gs.pot + added + extra) - added))

    def _reraise_chance(self, gs, opponents, total, behind):
        if total >= behind:
            return 0.0  # that puts them all-in
        column = 3 if gs.street == "preflop" else 5
        return max(self._rate(self._stats(gs.teams[p]), column) for p in opponents)

    def _reraised_value(self, gs, equity, added, behind):
        """Chips from now if we are re-raised: fold and lose what we added,
        or get it all in, whichever is better."""
        me = self.player_id
        effective = min(gs.stacks[me] + gs.bets[me], behind)
        dead = gs.pot - sum(gs.bets)
        all_in = RERAISED_EQUITY * equity * (dead + 2 * effective) - (effective - gs.bets[me])
        return max(-added, all_in)

    def _fold_equity(self, gs, opponents, size):
        fold = 1.0
        for p in opponents:
            rate = self._rate(self._stats(gs.teams[p]), 1)
            if gs.bets[p] == gs.current_bet and gs.current_bet > gs.big_blind:
                rate *= 0.6  # they just bet: raising them is harder
            fold *= min(0.95, rate * (0.7 + 0.6 * min(1.0, size)))
        return min(0.97, fold * self.fold_trust * (0.85 + 0.3 * self.rng.random()))

    # ---- what the opponents have shown -----------------------------------

    @staticmethod
    def _new_stats():
        # hands: hands seen before the flop; opens: of those, first move a raise
        # faced: times they faced a bet; folds: of those, how many they folded
        # post_actions, post_bets: their moves after the flop, and the raises
        # pre_faced, pre_raises: preflop raises they faced, and re-raised
        # post_faced, post_raises: the same after the flop
        # shown, bluffs: showdowns after they bet, and how many of those were air
        # actions, raises: all their moves, and the raises among them
        return {"hands": 0, "opens": 0, "faced": 0, "folds": 0,
                "post_actions": 0, "post_bets": 0, "pre_faced": 0, "pre_raises": 0,
                "post_faced": 0, "post_raises": 0,
                "shown": 0, "bluffs": 0}

    def _stats(self, team):
        return self.opp_stats.setdefault(team, self._new_stats())

    def _observe(self, actions, gs):
        for a in actions:
            if a.player_id == DEALER:
                self.facing_bet = False
                self.board_seen += 1
                continue
            if a.kind == "reveal_card":
                self.shown.setdefault(a.player_id, []).append(a.card)
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            if a.player_id != self.player_id:
                self._note(a, gs)
            if a.kind == "raise":
                self.facing_bet = True

    def _note(self, a, gs):
        p = a.player_id
        s = self._stats(gs.teams[p])
        if self.facing_bet:
            s["faced"] += 1
            s["folds"] += a.kind == "fold"
        top = self.range_top.get(p, 1.0)
        if self.board_seen == 0:  # gs is already past these actions
            if p not in self.acted_preflop:
                self.acted_preflop.add(p)
                s["hands"] += 1
                s["opens"] += a.kind == "raise"
            if self.facing_bet:
                s["pre_faced"] += 1
                s["pre_raises"] += a.kind == "raise"
            if a.kind == "raise" and self.facing_bet:
                top = min(top, self._rate(s, 3))
            elif a.kind == "raise":
                top = min(top, self._rate(s, 0))
            elif a.kind == "call" and self.facing_bet:
                top = min(top, 1 - self._rate(s, 1))
            self.range_top[p] = max(0.03, top)
        else:
            s["post_actions"] += 1
            s["post_bets"] += a.kind == "raise"
            if self.facing_bet:
                s["post_faced"] += 1
                s["post_raises"] += a.kind == "raise"
            if a.kind == "raise":
                self.aggression[p] = self.aggression.get(p, 0) + 1

    def _learn_from_showdown(self, gs):
        board = {c.rank for c in gs.board}
        for p, cards in self.shown.items():
            if p == self.player_id or not self.aggression.get(p):
                continue
            s = self._stats(gs.teams[p])
            s["shown"] += 1
            # Air: no pocket pair, and neither card pairs the board.
            s["bluffs"] += cards[0].rank != cards[1].rank and not board & {c.rank for c in cards}

    def _bluff_share(self, s):
        """Chance a hand outside their betting range bets anyway."""
        return max(0.08, min(0.6, self._rate(s, 4)))

    def _prior(self, s):
        """What to expect of them: the rows of ARCHETYPES, weighted by how
        well each explains what they have done so far."""
        if not self.USE_MATRIX:
            return NEUTRAL
        logs = []
        for row in ARCHETYPES:
            total = 0.0
            for rate, (hits, tries, _) in zip(row, COLUMNS):
                total += s[hits] * math.log(rate) + (s[tries] - s[hits]) * math.log(1 - rate)
            logs.append(TYPE_CONFIDENCE * total)
        top = max(logs)
        weights = [math.exp(x - top) for x in logs]
        return [sum(w * row[i] for w, row in zip(weights, ARCHETYPES)) / sum(weights)
                for i in range(len(COLUMNS))]

    def _rate(self, s, column):
        """How often they do what this column of ARCHETYPES counts: what they
        have done so far, starting from the prior."""
        hits, tries, weight = COLUMNS[column]
        return (s[hits] + weight * self._prior(s)[column]) / (s[tries] + weight)

    # ---- equity ----------------------------------------------------------

    def _candidates(self, gs, p, unseen, board):
        """The hands p may hold, as (a, b) pairs: inside their estimated
        starting range, and if they bet after the flop, mostly the best of
        those: someone who bets a share f of the time bets about their top
        f. A few others stay in as bluffs: a range is an estimate, not a fact."""
        rng = self.rng
        top = self.range_top.get(p, 1.0)
        hands = []
        for i, a in enumerate(unseen):
            for b in unseen[i + 1:]:
                if _percentile(a, b) <= top or rng.random() < 0.05:
                    hands.append((a, b))
        bets = self.aggression.get(p, 0)
        if not bets or not board or len(hands) < 2:
            return hands
        s = self._stats(gs.teams[p])
        share = self._rate(s, 2) ** bets
        hands.sort(key=lambda h: _score([h[0], h[1]] + board), reverse=True)
        keep = max(1, int(len(hands) * share))
        return hands[:keep] + [h for h in hands[keep:]
                                  if rng.random() < self._bluff_share(s) ** bets]

    def _equity(self, gs):
        """Share of the pot won against hands drawn from each opponent's
        estimated range, not against random hands."""
        rng = self.rng
        hand = [_to_int(c) for c in self.hand]
        board = [_to_int(c) for c in gs.board]
        known = set(hand + board)
        unseen = [c for c in range(52) if c not in known]
        missing = 5 - len(board)
        ranges = [self._candidates(gs, p, unseen, board)
                  for p in gs.players_in_hand if p != self.player_id]
        extra = missing + 2 * len(ranges)
        won = counted = 0.0
        for _ in range(SAMPLES):
            holes = [r[int(rng.random() * len(r))] for r in ranges]
            taken = [c for pair in holes for c in pair]
            if len(set(taken)) != len(taken):
                continue
            full_board = board + [c for c in rng.sample(unseen, extra) if c not in taken][:missing]
            mine = _score(hand + full_board)
            best, ties = True, 1
            for a, b in holes:
                theirs = _score([a, b] + full_board)
                if theirs > mine:
                    best = False
                    break
                if theirs == mine:
                    ties += 1
            counted += 1
            if best:
                won += 1.0 / ties
        return won / counted if counted else 0.5


class NoReraise(Model):
    def _reraise_chance(self, gs, opponents, total, behind):
        return 0.0
