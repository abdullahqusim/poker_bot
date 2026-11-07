from card_utils import Card
from game_state import GameStateView
from player_action import DEALER, PlayerAction

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
    """Plans for 25 big blinds heads-up: never limps, bets small and often on
    the flop, sizes value bets to get the stacks in by the river, and has a
    floor on how much it folds."""

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.rng = _Rng(2024)
        self.opp = {}  # team -> counters, see _counters
        self.top = 1.0  # share of starting hands the opponent still holds
        self.bets = 0  # their bets after the flop, this hand
        self.facing = False  # a raise is open on this street
        self.raised_last = False  # the last raise of the hand was ours
        self.mine = False  # the open raise on this street is ours
        self.seen = 0  # board cards replayed
        self.shown = {}
        self.first_pre = True  # they have not moved before the flop yet

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self.top = 1.0
        self.bets = 0
        self.facing = False
        self.raised_last = False
        self.mine = False
        self.seen = 0
        self.shown = {}
        self.first_pre = True

    def run(self, actions: list[PlayerAction], game_state: GameStateView) -> PlayerAction:
        try:
            self._observe(actions, game_state)
            if game_state.street == "preflop":
                return self._preflop(game_state)
            return self._postflop(game_state)
        except Exception:  # a safe move beats a fault
            if game_state.call_amount == 0:
                return PlayerAction.call(self.player_id)
            return PlayerAction.fold(self.player_id)

    def end_game(self, actions: list[PlayerAction], game_state: GameStateView):
        self._observe(actions, game_state)
        board = {c.rank for c in game_state.board}
        for p, cards in self.shown.items():
            if p != self.player_id and self.bets:
                c = self._counters(game_state.teams[p])
                c["shown"] += 1
                c["air"] += cards[0].rank != cards[1].rank and not board & {x.rank for x in cards}

    # ---- what they do ----------------------------------------------------

    @staticmethod
    def _rate(c, hits, tries, prior, weight):
        return (c[hits] + prior * weight) / (c[tries] + weight)

    def _counters(self, team):
        return self.opp.setdefault(team, {
            "pre": 0, "opens": 0,            # their first move, raises among them
            "vs_raise": 0, "pre_folds": 0, "reraises": 0,  # facing our preflop raise
            "vs_bet": 0, "post_folds": 0, "post_raises": 0,  # facing our bet
            "moves": 0, "bets": 0,           # after the flop
            "shown": 0, "air": 0,            # showdowns after betting, and bluffs
        })

    def _observe(self, actions, gs):
        for a in actions:
            if a.player_id == DEALER:
                self.seen += 1
                self.facing = self.mine = False
                continue
            if a.kind == "reveal_card":
                self.shown.setdefault(a.player_id, []).append(a.card)
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            if a.player_id != self.player_id:
                self._note(a, gs)
            if a.kind == "raise":
                self.facing = True
                self.mine = a.player_id == self.player_id
                self.raised_last = self.mine

    def _note(self, a, gs):
        c = self._counters(gs.teams[a.player_id])
        raised = a.kind == "raise"
        if self.seen == 0:
            if self.first_pre:
                self.first_pre = False
                c["pre"] += 1
                c["opens"] += raised and not self.facing
            if self.facing:
                c["vs_raise"] += 1
                c["pre_folds"] += a.kind == "fold"
                c["reraises"] += raised
            if raised and self.facing:
                self.top = min(self.top, self._rate(c, "reraises", "vs_raise", 0.15, 6) + 0.03)
            elif raised:
                self.top = min(self.top, self._rate(c, "opens", "pre", 0.7, 6))
            elif a.kind == "call" and self.facing:
                self.top = min(self.top, 1 - self._rate(c, "pre_folds", "vs_raise", 0.3, 6))
            return
        c["moves"] += 1
        c["bets"] += raised
        if self.bets is not None and raised:
            self.bets += 1
        if self.facing:
            c["vs_bet"] += 1
            c["post_folds"] += a.kind == "fold"
            c["post_raises"] += raised

    # ---- equity against what they can hold --------------------------------

    def _equity(self, gs, samples=500):
        rng = self.rng
        hand = [_to_int(c) for c in self.hand]
        board = [_to_int(c) for c in gs.board]
        known = set(hand + board)
        unseen = [c for c in range(52) if c not in known]
        c = self._counters(self._rival(gs))
        hands = []
        for i, a in enumerate(unseen):
            for b in unseen[i + 1:]:
                if _percentile(a, b) <= self.top or rng.random() < 0.04:
                    hands.append((a, b))
        if self.bets and board and len(hands) > 1:
            share = self._rate(c, "bets", "moves", 0.35, 12) ** self.bets
            bluff = max(0.2, self._rate(c, "air", "shown", 0.25, 6)) ** self.bets
            hands.sort(key=lambda h: _score([h[0], h[1]] + board), reverse=True)
            keep = max(1, int(len(hands) * share))
            hands = hands[:keep] + [h for h in hands[keep:] if rng.random() < bluff]
        missing = 5 - len(board)
        won = counted = 0.0
        for _ in range(samples):
            pair = hands[int(rng.random() * len(hands))]
            extra = [x for x in rng.sample(unseen, missing + 2) if x not in pair][:missing]
            full = board + extra
            mine, theirs = _score(hand + full), _score(list(pair) + full)
            counted += 1
            won += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
        return won / counted

    def _rival(self, gs):
        return next(gs.teams[p] for p in range(gs.num_players) if p != self.player_id)

    def _effective(self, gs):
        me = self.player_id
        return min(gs.stacks[me] + gs.bets[me],
                   max(gs.stacks[p] + gs.bets[p] for p in gs.players_in_hand if p != me))

    # ---- moves -------------------------------------------------------------

    def _call(self):
        return PlayerAction.call(self.player_id)

    def _fold(self, gs):
        return self._call() if gs.call_amount == 0 else PlayerAction.fold(self.player_id)

    def _raise(self, gs, total):
        if not gs.can_raise:
            return self._call()
        low, high = gs.raise_range
        return PlayerAction.raise_to(self.player_id, min(high, max(low, int(total))))

    def _bet(self, gs, fraction):
        return self._raise(gs, gs.current_bet + fraction * (gs.pot + gs.call_amount))

    def _preflop(self, gs):
        a, b = [_to_int(c) for c in self.hand]
        pct = _percentile(a, b)
        bb, r = gs.big_blind, self.rng.random()
        c = self._counters(self._rival(gs))
        depth = self._effective(gs) / bb
        folds = self._rate(c, "pre_folds", "vs_raise", 0.3, 6)
        reraises = self._rate(c, "reraises", "vs_raise", 0.15, 6)

        if gs.current_bet <= bb:
            if gs.button == self.player_id:  # first in: raise or fold, never limp
                if depth <= 10:
                    return self._raise(gs, gs.max_raise_to) if pct < 0.6 else self._fold(gs)
                limit = 0.85 if reraises < 0.2 else 0.7 if reraises < 0.35 else 0.55
                if folds > 0.6:
                    limit = 1.0
                return self._raise(gs, 2.5 * bb) if pct <= limit else self._fold(gs)
            return self._raise(gs, 3.5 * bb) if pct < 0.3 else self._call()  # they limped

        equity = self._equity(gs, 400)
        odds = gs.pot_odds
        shoved = gs.call_amount >= gs.stacks[self.player_id] or not gs.can_raise
        if shoved:
            return self._call() if equity > odds else self._fold(gs)
        opened = gs.current_bet <= 4 * bb
        if opened:  # they opened: re-raise, defend or fold
            if pct < 0.1 or (0.3 < pct < 0.42 and folds > 0.45 and r < 0.35):
                return self._raise(gs, gs.max_raise_to if depth <= 15 else 3 * gs.current_bet)
            return self._call() if equity * 0.85 > odds else self._fold(gs)
        # They re-raised us: get it in, call or let it go.
        if equity > 0.58:
            return self._raise(gs, gs.max_raise_to)
        return self._call() if equity * 0.9 > odds else self._fold(gs)

    def _value_size(self, gs):
        """The share of the pot that gets the stacks in by the river."""
        streets = {"flop": 3, "turn": 2, "river": 1}[gs.street]
        behind = self._effective(gs) - gs.current_bet
        ratio = behind / max(1, gs.pot + gs.call_amount)
        return max(0.33, min(1.0, ((1 + 2 * ratio) ** (1 / streets) - 1) / 2))

    def _postflop(self, gs):
        c = self._counters(self._rival(gs))
        equity = self._equity(gs)
        r = self.rng.random()
        folds = self._rate(c, "post_folds", "vs_bet", 0.42, 10)
        pot = gs.pot

        if gs.call_amount == 0:
            if equity > 0.62:
                return self._bet(gs, self._value_size(gs))
            if equity > 0.54 and r < 0.7:
                return self._bet(gs, 0.4)
            if gs.street == "flop" and self.raised_last and r < 0.3 + folds:
                return self._bet(gs, 0.33)  # small and often
            if gs.street == "turn" and self.raised_last and equity > 0.38 and r < folds:
                return self._bet(gs, 0.55)
            if gs.street == "river" and equity < 0.3 and folds > 0.4 and r < folds - 0.15:
                return self._bet(gs, 0.75)
            return self._call()

        odds = gs.pot_odds
        if equity > 0.8:
            return self._bet(gs, self._value_size(gs) * 1.5)
        if gs.street == "flop" and 0.42 < equity < 0.58 and folds > 0.45 and r < 0.12:
            return self._bet(gs, 1.0)  # a semi-bluff raise
        # The floor: facing a frequent bettor, call a little lighter than the price.
        often = self._rate(c, "bets", "moves", 0.35, 12) > 0.45
        needed = odds - (0.05 if often else 0.0) + (0.03 if gs.street != "river" else 0.0)
        return self._call() if equity > needed else self._fold(gs)
