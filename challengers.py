"""
Standalone suite of 6 distinct, tournament-grade No-Limit Texas Hold'em bots
plus an adaptive Meta-Ensemble bot (EnsembleBot / Model).

Completely independent of existing bots in the repository.

Bots included:
1. GTOBot ("gto_apex")          - Game-Theoretic Optimal: Minimum Defense Frequency
                                  (MDF), geometric multi-street sizing, balanced
                                  value-to-bluff ratios, blocker river bluffs, and
                                  stack-depth equilibrium preflop charts.
2. BayesianBot ("bayesian")     - Explicit 169-hand Bayesian range tracker per
                                  opponent. Updates posterior hand weights on every
                                  action, calibrates from showdown reveals, and
                                  maximizes EV across candidate bet sizes.
3. SharkBot ("hud_shark")       - Persistent per-team statistical HUD & exploitative
                                  counter-strategy engine (Nit, Station, Maniac,
                                  Shover, LAG, TAG adaptations).
4. PressureBot ("lag_pressure") - Board-texture, range-advantage vs nut-advantage,
                                  positional float, delayed probe, and blocker
                                  multi-barrel specialist.
5. FortressBot ("fortress")     - Short-stack Nash push/fold + multi-way pot control
                                  + monster slowplay/check-raise trapper.
6. RolloutBot ("mc_rollout")    - Fast bitwise rollout simulation & Effective Hand
                                  Strength (EHS = HS*(1-NP) + (1-HS)*PP) engine.
7. EnsembleBot ("ensemble")     - Contextual Hedge/Bandit meta-controller combining
                                  all 6 specialists with per-opponent weight learning.
"""

import collections
import math
import random

from card_utils import Card, RANKS, SUITS
from player_action import DEALER, PlayerAction


# ---------------------------------------------------------------------------
# 1. ULTRA-FAST BITWISE 5-TO-7 CARD EVALUATOR (Precomputed at import time)
# ---------------------------------------------------------------------------
# Internal card encoding: int 0..51 where rank = c >> 2 (0..12 for 2..A),
# suit = c & 3 (0..3 for C, D, H, S).

_POPCOUNT = [0] * 8192
_HIGH1 = [0] * 8192
_TOP2 = [0] * 8192
_TOP2_MASK = [0] * 8192
_TOP3 = [0] * 8192
_TOP5 = [0] * 8192
_STRAIGHT = [0] * 8192


def _init_eval_tables():
    for mask in range(8192):
        bits = [r for r in range(12, -1, -1) if (mask >> r) & 1]
        _POPCOUNT[mask] = len(bits)
        if bits:
            _HIGH1[mask] = bits[0]
        if len(bits) >= 2:
            _TOP2[mask] = (bits[0] << 4) | bits[1]
            _TOP2_MASK[mask] = (1 << bits[0]) | (1 << bits[1])
        if len(bits) >= 3:
            _TOP3[mask] = (bits[0] << 8) | (bits[1] << 4) | bits[2]
        if len(bits) >= 5:
            _TOP5[mask] = (
                (bits[0] << 16)
                | (bits[1] << 12)
                | (bits[2] << 8)
                | (bits[3] << 4)
                | (bits[4]
            ))
        st_high = 0
        for high_r in range(12, 3, -1):
            st_mask = 0x1F << (high_r - 4)
            if (mask & st_mask) == st_mask:
                st_high = high_r
                break
        if st_high == 0 and (mask & 0x100F) == 0x100F:
            st_high = 3
        _STRAIGHT[mask] = st_high


_init_eval_tables()


def _card_to_int(card: Card) -> int:
    return (RANKS.index(card.rank) << 2) | SUITS.index(card.suit)


def _fast_eval(cards) -> int:
    """
    Evaluates 5 to 7 integer cards (0..51) into a single integer rank.
    Strictly order-isomorphic to card_utils.evaluate(cards).
    Top bits (score >> 20) equal the category index 0..8:
    0=high_card, 1=pair, 2=two_pair, 3=trips, 4=straight,
    5=flush, 6=full_house, 7=quads, 8=straight_flush.
    """
    s0 = s1 = s2 = s3 = 0
    m1 = m2 = m3 = m4 = 0
    for c in cards:
        r = c >> 2
        b = 1 << r
        st = c & 3
        if st == 0:
            s0 |= b
        elif st == 1:
            s1 |= b
        elif st == 2:
            s2 |= b
        else:
            s3 |= b
        m4 |= m3 & b
        m3 |= m2 & b
        m2 |= m1 & b
        m1 |= b

    if _POPCOUNT[s0] >= 5:
        sf = _STRAIGHT[s0]
        return ((8 << 20) | sf) if sf else ((5 << 20) | _TOP5[s0])
    if _POPCOUNT[s1] >= 5:
        sf = _STRAIGHT[s1]
        return ((8 << 20) | sf) if sf else ((5 << 20) | _TOP5[s1])
    if _POPCOUNT[s2] >= 5:
        sf = _STRAIGHT[s2]
        return ((8 << 20) | sf) if sf else ((5 << 20) | _TOP5[s2])
    if _POPCOUNT[s3] >= 5:
        sf = _STRAIGHT[s3]
        return ((8 << 20) | sf) if sf else ((5 << 20) | _TOP5[s3])

    if m4:
        qr = _HIGH1[m4]
        kicker = _HIGH1[m1 ^ (1 << qr)]
        return (7 << 20) | (qr << 16) | kicker

    if m3:
        tr = _HIGH1[m3]
        rem_pairs = m2 ^ (1 << tr)
        if rem_pairs:
            pr = _HIGH1[rem_pairs]
            return (6 << 20) | (tr << 16) | (pr << 12)
        st = _STRAIGHT[m1]
        if st:
            return (4 << 20) | st
        kickers = _TOP2[m1 ^ (1 << tr)]
        return (3 << 20) | (tr << 16) | kickers

    st = _STRAIGHT[m1]
    if st:
        return (4 << 20) | st

    if m2:
        if m2 & (m2 - 1):
            top2_p = _TOP2[m2]
            top2_m = _TOP2_MASK[m2]
            kicker = _HIGH1[m1 ^ top2_m]
            return (2 << 20) | (top2_p << 8) | kicker
        pr = _HIGH1[m2]
        kickers = _TOP3[m1 ^ (1 << pr)]
        return (1 << 20) | (pr << 16) | kickers

    return _TOP5[m1]


# ---------------------------------------------------------------------------
# 2. PREFLOP 169 CANONICAL HAND ENGINE & NASH EQUILIBRIUM TABLES
# ---------------------------------------------------------------------------

_PREFLOP_HU_EQUITY = [0.5] * 169
_PREFLOP_6MAX_SCORE = [0.5] * 169
_NASH_PUSH_BB = [1.0] * 169
_NASH_CALL_BB = [1.0] * 169
_COMBO_COUNTS = [6] * 169
_CANONICAL_REPS = []


def _canonical_idx(c1: int, c2: int) -> int:
    r1, s1 = c1 >> 2, c1 & 3
    r2, s2 = c2 >> 2, c2 & 3
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return r1
    base = (r1 * (r1 - 1)) // 2 + r2
    return (13 + base) if (s1 == s2) else (91 + base)


def _init_preflop_tables():
    reps = [None] * 169
    for r1 in range(13):
        reps[r1] = (r1, r1, False)
        _COMBO_COUNTS[r1] = 6
        for r2 in range(r1):
            base = (r1 * (r1 - 1)) // 2 + r2
            reps[13 + base] = (r1, r2, True)
            reps[91 + base] = (r1, r2, False)
            _COMBO_COUNTS[13 + base] = 4
            _COMBO_COUNTS[91 + base] = 12
    _CANONICAL_REPS.extend(reps)

    for idx, (r1, r2, suited) in enumerate(reps):
        v1, v2 = r1 + 2, r2 + 2
        if r1 == r2:
            hu_eq = 0.503 + 0.0291 * r1 + 0.00015 * (r1 ** 2)
            mw_score = 0.54 + 0.035 * r1
            push_bb = 25.0
            call_bb = 7.5 + 1.85 * r1 + 0.12 * (r1 ** 2)
        else:
            gap = r1 - r2
            high_factor = (v1 - 2) / 12.0
            low_factor = (v2 - 2) / 12.0
            suit_bonus = 0.028 if suited else 0.0
            conn_bonus = (
                0.018 if gap == 1 else
                0.012 if gap == 2 else
                0.006 if gap == 3 else
                0.002 if gap == 4 else 0.0
            )
            if r1 == 12 and r2 <= 3:
                conn_bonus += 0.010
            hu_eq = (
                0.315
                + 0.215 * high_factor
                + 0.105 * low_factor
                + suit_bonus
                + conn_bonus
            )
            hu_eq = max(0.32, min(0.675, hu_eq))

            mw_suit = 0.055 if suited else 0.0
            mw_conn = conn_bonus * 1.8
            mw_score = (
                0.22
                + 0.28 * (high_factor ** 1.25)
                + 0.16 * (low_factor ** 1.15)
                + mw_suit
                + mw_conn
            )
            if r1 == 12 and suited:
                mw_score += 0.045

            if r1 == 12:
                push_bb = (25.0 if suited else (11.0 + 1.4 * r2))
                call_bb = (10.5 + 1.3 * r2) if suited else (7.0 + 1.25 * r2)
            elif r1 == 11:
                push_bb = (11.0 + 1.5 * r2) if suited else (4.5 + 1.35 * r2)
                call_bb = (6.0 + 1.1 * r2) if suited else (3.5 + 1.0 * r2)
            elif r1 == 10:
                push_bb = (7.0 + 1.4 * r2) if suited else (2.8 + 1.1 * r2)
                call_bb = (4.0 + 0.9 * r2) if suited else (2.2 + 0.75 * r2)
            else:
                push_bb = max(1.2, (hu_eq - 0.34) * 42.0 + (3.5 if suited else 0.0) + conn_bonus * 120.0)
                call_bb = max(1.0, (hu_eq - 0.38) * 32.0 + (1.5 if suited else 0.0))

        _PREFLOP_HU_EQUITY[idx] = hu_eq
        _PREFLOP_6MAX_SCORE[idx] = min(0.99, mw_score)
        _NASH_PUSH_BB[idx] = min(25.0, push_bb)
        _NASH_CALL_BB[idx] = min(25.0, call_bb)


_init_preflop_tables()

_SORTED_169_BY_MW = sorted(range(169), key=lambda i: _PREFLOP_6MAX_SCORE[i], reverse=True)
_SORTED_169_BY_HU = sorted(range(169), key=lambda i: _PREFLOP_HU_EQUITY[i], reverse=True)
_MW_PERCENTILE = [0.0] * 169
_HU_PERCENTILE = [0.0] * 169


def _init_percentiles():
    cum = 0
    for i in _SORTED_169_BY_MW:
        _MW_PERCENTILE[i] = 1.0 - (cum / 1326.0)
        cum += _COMBO_COUNTS[i]
    cum = 0
    for i in _SORTED_169_BY_HU:
        _HU_PERCENTILE[i] = 1.0 - (cum / 1326.0)
        cum += _COMBO_COUNTS[i]


_init_percentiles()


# ---------------------------------------------------------------------------
# 3. BOARD TEXTURE, TRUE HAND TIER & ACTION-CONDITIONED ROLLOUT ENGINE
# ---------------------------------------------------------------------------

class HandAnalysis:
    """Comprehensive structural & action-conditioned equity evaluation."""

    __slots__ = (
        "street", "hole_ints", "board_ints", "canon_idx",
        "hu_preflop_eq", "mw_preflop_score", "hu_percentile", "mw_percentile",
        "nash_push_bb", "nash_call_bb",
        "cat", "real_cat", "board_cat", "hand_score", "board_score",
        "is_playing_board", "hole_cards_used",
        "pair_type", "is_overpair", "is_top_pair", "top_pair_kicker",
        "is_set", "is_trips", "is_two_pair_both_hole",
        "flush_draw", "nut_flush_draw", "backdoor_flush_draw",
        "straight_outs", "overcards", "has_nut_flush_blocker",
        "board_wetness", "board_paired", "board_trips", "board_monotone",
        "board_two_tone", "board_four_flush", "board_connected", "board_high_rank",
        "raw_equity", "multiway_equity", "vs_bet_equity", "vs_check_equity",
        "pp", "np", "ehs",
    )

    def __init__(self, hand: list[Card], board: tuple[Card, ...], num_opponents: int,
                 rng: random.Random, num_samples: int = 160):
        c1, c2 = _card_to_int(hand[0]), _card_to_int(hand[1])
        self.hole_ints = (c1, c2)
        self.board_ints = tuple(_card_to_int(c) for c in board)
        self.canon_idx = _canonical_idx(c1, c2)
        self.hu_preflop_eq = _PREFLOP_HU_EQUITY[self.canon_idx]
        self.mw_preflop_score = _PREFLOP_6MAX_SCORE[self.canon_idx]
        self.hu_percentile = _HU_PERCENTILE[self.canon_idx]
        self.mw_percentile = _MW_PERCENTILE[self.canon_idx]
        self.nash_push_bb = _NASH_PUSH_BB[self.canon_idx]
        self.nash_call_bb = _NASH_CALL_BB[self.canon_idx]

        n_board = len(self.board_ints)
        if n_board == 0:
            self.street = "preflop"
            self.cat = 1 if (c1 >> 2) == (c2 >> 2) else 0
            self.real_cat = self.cat
            self.board_cat = 0
            self.hand_score = 0
            self.board_score = 0
            self.is_playing_board = False
            self.hole_cards_used = 2
            self.pair_type = "pocket_pair" if self.cat == 1 else "unpaired"
            self.is_overpair = False
            self.is_top_pair = False
            self.top_pair_kicker = max(c1 >> 2, c2 >> 2)
            self.is_set = False
            self.is_trips = False
            self.is_two_pair_both_hole = False
            self.flush_draw = False
            self.nut_flush_draw = False
            self.backdoor_flush_draw = (c1 & 3) == (c2 & 3)
            self.straight_outs = 0
            self.overcards = 0
            self.has_nut_flush_blocker = False
            self.board_wetness = 0.0
            self.board_paired = False
            self.board_trips = False
            self.board_monotone = False
            self.board_two_tone = False
            self.board_four_flush = False
            self.board_connected = False
            self.board_high_rank = max(c1 >> 2, c2 >> 2)
            eq = self.hu_preflop_eq
            mw_eq = eq ** max(1, num_opponents)
            if num_opponents > 1:
                playability_adj = (self.mw_preflop_score - 0.5) * 0.06
                mw_eq = max(0.05, min(0.85, mw_eq + playability_adj))
            self.raw_equity = eq
            self.multiway_equity = mw_eq
            self.vs_check_equity = mw_eq
            self.vs_bet_equity = max(0.05, mw_eq - 0.05)
            self.pp = 0.12 if ((c1 & 3) == (c2 & 3) or abs((c1 >> 2) - (c2 >> 2)) <= 2) else 0.05
            self.np = 0.08
            self.ehs = mw_eq
            return

        self.street = "flop" if n_board == 3 else ("turn" if n_board == 4 else "river")
        all_cards = self.hole_ints + self.board_ints
        self.hand_score = _fast_eval(all_cards)
        self.cat = self.hand_score >> 20

        b_ranks = sorted((c >> 2 for c in self.board_ints), reverse=True)
        b_suits = [c & 3 for c in self.board_ints]
        self.board_high_rank = b_ranks[0]
        b_rank_counts = collections.Counter(b_ranks)
        self.board_paired = any(cnt >= 2 for cnt in b_rank_counts.values())
        self.board_trips = any(cnt >= 3 for cnt in b_rank_counts.values())
        board_num_pairs = sum(1 for cnt in b_rank_counts.values() if cnt >= 2)

        suit_counts = collections.Counter(b_suits)
        max_suit, max_suit_cnt = suit_counts.most_common(1)[0]
        self.board_monotone = max_suit_cnt >= 3 and n_board == 3
        self.board_two_tone = max_suit_cnt == 2 and n_board <= 4
        self.board_four_flush = max_suit_cnt >= 4

        unique_b = sorted(set(b_ranks))
        close_gaps = sum(
            1 for i in range(len(unique_b))
            for j in range(i + 1, len(unique_b))
            if unique_b[j] - unique_b[i] <= 4
        )
        self.board_connected = close_gaps >= 3
        wetness = 0.0
        if max_suit_cnt >= 3:
            wetness += 0.45
        elif max_suit_cnt == 2:
            wetness += 0.20
        wetness += min(0.45, close_gaps * 0.12)
        if self.board_paired:
            wetness -= 0.10
        self.board_wetness = max(0.0, min(1.0, wetness))

        if n_board == 5:
            self.board_score = _fast_eval(self.board_ints)
            self.board_cat = self.board_score >> 20
            self.is_playing_board = (self.hand_score == self.board_score)
        else:
            self.board_score = 0
            self.board_cat = 3 if self.board_trips else (2 if board_num_pairs >= 2 else (1 if self.board_paired else 0))
            self.is_playing_board = False

        hr1, hr2 = max(c1 >> 2, c2 >> 2), min(c1 >> 2, c2 >> 2)
        hs1, hs2 = c1 & 3, c2 & 3
        b_set = set(b_ranks)
        self.overcards = int(hr1 > b_ranks[0]) + int(hr2 > b_ranks[0])

        self.is_overpair = (hr1 == hr2) and (hr1 > b_ranks[0])
        self.is_set = (hr1 == hr2) and (hr1 in b_set)
        self.is_trips = (hr1 != hr2) and (
            b_rank_counts.get(hr1, 0) >= 2 or b_rank_counts.get(hr2, 0) >= 2
        )
        # True two-pair using both hole cards on distinct non-paired board ranks
        self.is_two_pair_both_hole = (
            (hr1 != hr2)
            and (b_rank_counts.get(hr1, 0) == 1)
            and (b_rank_counts.get(hr2, 0) == 1)
        )

        # Unpaired board ranks (ranks that appear exactly once on the board)
        single_b_ranks = sorted((r for r, cnt in b_rank_counts.items() if cnt == 1), reverse=True)
        highest_single = single_b_ranks[0] if single_b_ranks else b_ranks[0]

        if self.is_set:
            self.is_top_pair = False
            self.top_pair_kicker = hr1
            self.pair_type = "set"
            self.hole_cards_used = 2
        elif self.is_trips:
            self.is_top_pair = False
            self.top_pair_kicker = hr2 if b_rank_counts.get(hr1, 0) >= 2 else hr1
            self.pair_type = "trips"
            self.hole_cards_used = 1
        elif self.is_two_pair_both_hole:
            self.is_top_pair = (hr1 == b_ranks[0])
            self.top_pair_kicker = hr2
            self.pair_type = "two_pair"
            self.hole_cards_used = 2
        elif self.is_overpair:
            self.is_top_pair = False
            self.top_pair_kicker = hr1
            self.pair_type = "overpair"
            self.hole_cards_used = 2
        elif hr1 == highest_single or hr2 == highest_single:
            # Paired the highest unpaired board card
            paired_r = highest_single
            kicker_r = hr2 if hr1 == paired_r else hr1
            # Only true top pair if no higher paired rank sits on the board,
            # or if the board is paired above it, it's top available pair
            self.is_top_pair = (paired_r == b_ranks[0])
            self.top_pair_kicker = kicker_r
            self.pair_type = "top_pair" if self.is_top_pair else "second_pair"
            self.hole_cards_used = 1
        elif hr1 == hr2:
            self.is_top_pair = False
            self.top_pair_kicker = hr1
            if hr1 > b_ranks[-1]:
                self.pair_type = "mid_pocket_pair"
            else:
                self.pair_type = "underpair"
            self.hole_cards_used = 2
        elif hr1 in b_set or hr2 in b_set:
            self.is_top_pair = False
            paired_r = hr1 if hr1 in b_set else hr2
            self.top_pair_kicker = hr2 if hr1 in b_set else hr1
            if len(unique_b) >= 2 and paired_r >= unique_b[-2]:
                self.pair_type = "second_pair"
            else:
                self.pair_type = "bottom_pair"
            self.hole_cards_used = 1
        else:
            self.is_top_pair = False
            self.top_pair_kicker = hr1
            self.pair_type = "unpaired"
            self.hole_cards_used = 0

        # Compute `real_cat`: true hole-card-adjusted category (never fooled by board pairs!)
        if self.is_playing_board:
            self.real_cat = 0
        elif self.cat <= 1:
            self.real_cat = 0 if self.pair_type == "unpaired" else 1
        elif self.cat == 2:
            if self.is_two_pair_both_hole or (self.is_overpair and self.board_paired):
                self.real_cat = 2
            elif self.pair_type == "unpaired":
                self.real_cat = 0
            else:
                # Board has a pair and we have a single pair (top/second/bottom/underpair)
                self.real_cat = 1
        elif self.cat == 3:
            self.real_cat = 0 if (self.board_trips and self.pair_type == "unpaired") else 3
        elif self.cat == 4:
            # Check if 4-card straight is on board and we only hold bottom card
            board_mask = 0
            for r in b_ranks:
                board_mask |= (1 << r)
            st_rank = self.hand_score & 0xF
            # If any single rank >= st_rank completes a higher straight on board_mask, be cautious
            higher_st_possible = any(
                _STRAIGHT[board_mask | (1 << tr)] > st_rank for tr in range(13)
            )
            self.real_cat = 2 if (higher_st_possible and hr1 < st_rank) else 4
        elif self.cat == 5:
            if self.board_four_flush:
                hole_flush_ranks = [c >> 2 for c in self.hole_ints if (c & 3) == max_suit]
                best_hf = max(hole_flush_ranks) if hole_flush_ranks else -1
                self.real_cat = 5 if best_hf >= 9 else (2 if best_hf >= 6 else 1)
            else:
                self.real_cat = 5
        else:
            self.real_cat = self.cat

        # Draw & Blocker detection
        all_suits = b_suits + [hs1, hs2]
        total_suit_counts = collections.Counter(all_suits)
        best_s, best_s_cnt = total_suit_counts.most_common(1)[0]
        hole_in_best_s = [(c >> 2) for c in self.hole_ints if (c & 3) == best_s]
        self.flush_draw = (n_board < 5) and (best_s_cnt == 4) and (len(hole_in_best_s) >= 1)
        self.nut_flush_draw = self.flush_draw and (
            12 in hole_in_best_s or (11 in hole_in_best_s and ((12 << 2) | best_s) in self.board_ints)
        )
        self.backdoor_flush_draw = (n_board == 3) and (best_s_cnt == 3) and (len(hole_in_best_s) >= 1)

        if max_suit_cnt >= 3:
            ace_of_flush = (12 << 2) | max_suit
            king_of_flush = (11 << 2) | max_suit
            self.has_nut_flush_blocker = (
                ace_of_flush in self.hole_ints
                or (ace_of_flush in self.board_ints and king_of_flush in self.hole_ints)
            )
        else:
            self.has_nut_flush_blocker = False

        st_outs = 0
        if n_board < 5 and self.cat < 4:
            hole_mask = (1 << hr1) | (1 << hr2)
            board_mask = 0
            for r in b_ranks:
                board_mask |= (1 << r)
            comb_mask = hole_mask | board_mask
            if not _STRAIGHT[board_mask]:
                for test_r in range(13):
                    tb = 1 << test_r
                    if not (comb_mask & tb):
                        new_st = _STRAIGHT[comb_mask | tb]
                        if new_st and new_st != _STRAIGHT[board_mask | tb]:
                            st_outs += 4
        self.straight_outs = st_outs

        self._compute_rollout_metrics(num_opponents, b_set, b_ranks[0], rng, num_samples)

    def _compute_rollout_metrics(self, num_opponents: int, b_set: set[int],
                                 b_max_r: int, rng: random.Random, num_samples: int):
        dead = set(self.hole_ints) | set(self.board_ints)
        rem_deck = [c for c in range(52) if c not in dead]
        n_board = len(self.board_ints)
        to_deal = 5 - n_board
        n_opp = max(1, min(5, num_opponents))

        hu_wins = 0.0
        mw_wins = 0.0
        bet_w_total = 0.0
        bet_w_wins = 0.0
        check_w_total = 0.0
        check_w_wins = 0.0
        ahead_now = 0
        ahead_lost = 0.0
        behind_now = 0
        behind_won = 0.0

        my_now_score = self.hand_score
        b_list = list(self.board_ints)
        h0, h1 = self.hole_ints
        board_cat_now = self.board_cat

        # Bluff/weak-pair betting weights depend on street (rivers have far fewer bluffs/thin bets)
        weak_pair_bet_w = 0.25 if n_board == 3 else (0.12 if n_board == 4 else 0.06)
        air_bet_w = 0.20 if n_board == 3 else (0.11 if n_board == 4 else 0.05)

        need_cards = to_deal + 2 * n_opp
        for _ in range(num_samples):
            drawn = rng.sample(rem_deck, need_cards)
            runout = drawn[:to_deal]
            full_board = b_list + runout if to_deal > 0 else b_list
            my_final = _fast_eval((h0, h1, *full_board)) if to_deal > 0 else my_now_score

            o0, o1 = drawn[to_deal], drawn[to_deal + 1]
            opp1_now = _fast_eval((o0, o1, *b_list))
            opp1_final = _fast_eval((o0, o1, *full_board)) if to_deal > 0 else opp1_now

            # Preflop playability prior for opponent hand
            pf_w = _PREFLOP_HU_EQUITY[_canonical_idx(o0, o1)]
            prior_w = 0.25 + 0.75 * (pf_w ** 1.4)

            # Classify opponent's current-street strength for action-conditioned equity
            or0, or1 = o0 >> 2, o1 >> 2
            opp_cat_now = opp1_now >> 20
            opp_hit_board = (or0 in b_set) or (or1 in b_set)
            opp_pocket = (or0 == or1)

            if opp_cat_now > board_cat_now + 1 or (opp_cat_now >= 2 and not self.board_paired):
                # Two pair, trips, straight, flush, full house+
                bw = 1.0 * prior_w
                cw = 0.22 * prior_w
            elif or0 == b_max_r or or1 == b_max_r or (opp_pocket and or0 > b_max_r):
                # Top pair or overpair
                bw = 0.82 * prior_w
                cw = 0.45 * prior_w
            elif opp_hit_board or opp_pocket:
                # Middle / bottom / underpair
                bw = weak_pair_bet_w * prior_w
                cw = 0.90 * prior_w
            else:
                # Air or draw
                bw = air_bet_w * prior_w
                cw = 1.0 * prior_w

            bet_w_total += bw
            check_w_total += cw

            if my_final > opp1_final:
                hu_wins += 1.0
                bet_w_wins += bw
                check_w_wins += cw
            elif my_final == opp1_final:
                hu_wins += 0.5
                bet_w_wins += 0.5 * bw
                check_w_wins += 0.5 * cw

            if to_deal > 0:
                if my_now_score > opp1_now:
                    ahead_now += 1
                    if my_final < opp1_final:
                        ahead_lost += 1.0
                    elif my_final == opp1_final:
                        ahead_lost += 0.5
                elif my_now_score < opp1_now:
                    behind_now += 1
                    if my_final > opp1_final:
                        behind_won += 1.0
                    elif my_final == opp1_final:
                        behind_won += 0.5

            best_opp = opp1_final
            for k in range(1, n_opp):
                ok0 = drawn[to_deal + 2 * k]
                ok1 = drawn[to_deal + 2 * k + 1]
                sc = _fast_eval((ok0, ok1, *full_board))
                if sc > best_opp:
                    best_opp = sc
            if my_final > best_opp:
                mw_wins += 1.0
            elif my_final == best_opp:
                mw_wins += 0.5

        self.raw_equity = hu_wins / num_samples
        self.multiway_equity = mw_wins / num_samples
        self.vs_bet_equity = (bet_w_wins / max(1e-9, bet_w_total)) ** n_opp
        self.vs_check_equity = (check_w_wins / max(1e-9, check_w_total)) ** n_opp
        self.pp = (behind_won / behind_now) if behind_now > 0 else 0.0
        self.np = (ahead_lost / ahead_now) if ahead_now > 0 else 0.0
        hs = self.vs_check_equity
        self.ehs = hs * (1.0 - self.np) + (1.0 - hs) * self.pp

    def conditioned_equity(self, facing_bet: bool, bet_to_pot: float,
                           opp_barrels: int, bluff_adj: float = 0.0) -> float:
        """
        Returns our effective equity conditioned on the opponent's action this street
        and how many postflop streets they have barreled.
        """
        if not facing_bet:
            return self.ehs
        base = self.vs_bet_equity
        # Discount further against large bets (>60% pot) or multi-street barrels (2nd/3rd barrel)
        size_disc = min(0.14, max(0.0, (bet_to_pot - 0.50) * 0.16))
        barrel_disc = 0.05 * max(0, opp_barrels - 1)
        # On flop/turn, draws retain their direct positive potential (PP)
        draw_eq = self.pp * (0.85 if self.street == "flop" else 0.75)
        # Structural caps on river/turn when holding no made pair or bottom pair
        if self.street == "river":
            if self.real_cat == 0:
                return max(0.02, min(0.12, base - size_disc - barrel_disc + bluff_adj))
            if self.pair_type in ("bottom_pair", "underpair") and (opp_barrels >= 2 or bet_to_pot >= 0.45):
                return max(0.05, min(0.20, base - size_disc - barrel_disc + bluff_adj))
            if self.pair_type in ("second_pair", "mid_pocket_pair") and opp_barrels >= 3 and bet_to_pot >= 0.55:
                return max(0.08, min(0.24, base - size_disc - barrel_disc + bluff_adj))
        elif self.street == "turn":
            if self.real_cat == 0 and not self.flush_draw and self.straight_outs < 8:
                return max(0.03, min(0.16, draw_eq + bluff_adj))
            if self.pair_type in ("bottom_pair", "underpair") and opp_barrels >= 2 and bet_to_pot >= 0.50:
                return max(draw_eq, min(0.22, base - size_disc - barrel_disc + bluff_adj))
        elif self.street == "flop":
            if self.real_cat == 0 and not self.flush_draw and self.straight_outs < 4 and self.overcards < 2:
                return max(0.04, min(0.18, draw_eq + bluff_adj))

        return max(draw_eq, min(0.98, base - size_disc - barrel_disc + bluff_adj))


# ---------------------------------------------------------------------------
# 4. HAND ACTION TRACKER & PERSISTENT OPPONENT HUD
# ---------------------------------------------------------------------------

class OpponentStats:
    """Persistent statistics for one opponent team across hands."""

    def __init__(self):
        self.hands_seen = 0
        self.vpip_hands = 0
        self.pfr_hands = 0
        self.three_bet_opps = 0
        self.three_bets = 0
        self.fold_to_steal_opps = 0
        self.folds_to_steal = 0
        self.pre_shoves = 0
        self.cbet_opps = 0
        self.cbets = 0
        self.fold_to_cbet_opps = 0
        self.folds_to_cbet = 0
        self.postflop_bets_raises = 0
        self.postflop_calls = 0
        self.postflop_folds = 0
        self.river_bet_opps = 0
        self.river_bets = 0
        self.fold_to_river_opps = 0
        self.folds_to_river = 0
        self.showdowns = 0
        self.showdown_wins = 0
        self.showdown_bluffs = 0

    @property
    def vpip(self) -> float:
        return (self.vpip_hands + 2.7) / (self.hands_seen + 6.0)

    @property
    def pfr(self) -> float:
        return (self.pfr_hands + 1.8) / (self.hands_seen + 6.0)

    @property
    def three_bet_freq(self) -> float:
        return (self.three_bets + 0.6) / (self.three_bet_opps + 6.0)

    @property
    def fold_to_steal(self) -> float:
        return (self.folds_to_steal + 2.5) / (self.fold_to_steal_opps + 5.0)

    @property
    def shove_freq(self) -> float:
        return self.pre_shoves / max(1, self.hands_seen)

    @property
    def cbet_freq(self) -> float:
        return (self.cbets + 2.4) / (self.cbet_opps + 4.0)

    @property
    def fold_to_cbet(self) -> float:
        return (self.folds_to_cbet + 2.0) / (self.fold_to_cbet_opps + 4.0)

    @property
    def aggression_factor(self) -> float:
        return (self.postflop_bets_raises + 2.0) / (self.postflop_calls + 2.0)

    @property
    def fold_to_postflop(self) -> float:
        tot = self.postflop_bets_raises + self.postflop_calls + self.postflop_folds
        return (self.postflop_folds + 2.0) / (tot + 5.0)

    @property
    def fold_to_river(self) -> float:
        return (self.folds_to_river + 1.8) / (self.fold_to_river_opps + 4.0)

    @property
    def bluff_tendency(self) -> float:
        return (self.showdown_bluffs + 0.4) / (self.showdowns + 4.0)

    def archetype(self) -> str:
        if self.hands_seen < 6:
            return "unknown"
        if self.shove_freq > 0.30:
            return "shover"
        if self.vpip < 0.28 and self.pfr < 0.20 and self.fold_to_postflop > 0.48:
            return "nit"
        if self.vpip > 0.58 and self.aggression_factor < 1.05 and self.fold_to_postflop < 0.35:
            return "station"
        if self.pfr > 0.45 or self.aggression_factor > 2.4:
            return "maniac"
        if self.vpip > 0.42 and self.pfr > 0.30 and self.aggression_factor >= 1.3:
            return "lag"
        return "tag"


class TableTracker:
    """Tracks full hand history and updates per-team OpponentStats across hands."""

    def __init__(self):
        self.stats = {}
        self.reset_hand()

    def reset_hand(self):
        self.initialized = False
        self.street = "preflop"
        self.board_cards = []
        self.pf_raises = 0
        self.pf_aggressor = None
        self.street_raises = 0
        self.street_aggressor = None
        self.prev_street_aggressor = None
        self.vpip_set = set()
        self.pfr_set = set()
        self.three_bet_set = set()
        self.cbet_fired = False
        self.cbet_opp_recorded = False
        self.postflop_aggressors = set()
        self.player_barrels = collections.Counter()  # pid -> number of postflop streets bet/raised
        self.streets_bet_by_player = collections.defaultdict(set)
        self.shown_cards = {}
        self.wins = {}
        self.teams = ()
        self.big_blind = 20
        self.num_players = 2
        self.hand_actions = []

    def get_stats(self, team_name: str) -> OpponentStats:
        if team_name not in self.stats:
            self.stats[team_name] = OpponentStats()
        return self.stats[team_name]

    def max_opp_barrels(self, my_id: int) -> int:
        return max((cnt for pid, cnt in self.player_barrels.items() if pid != my_id), default=0)

    def sync(self, actions: list[PlayerAction], game_state, my_id: int, is_end: bool = False):
        if not self.initialized:
            self.initialized = True
            self.teams = tuple(game_state.teams)
            self.big_blind = game_state.big_blind
            self.num_players = game_state.num_players
            for t in self.teams:
                if t not in self.stats:
                    self.stats[t] = OpponentStats()

        for act in actions:
            self.hand_actions.append(act)
            pid = act.player_id
            kind = act.kind

            if pid == DEALER:
                if kind == "reveal_card" and act.card is not None:
                    self.board_cards.append(act.card)
                    nb = len(self.board_cards)
                    if nb in (3, 4, 5):
                        self.prev_street_aggressor = (
                            self.pf_aggressor if nb == 3 else self.street_aggressor
                        )
                        self.street = "flop" if nb == 3 else ("turn" if nb == 4 else "river")
                        self.street_raises = 0
                        self.street_aggressor = None
                continue

            if kind == "reveal_card":
                if act.card is not None:
                    self.shown_cards.setdefault(pid, []).append(act.card)
                continue

            if kind == "win":
                self.wins[pid] = self.wins.get(pid, 0) + act.amount
                continue

            if pid >= len(self.teams):
                continue
            tname = self.teams[pid]
            st = self.get_stats(tname)

            if self.street == "preflop":
                if kind == "call":
                    self.vpip_set.add(pid)
                elif kind == "raise":
                    self.vpip_set.add(pid)
                    self.pfr_set.add(pid)
                    if self.pf_raises == 1:
                        self.three_bet_set.add(pid)
                        st.three_bet_opps += 1
                        st.three_bets += 1
                    self.pf_raises += 1
                    self.pf_aggressor = pid
                    if act.amount >= 12 * self.big_blind:
                        st.pre_shoves += 1
                elif kind == "fold":
                    if self.pf_raises == 1:
                        st.three_bet_opps += 1
                        st.fold_to_steal_opps += 1
                        st.folds_to_steal += 1
            else:
                if self.street == "flop" and not self.cbet_opp_recorded:
                    if pid == self.pf_aggressor and self.street_raises == 0:
                        self.cbet_opp_recorded = True
                        st.cbet_opps += 1
                        if kind == "raise":
                            st.cbets += 1
                            self.cbet_fired = True

                if kind == "raise":
                    st.postflop_bets_raises += 1
                    self.street_raises += 1
                    self.street_aggressor = pid
                    self.postflop_aggressors.add(pid)
                    if self.street not in self.streets_bet_by_player[pid]:
                        self.streets_bet_by_player[pid].add(self.street)
                        self.player_barrels[pid] += 1
                    if self.street == "river":
                        st.river_bet_opps += 1
                        st.river_bets += 1
                elif kind == "call":
                    if self.street_raises > 0:
                        st.postflop_calls += 1
                        if self.street == "flop" and self.cbet_fired:
                            st.fold_to_cbet_opps += 1
                        if self.street == "river":
                            st.fold_to_river_opps += 1
                elif kind == "fold":
                    if self.street_raises > 0:
                        st.postflop_folds += 1
                        if self.street == "flop" and self.cbet_fired:
                            st.fold_to_cbet_opps += 1
                            st.folds_to_cbet += 1
                        if self.street == "river":
                            st.fold_to_river_opps += 1
                            st.folds_to_river += 1

        if is_end:
            for pid, tname in enumerate(self.teams):
                if pid == my_id:
                    continue
                st = self.get_stats(tname)
                st.hands_seen += 1
                if pid in self.vpip_set:
                    st.vpip_hands += 1
                if pid in self.pfr_set:
                    st.pfr_hands += 1
                if pid in self.shown_cards and len(self.shown_cards[pid]) == 2:
                    st.showdowns += 1
                    if self.wins.get(pid, 0) > 0:
                        st.showdown_wins += 1
                    if len(self.board_cards) == 5:
                        cards_int = [_card_to_int(c) for c in self.shown_cards[pid] + self.board_cards]
                        cat = _fast_eval(cards_int) >> 20
                        if cat <= 0 and pid in self.postflop_aggressors:
                            st.showdown_bluffs += 1


# ---------------------------------------------------------------------------
# 5. SHARED SAFE ACTION BUILDER & POSITION / SIZING HELPERS
# ---------------------------------------------------------------------------

def _safe_action(game_state, me: int, kind: str, amount: int = 0) -> PlayerAction:
    me = int(me)
    if kind == "raise" and game_state.can_raise and game_state.raise_range is not None:
        low, high = game_state.raise_range
        target = int(round(amount))
        target = max(int(low), min(int(high), target))
        act = PlayerAction.raise_to(me, target)
    elif kind == "fold":
        if game_state.call_amount == 0:
            act = PlayerAction.call(me)
        else:
            act = PlayerAction.fold(me)
    else:
        act = PlayerAction.call(me)

    try:
        game_state.validate_action(act)
        return act
    except Exception:
        return PlayerAction.call(me)


def _raise_by_pot_fraction(game_state, fraction: float) -> int:
    me = game_state.to_act
    my_bet = game_state.bets[me]
    call_amt = game_state.call_amount
    pot_after_call = game_state.pot + call_amt
    target_total = my_bet + call_amt + int(round(pot_after_call * fraction))
    if game_state.raise_range is not None:
        low, high = game_state.raise_range
        return max(int(low), min(int(high), target_total))
    return target_total


def _geometric_fraction(spr: float, streets_left: int) -> float:
    if spr <= 0.05 or streets_left <= 0:
        return 1.0
    s = max(1, streets_left)
    p = 0.5 * ((1.0 + 2.0 * spr) ** (1.0 / s) - 1.0)
    return max(0.25, min(1.5, p))


def _in_position(game_state, me: int) -> bool:
    btn = game_state.button
    n = game_state.num_players
    live = set(game_state.players_in_hand)
    if len(live) <= 1:
        return True
    for step in range(n, 0, -1):
        seat = (btn + step) % n
        if seat in live:
            return seat == me
    return False


def _deterministic_mix(hand: list[Card], board: tuple[Card, ...], pot: int, salt: int = 0) -> float:
    h = (
        _card_to_int(hand[0]) * 131
        + _card_to_int(hand[1]) * 17
        + len(board) * 1009
        + pot * 37
        + salt * 7919
    )
    for i, c in enumerate(board):
        h ^= (_card_to_int(c) + 19) * (257 ** (i + 1))
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = ((h >> 16) ^ h) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


# ---------------------------------------------------------------------------
# BOT 1: GTOBot ("gto_apex")
# Game-Theoretic Optimal: Action-Conditioned MDF, Geometric Sizing, Balanced Bluffs
# ---------------------------------------------------------------------------

class GTOBot:
    def __init__(self, seed: int = 101):
        self.hand = []
        self.player_id = 0
        self.rng = random.Random(seed)
        self.tracker = TableTracker()

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self.tracker.reset_hand()

    def end_game(self, actions: list[PlayerAction], game_state):
        self.tracker.sync(actions, game_state, self.player_id, is_end=True)

    def run(self, actions: list[PlayerAction], game_state) -> PlayerAction:
        self.tracker.sync(actions, game_state, self.player_id, is_end=False)
        n_opp = max(1, len(game_state.players_in_hand) - 1)
        analysis = HandAnalysis(self.hand, game_state.board, n_opp, self.rng, num_samples=150)
        return self.decide(game_state, analysis)

    def decide(self, game_state, a: HandAnalysis) -> PlayerAction:
        me = self.player_id
        bb = max(1, game_state.big_blind)
        eff_bb = game_state.effective_stack(me) / bb
        my_total_bb = (game_state.stacks[me] + game_state.bets[me]) / bb
        call_amt = game_state.call_amount
        pot = game_state.pot
        pot_odds = game_state.pot_odds
        ip = _in_position(game_state, me)
        mix = _deterministic_mix(self.hand, game_state.board, pot, salt=1)

        if game_state.street == "preflop":
            return self._play_preflop(game_state, a, eff_bb, my_total_bb, call_amt, bb, pot_odds, ip, mix)
        return self._play_postflop(game_state, a, call_amt, pot, pot_odds, ip, mix)

    def _play_preflop(self, game_state, a: HandAnalysis, eff_bb: float, my_total_bb: float,
                      call_amt: int, bb: int, pot_odds: float, ip: bool, mix: float) -> PlayerAction:
        me = self.player_id
        n_in = len(game_state.players_in_hand)
        hu = (game_state.num_players == 2 or n_in == 2)
        pct = a.hu_percentile if hu else a.mw_percentile
        eq = a.multiway_equity

        if eff_bb <= 13.0 or my_total_bb <= 13.0:
            stack_bb = min(eff_bb, my_total_bb)
            if self.tracker.pf_raises == 0:
                if a.nash_push_bb >= stack_bb * (0.95 if hu else 1.25):
                    return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                if call_amt == 0 or (call_amt <= bb // 2 and pct >= 0.35):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")
            else:
                if a.nash_call_bb >= (call_amt / bb) * (0.95 if hu else 1.2) or (eq >= pot_odds + 0.05 and pct >= 0.60):
                    if pct >= 0.80 and game_state.can_raise:
                        return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

        if self.tracker.pf_raises == 0:
            open_thresh = 0.22 if hu else max(0.45, 0.78 - 0.10 * (n_in - 2))
            if call_amt == 0:
                if pct >= (0.60 if hu else 0.72):
                    size = int(round(bb * (3.0 if hu else 3.5)))
                    return _safe_action(game_state, me, "raise", size)
                return _safe_action(game_state, me, "call")

            if pct >= open_thresh:
                open_size = int(round(bb * (2.2 if hu else 2.5)))
                return _safe_action(game_state, me, "raise", open_size)
            if hu and call_amt <= bb // 2 and (pct >= 0.15 or a.mw_preflop_score >= 0.45):
                return _safe_action(game_state, me, "call")
            return _safe_action(game_state, me, "fold")

        if self.tracker.pf_raises == 1:
            raise_bb = game_state.current_bet / bb
            if raise_bb >= 10.0:
                req_eq = pot_odds + (0.04 if hu else 0.08)
                if (eq >= req_eq and pct >= (0.70 if hu else 0.82)) or pct >= 0.90:
                    if pct >= 0.93 and game_state.can_raise:
                        return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

            val_3bet_thresh = 0.84 if hu else 0.90
            bluff_3bet = (
                0.60 <= pct < val_3bet_thresh
                and ((self.hand[0].suit == self.hand[1].suit and max(c.value for c in self.hand) >= 13)
                     or a.cat == 1)
                and mix < 0.28
            )
            if (pct >= val_3bet_thresh or bluff_3bet) and game_state.can_raise:
                mult = 3.0 if ip else 3.4
                target = int(round(game_state.current_bet * mult))
                if target >= 0.42 * (game_state.stacks[me] + game_state.bets[me]):
                    target = game_state.max_raise_to
                return _safe_action(game_state, me, "raise", target)

            pot = game_state.pot
            mdf = pot / (pot + call_amt) if call_amt > 0 else 1.0
            realization = 1.05 if ip else 0.94
            if eq * realization >= pot_odds - 0.01 and pct >= max(0.32 if hu else 0.50, 1.0 - mdf * 0.82):
                return _safe_action(game_state, me, "call")
            return _safe_action(game_state, me, "fold")

        if pct >= (0.91 if hu else 0.95):
            if game_state.can_raise and (pct >= 0.95 or eff_bb <= 28):
                return _safe_action(game_state, me, "raise", game_state.max_raise_to)
            return _safe_action(game_state, me, "call")
        if eq >= pot_odds + 0.04 and pct >= (0.78 if hu else 0.86) and call_amt <= 0.35 * (game_state.stacks[me] + game_state.bets[me]):
            return _safe_action(game_state, me, "call")
        return _safe_action(game_state, me, "fold")

    def _play_postflop(self, game_state, a: HandAnalysis, call_amt: int, pot: int,
                       pot_odds: float, ip: bool, mix: float) -> PlayerAction:
        me = self.player_id
        streets_left = {"flop": 3, "turn": 2, "river": 1}[game_state.street]
        spr = game_state.spr(me)
        geo_frac = _geometric_fraction(spr, streets_left)
        ehs = a.ehs

        if a.is_playing_board and call_amt > 0:
            return _safe_action(game_state, me, "fold")

        # 1. Checked to us (call_amt == 0)
        if call_amt == 0:
            if not game_state.can_raise:
                return _safe_action(game_state, me, "call")

            # Strong value hands (True two pair+, or very strong top pair / overpair)
            if a.real_cat >= 2 or (a.real_cat == 1 and (a.is_overpair or (a.is_top_pair and a.top_pair_kicker >= 9)) and ehs >= 0.72):
                if a.real_cat >= 3 and a.board_wetness < 0.25 and ip and game_state.street == "flop" and mix < 0.22:
                    return _safe_action(game_state, me, "call")
                frac = max(0.50, min(0.95, geo_frac))
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))

            # Medium value / protection (top pair or good second pair)
            if a.real_cat >= 1 and ehs >= 0.58:
                if game_state.street == "flop" or (game_state.street == "turn" and a.is_top_pair and mix < 0.65) or (game_state.street == "river" and a.is_top_pair and ehs >= 0.68):
                    frac = 0.35 if a.board_wetness < 0.35 else 0.55
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))
                return _safe_action(game_state, me, "call")

            # Semi-bluffs on flop/turn
            is_semibluff = (
                game_state.street in ("flop", "turn")
                and (a.flush_draw or a.straight_outs >= 8 or (a.straight_outs >= 4 and a.overcards >= 1))
            )
            if is_semibluff and mix < (0.68 if ip else 0.50):
                frac = 0.48 if a.nut_flush_draw or a.straight_outs >= 8 else 0.35
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))

            # Flop range c-bet on dry high boards
            if game_state.street == "flop" and self.tracker.pf_aggressor == me and len(game_state.players_in_hand) == 2:
                if a.board_wetness < 0.38 and a.board_high_rank >= 10 and mix < 0.60:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.33))

            # Balanced river bluff ONLY with nut blocker and missed hand
            if game_state.street == "river" and a.real_cat == 0 and a.has_nut_flush_blocker and len(game_state.players_in_hand) == 2:
                if mix < 0.24:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.62))

            return _safe_action(game_state, me, "call")

        # 2. Facing a bet/raise (call_amt > 0)
        bet_to_pot = call_amt / max(1, pot - call_amt)
        opp_barrels = self.tracker.max_opp_barrels(me)
        cond_eq = a.conditioned_equity(True, bet_to_pot, opp_barrels)

        # Value raise ONLY with genuine Two Pair+ (`real_cat >= 2`) or monster combo draw
        if a.real_cat >= 2 and cond_eq >= 0.72 and game_state.can_raise:
            if a.real_cat >= 3 and (spr <= 1.8 or cond_eq >= 0.85):
                return _safe_action(game_state, me, "raise", game_state.max_raise_to)
            frac = max(0.60, min(1.0, geo_frac))
            return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))

        if game_state.street in ("flop", "turn") and game_state.can_raise:
            if (a.nut_flush_draw and a.straight_outs >= 4) or (a.flush_draw and a.straight_outs >= 8):
                if mix < 0.55:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.65))

        implied_bonus = 0.0
        if game_state.street in ("flop", "turn") and spr >= 1.2:
            if a.nut_flush_draw or a.straight_outs >= 8:
                implied_bonus = 0.06
            elif a.flush_draw or a.straight_outs >= 4:
                implied_bonus = 0.03

        if cond_eq + implied_bonus >= pot_odds + 0.01:
            return _safe_action(game_state, me, "call")

        return _safe_action(game_state, me, "fold")


# ---------------------------------------------------------------------------
# BOT 2: BayesianBot ("bayesian_ranger")
# Combinatorial 169-Hand Bayesian Range Filtering & Explicit EV Maximization
# ---------------------------------------------------------------------------

class BayesianBot:
    def __init__(self, seed: int = 202):
        self.hand = []
        self.player_id = 0
        self.rng = random.Random(seed)
        self.tracker = TableTracker()
        self.opp_looseness = {}
        self.range_weights = []

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self.tracker.reset_hand()
        self.range_weights = list(_COMBO_COUNTS)

    def end_game(self, actions: list[PlayerAction], game_state):
        self.tracker.sync(actions, game_state, self.player_id, is_end=True)
        for pid, cards in self.tracker.shown_cards.items():
            if pid == self.player_id or len(cards) != 2 or pid >= len(self.tracker.teams):
                continue
            tname = self.tracker.teams[pid]
            c_idx = _canonical_idx(_card_to_int(cards[0]), _card_to_int(cards[1]))
            pct = _HU_PERCENTILE[c_idx]
            old_l = self.opp_looseness.get(tname, 1.0)
            observed_l = 1.45 - 0.80 * pct
            self.opp_looseness[tname] = 0.85 * old_l + 0.15 * observed_l

    def _update_bayesian_range(self, game_state):
        weights = [float(c) for c in _COMBO_COUNTS]
        live_opps = [p for p in game_state.players_in_hand if p != self.player_id]
        looseness = 1.0
        fold_tendency = 0.40
        if live_opps:
            tnames = [game_state.teams[p] for p in live_opps]
            looseness = sum(self.opp_looseness.get(t, 1.0) for t in tnames) / len(tnames)
            stats_list = [self.tracker.get_stats(t) for t in tnames]
            vpip_avg = sum(s.vpip for s in stats_list) / len(stats_list)
            looseness = 0.5 * looseness + 0.5 * (vpip_avg / 0.42)
            fold_tendency = sum(s.fold_to_postflop for s in stats_list) / len(stats_list)

        pf_raises = self.tracker.pf_raises
        for idx in range(169):
            pct = _HU_PERCENTILE[idx]
            if pf_raises == 0:
                likelihood = 0.35 if pct > 0.90 else (0.85 if pct >= 0.25 / max(0.5, looseness) else 0.30)
            elif pf_raises == 1:
                cutoff = max(0.15, 0.50 - 0.22 * looseness)
                likelihood = 1.0 if pct >= cutoff else 0.12
            else:
                cutoff = max(0.45, 0.82 - 0.18 * looseness)
                likelihood = 1.0 if pct >= cutoff else 0.05
            weights[idx] *= likelihood

        self.range_weights = weights
        return looseness, fold_tendency

    def run(self, actions: list[PlayerAction], game_state) -> PlayerAction:
        self.tracker.sync(actions, game_state, self.player_id, is_end=False)
        n_opp = max(1, len(game_state.players_in_hand) - 1)
        a = HandAnalysis(self.hand, game_state.board, n_opp, self.rng, num_samples=150)
        return self.decide(game_state, a)

    def decide(self, game_state, a: HandAnalysis) -> PlayerAction:
        me = self.player_id
        looseness, fold_tendency = self._update_bayesian_range(game_state)
        pot = game_state.pot
        call_amt = game_state.call_amount
        pot_odds = game_state.pot_odds
        bb = max(1, game_state.big_blind)

        if a.is_playing_board and call_amt > 0:
            return _safe_action(game_state, me, "fold")

        if game_state.street == "preflop":
            hu = len(game_state.players_in_hand) == 2
            pct = a.hu_percentile if hu else a.mw_percentile
            if self.tracker.pf_raises == 0:
                if pct >= (0.24 if hu else 0.55):
                    return _safe_action(game_state, me, "raise", int(round(bb * (2.25 if hu else 2.5))))
                if call_amt == 0 or (hu and call_amt <= bb // 2 and pct >= 0.16):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")
            if self.tracker.pf_raises == 1:
                if pct >= (0.83 if hu else 0.89) and game_state.can_raise and game_state.current_bet <= 5 * bb:
                    return _safe_action(game_state, me, "raise", int(round(game_state.current_bet * 3.0)))
                if a.multiway_equity >= pot_odds and pct >= (0.35 if hu else 0.52):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")
            if pct >= 0.91:
                return _safe_action(game_state, me, "raise" if game_state.can_raise else "call", game_state.max_raise_to)
            if a.multiway_equity >= pot_odds + 0.04 and pct >= 0.78 and call_amt <= 0.35 * (game_state.stacks[me] + game_state.bets[me]):
                return _safe_action(game_state, me, "call")
            return _safe_action(game_state, me, "fold")

        # Postflop Bayesian Action-Conditioned EV
        bet_to_pot = call_amt / max(1, pot - call_amt) if call_amt > 0 else 0.0
        opp_barrels = self.tracker.max_opp_barrels(me)
        eq = a.conditioned_equity(call_amt > 0, bet_to_pot, opp_barrels)

        ev_call = eq * (pot + call_amt) - call_amt
        best_kind = "call" if (call_amt == 0 or ev_call >= 0.05 * bb) else "fold"
        best_ev = 0.0 if best_kind == "fold" else ev_call
        best_amount = 0

        # Only consider raising postflop if we have a real value hand (real_cat >= 2, or strong top pair when checked to) or strong draw
        can_consider_raise = (
            a.real_cat >= 2
            or (call_amt == 0 and a.real_cat == 1 and (a.is_top_pair or a.is_overpair))
            or (game_state.street != "river" and (a.nut_flush_draw or a.straight_outs >= 8))
        )

        if game_state.can_raise and game_state.raise_range is not None and can_consider_raise:
            low_r, high_r = game_state.raise_range
            my_bet = game_state.bets[me]
            candidates = [
                _raise_by_pot_fraction(game_state, 0.45),
                _raise_by_pot_fraction(game_state, 0.72),
            ]
            if a.real_cat >= 3:
                candidates.append(_raise_by_pot_fraction(game_state, 1.05))
                if game_state.spr(me) <= 2.0:
                    candidates.append(high_r)

            seen_totals = set()
            n_opp = max(1, len(game_state.players_in_hand) - 1)
            for total in candidates:
                total = max(int(low_r), min(int(high_r), int(total)))
                if total in seen_totals:
                    continue
                seen_totals.add(total)
                cost = total - my_bet
                raise_inc = total - game_state.current_bet
                pot_after_call = pot + call_amt
                size_ratio = raise_inc / max(1, pot_after_call)

                # Cap overbets unless holding trips/nutted hand
                if size_ratio > 1.15 and a.real_cat < 3:
                    continue

                base_fe = fold_tendency * (size_ratio / (size_ratio + 0.95))
                fe = max(0.05, min(0.70, base_fe)) ** n_opp

                selection_discount = 0.78 + 0.18 * (eq ** 2) - 0.08 * min(1.5, size_ratio)
                eq_when_called = max(0.02, min(0.98, eq * selection_discount))
                if game_state.street in ("flop", "turn") and (a.nut_flush_draw or a.straight_outs >= 8):
                    eq_when_called = min(0.92, eq_when_called + 0.05)

                final_pot_if_called = pot + cost + raise_inc
                ev_called = eq_when_called * final_pot_if_called - cost
                ev_raise = fe * pot + (1.0 - fe) * ev_called

                if ev_raise > best_ev + 0.25 * bb:
                    best_ev = ev_raise
                    best_kind = "raise"
                    best_amount = total

        if best_kind == "raise":
            return _safe_action(game_state, me, "raise", best_amount)
        if best_kind == "call" or call_amt == 0:
            return _safe_action(game_state, me, "call")
        return _safe_action(game_state, me, "fold")


# ---------------------------------------------------------------------------
# BOT 3: SharkBot ("hud_shark")
# Exploitative HUD Profiler & Targeted Counter-Strategy Engine
# ---------------------------------------------------------------------------

class SharkBot:
    def __init__(self, seed: int = 303):
        self.hand = []
        self.player_id = 0
        self.rng = random.Random(seed)
        self.tracker = TableTracker()
        self.gto_fallback = GTOBot(seed=seed + 1)

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self.tracker.reset_hand()
        self.gto_fallback.seed(hand, player_id)

    def end_game(self, actions: list[PlayerAction], game_state):
        self.tracker.sync(actions, game_state, self.player_id, is_end=True)

    def _primary_opponent_stats(self, game_state) -> OpponentStats:
        opps = [p for p in game_state.players_in_hand if p != self.player_id]
        if not opps:
            return OpponentStats()
        agg = self.tracker.street_aggressor or self.tracker.pf_aggressor
        target_p = agg if (agg in opps) else opps[0]
        return self.tracker.get_stats(game_state.teams[target_p])

    def run(self, actions: list[PlayerAction], game_state) -> PlayerAction:
        self.tracker.sync(actions, game_state, self.player_id, is_end=False)
        self.gto_fallback.tracker = self.tracker
        n_opp = max(1, len(game_state.players_in_hand) - 1)
        a = HandAnalysis(self.hand, game_state.board, n_opp, self.rng, num_samples=150)
        return self.decide(game_state, a)

    def decide(self, game_state, a: HandAnalysis) -> PlayerAction:
        me = self.player_id
        st = self._primary_opponent_stats(game_state)
        arch = st.archetype()
        call_amt = game_state.call_amount
        pot = game_state.pot
        pot_odds = game_state.pot_odds
        bb = max(1, game_state.big_blind)
        hu = len(game_state.players_in_hand) == 2
        pct = a.hu_percentile if hu else a.mw_percentile
        ehs = a.ehs
        eq = a.multiway_equity

        if arch in ("unknown", "tag"):
            return self.gto_fallback.decide(game_state, a)

        if arch == "shover":
            if game_state.street == "preflop":
                req_pct = 0.48 if hu else 0.65
                if pct >= req_pct or a.cat == 1 or eq >= pot_odds:
                    if pct >= 0.72 and game_state.can_raise:
                        return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")
            else:
                if a.real_cat >= 1 or eq >= pot_odds:
                    if a.real_cat >= 2 and game_state.can_raise:
                        return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

        if arch == "nit":
            if game_state.street == "preflop":
                if self.tracker.pf_raises == 0:
                    if pct >= (0.12 if hu else 0.35) or st.fold_to_steal > 0.62:
                        return _safe_action(game_state, me, "raise", int(round(bb * 2.1)))
                    return _safe_action(game_state, me, "call" if call_amt == 0 else "fold")
                if pct >= 0.90 and game_state.can_raise:
                    return _safe_action(game_state, me, "raise", int(round(game_state.current_bet * 3.0)))
                if pct >= 0.70 and eq >= pot_odds:
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

            if call_amt == 0:
                if game_state.can_raise and (a.real_cat >= 2 or st.fold_to_cbet > 0.52 or st.fold_to_postflop > 0.50):
                    frac = 0.65 if a.real_cat >= 2 else 0.33
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))
                return _safe_action(game_state, me, "call")
            else:
                bet_to_pot = call_amt / max(1, pot - call_amt)
                cond_eq = a.conditioned_equity(True, bet_to_pot, self.tracker.max_opp_barrels(me), bluff_adj=-0.04)
                if a.real_cat >= 3 and game_state.can_raise:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.75))
                if cond_eq >= pot_odds + 0.04:
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

        if arch == "station":
            if game_state.street == "preflop":
                if self.tracker.pf_raises == 0:
                    if pct >= (0.40 if hu else 0.58):
                        return _safe_action(game_state, me, "raise", int(round(bb * 3.0)))
                    if call_amt <= bb // 2 and pct >= 0.25:
                        return _safe_action(game_state, me, "call")
                    return _safe_action(game_state, me, "fold")
                if pct >= 0.84 and game_state.can_raise:
                    return _safe_action(game_state, me, "raise", int(round(game_state.current_bet * 3.2)))
                if eq >= pot_odds and pct >= 0.42:
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

            if call_amt == 0:
                if not game_state.can_raise:
                    return _safe_action(game_state, me, "call")
                if a.real_cat >= 2:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.90))
                if a.real_cat == 1 and (a.is_top_pair or a.is_overpair) and ehs >= 0.62:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.65))
                return _safe_action(game_state, me, "call")
            else:
                bet_to_pot = call_amt / max(1, pot - call_amt)
                cond_eq = a.conditioned_equity(True, bet_to_pot, self.tracker.max_opp_barrels(me))
                if a.real_cat >= 2 and cond_eq >= 0.75 and game_state.can_raise:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.85))
                if cond_eq >= pot_odds + 0.02:
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

        # vs Maniac or LAG: only give bluff bonus if opponent actually showed bluffs or extreme AF,
        # and NEVER call down with high-card (real_cat == 0) on turn/river!
        bluff_bonus = min(0.06, st.bluff_tendency * 0.15) if a.real_cat >= 1 else 0.0
        if game_state.street == "preflop":
            return self.gto_fallback.decide(game_state, a)

        if call_amt == 0:
            if a.real_cat >= 3 and game_state.street in ("flop", "turn"):
                if _deterministic_mix(self.hand, game_state.board, pot, salt=3) < 0.45:
                    return _safe_action(game_state, me, "call")
            return self.gto_fallback.decide(game_state, a)
        else:
            bet_to_pot = call_amt / max(1, pot - call_amt)
            cond_eq = a.conditioned_equity(True, bet_to_pot, self.tracker.max_opp_barrels(me), bluff_adj=bluff_bonus)
            if a.real_cat >= 2 and cond_eq >= 0.74 and game_state.can_raise:
                if game_state.street == "river" or game_state.spr(me) <= 2.0:
                    return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.75))
            if cond_eq >= pot_odds + 0.01:
                return _safe_action(game_state, me, "call")
            return _safe_action(game_state, me, "fold")


# ---------------------------------------------------------------------------
# BOT 4: PressureBot ("lag_pressure")
# Board-Texture, Range-Advantage vs Nut-Advantage, Positional Float & Barrel Bot
# ---------------------------------------------------------------------------

class PressureBot:
    def __init__(self, seed: int = 404):
        self.hand = []
        self.player_id = 0
        self.rng = random.Random(seed)
        self.tracker = TableTracker()

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self.tracker.reset_hand()

    def end_game(self, actions: list[PlayerAction], game_state):
        self.tracker.sync(actions, game_state, self.player_id, is_end=True)

    def run(self, actions: list[PlayerAction], game_state) -> PlayerAction:
        self.tracker.sync(actions, game_state, self.player_id, is_end=False)
        n_opp = max(1, len(game_state.players_in_hand) - 1)
        a = HandAnalysis(self.hand, game_state.board, n_opp, self.rng, num_samples=150)
        return self.decide(game_state, a)

    def decide(self, game_state, a: HandAnalysis) -> PlayerAction:
        me = self.player_id
        bb = max(1, game_state.big_blind)
        call_amt = game_state.call_amount
        pot = game_state.pot
        pot_odds = game_state.pot_odds
        ip = _in_position(game_state, me)
        hu = len(game_state.players_in_hand) == 2
        pct = a.hu_percentile if hu else a.mw_percentile
        ehs = a.ehs
        mix = _deterministic_mix(self.hand, game_state.board, pot, salt=4)

        if game_state.street == "preflop":
            eff_bb = game_state.effective_stack(me) / bb
            if eff_bb <= 12.0:
                if a.nash_push_bb >= eff_bb and self.tracker.pf_raises == 0:
                    return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                if a.nash_call_bb >= (call_amt / bb) or (a.multiway_equity >= pot_odds + 0.04 and pct >= 0.58):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

            if self.tracker.pf_raises == 0:
                open_cut = 0.18 if hu else (0.40 if ip else 0.55)
                if pct >= open_cut:
                    return _safe_action(game_state, me, "raise", int(round(bb * (2.25 if hu else 2.5))))
                if call_amt == 0 or (hu and call_amt <= bb // 2 and pct >= 0.15):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

            if self.tracker.pf_raises == 1:
                is_play_3bet = (
                    (self.hand[0].suit == self.hand[1].suit and pct >= 0.56 and mix < 0.35)
                    or pct >= (0.82 if hu else 0.88)
                )
                if is_play_3bet and game_state.can_raise and game_state.current_bet <= 4 * bb:
                    mult = 2.8 if ip else 3.3
                    return _safe_action(game_state, me, "raise", int(round(game_state.current_bet * mult)))
                if a.multiway_equity * (1.06 if ip else 0.95) >= pot_odds and pct >= (0.34 if hu else 0.50):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

            if pct >= 0.91:
                if game_state.can_raise and (pct >= 0.95 or eff_bb <= 26):
                    return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                return _safe_action(game_state, me, "call")
            if a.multiway_equity >= pot_odds + 0.04 and pct >= 0.76 and call_amt <= 0.35 * (game_state.stacks[me] + game_state.bets[me]):
                return _safe_action(game_state, me, "call")
            return _safe_action(game_state, me, "fold")

        if a.is_playing_board and call_amt > 0:
            return _safe_action(game_state, me, "fold")

        if call_amt == 0:
            if not game_state.can_raise:
                return _safe_action(game_state, me, "call")

            # Strong value (Two pair+ or strong top pair / overpair) or nut draws
            if a.real_cat >= 2 or (a.real_cat == 1 and (a.is_overpair or (a.is_top_pair and a.top_pair_kicker >= 9))) or a.nut_flush_draw or a.straight_outs >= 8:
                frac = 0.70 if a.board_wetness >= 0.40 else 0.50
                if game_state.street == "river" and a.real_cat >= 3:
                    frac = 0.90
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))

            # Medium value on flop/turn
            if a.real_cat == 1 and ehs >= 0.58 and game_state.street in ("flop", "turn"):
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.40))

            # Flop range c-bet
            if game_state.street == "flop" and hu:
                if (self.tracker.pf_aggressor == me or ip) and (a.board_high_rank >= 10 or a.board_paired or a.backdoor_flush_draw or a.straight_outs >= 4):
                    if mix < 0.72:
                        return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.33))

            # Turn delayed probe when opponent has NOT bet this hand (opp_barrels == 0)
            if game_state.street == "turn" and hu and self.tracker.max_opp_barrels(me) == 0:
                if (ip or a.flush_draw or a.straight_outs >= 4) and mix < 0.55:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.50))

            # River blocker pressure ONLY when opponent checked twice or we hold nut blocker
            if game_state.street == "river" and hu and a.real_cat == 0 and self.tracker.max_opp_barrels(me) <= 1:
                if a.has_nut_flush_blocker and mix < 0.28:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.65))

            return _safe_action(game_state, me, "call")

        # Facing a postflop bet: use disciplined action-conditioned equity!
        bet_to_pot = call_amt / max(1, pot - call_amt)
        opp_barrels = self.tracker.max_opp_barrels(me)
        cond_eq = a.conditioned_equity(True, bet_to_pot, opp_barrels)

        if game_state.can_raise:
            if (a.real_cat >= 2 and cond_eq >= 0.74) or (game_state.street == "flop" and (a.nut_flush_draw and a.straight_outs >= 4) and mix < 0.50):
                if a.real_cat >= 3 and (game_state.spr(me) <= 1.8 or cond_eq >= 0.85):
                    return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.70))

        draw_bonus = 0.05 if (game_state.street != "river" and (a.nut_flush_draw or a.straight_outs >= 8)) else 0.0
        if cond_eq + draw_bonus >= pot_odds + 0.01:
            return _safe_action(game_state, me, "call")

        return _safe_action(game_state, me, "fold")


# ---------------------------------------------------------------------------
# BOT 5: FortressBot ("nash_fortress")
# Short-Stack Nash Push/Fold + Multi-Way Pot Control + Monster Trapper
# ---------------------------------------------------------------------------

class FortressBot:
    def __init__(self, seed: int = 505):
        self.hand = []
        self.player_id = 0
        self.rng = random.Random(seed)
        self.tracker = TableTracker()

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self.tracker.reset_hand()

    def end_game(self, actions: list[PlayerAction], game_state):
        self.tracker.sync(actions, game_state, self.player_id, is_end=True)

    def run(self, actions: list[PlayerAction], game_state) -> PlayerAction:
        self.tracker.sync(actions, game_state, self.player_id, is_end=False)
        n_opp = max(1, len(game_state.players_in_hand) - 1)
        a = HandAnalysis(self.hand, game_state.board, n_opp, self.rng, num_samples=150)
        return self.decide(game_state, a)

    def decide(self, game_state, a: HandAnalysis) -> PlayerAction:
        me = self.player_id
        bb = max(1, game_state.big_blind)
        eff_bb = game_state.effective_stack(me) / bb
        my_total_bb = (game_state.stacks[me] + game_state.bets[me]) / bb
        stack_bb = min(eff_bb, my_total_bb)
        call_amt = game_state.call_amount
        pot = game_state.pot
        pot_odds = game_state.pot_odds
        hu = len(game_state.players_in_hand) == 2
        pct = a.hu_percentile if hu else a.mw_percentile
        ehs = a.ehs
        mix = _deterministic_mix(self.hand, game_state.board, pot, salt=5)

        if game_state.street == "preflop":
            if stack_bb <= 15.0:
                if self.tracker.pf_raises == 0:
                    if a.nash_push_bb >= stack_bb * (0.98 if hu else 1.30):
                        return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                    if call_amt == 0 or (hu and call_amt <= bb // 2 and pct >= 0.25):
                        return _safe_action(game_state, me, "call")
                    return _safe_action(game_state, me, "fold")
                else:
                    if a.nash_call_bb >= (call_amt / bb) * (0.95 if hu else 1.25) or (a.multiway_equity >= pot_odds + 0.05 and pct >= 0.62):
                        if pct >= 0.82 and game_state.can_raise:
                            return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                        return _safe_action(game_state, me, "call")
                    return _safe_action(game_state, me, "fold")

            if self.tracker.pf_raises == 0:
                if pct >= (0.26 if hu else 0.56):
                    return _safe_action(game_state, me, "raise", int(round(bb * (2.2 if hu else 2.5))))
                if call_amt == 0 or (hu and call_amt <= bb // 2 and pct >= 0.18):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

            if self.tracker.pf_raises == 1:
                if pct >= (0.85 if hu else 0.90) and game_state.can_raise:
                    if hu and pct >= 0.985 and _in_position(game_state, me) and mix < 0.25 and game_state.current_bet <= 4 * bb:
                        return _safe_action(game_state, me, "call")
                    return _safe_action(game_state, me, "raise", int(round(game_state.current_bet * 3.1)))
                if a.multiway_equity >= pot_odds and pct >= (0.38 if hu else 0.58):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")

            if pct >= 0.92:
                if game_state.can_raise:
                    return _safe_action(game_state, me, "raise", game_state.max_raise_to)
                return _safe_action(game_state, me, "call")
            if a.multiway_equity >= pot_odds + 0.04 and pct >= 0.80 and call_amt <= 0.30 * (game_state.stacks[me] + game_state.bets[me]):
                return _safe_action(game_state, me, "call")
            return _safe_action(game_state, me, "fold")

        if a.is_playing_board and call_amt > 0:
            return _safe_action(game_state, me, "fold")

        is_monster = (a.real_cat >= 3) or (a.real_cat == 2 and ehs >= 0.82)

        if call_amt == 0:
            if not game_state.can_raise:
                return _safe_action(game_state, me, "call")

            if is_monster:
                if game_state.street in ("flop", "turn") and a.board_wetness < 0.45 and mix < 0.40:
                    return _safe_action(game_state, me, "call")
                frac = 0.80 if game_state.street == "river" else 0.62
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))

            if (a.real_cat >= 2 or (a.real_cat == 1 and (a.is_overpair or a.is_top_pair))) and ehs >= 0.64:
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.48))

            if a.real_cat == 1 and ehs >= 0.48:
                if game_state.street == "flop" and hu and self.tracker.pf_aggressor == me and a.board_wetness < 0.35 and mix < 0.45:
                    return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.33))
                return _safe_action(game_state, me, "call")

            if game_state.street != "river" and (a.nut_flush_draw or a.straight_outs >= 8) and mix < 0.45:
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.42))

            return _safe_action(game_state, me, "call")

        bet_to_pot = call_amt / max(1, pot - call_amt)
        opp_barrels = self.tracker.max_opp_barrels(me)
        cond_eq = a.conditioned_equity(True, bet_to_pot, opp_barrels)

        if is_monster and cond_eq >= 0.76 and game_state.can_raise:
            if game_state.street == "river" or game_state.spr(me) <= 2.0:
                return _safe_action(game_state, me, "raise", game_state.max_raise_to)
            return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.75))

        implied = 0.05 if (game_state.street != "river" and (a.nut_flush_draw or a.straight_outs >= 8)) else 0.0
        mw_margin = 0.03 * max(0, len(game_state.players_in_hand) - 2)
        if cond_eq + implied >= pot_odds + 0.01 + mw_margin:
            return _safe_action(game_state, me, "call")

        return _safe_action(game_state, me, "fold")


# ---------------------------------------------------------------------------
# BOT 6: RolloutBot ("mc_rollout")
# Deep Stratified Rollout & Effective Hand Strength (EHS / Poki Architecture)
# ---------------------------------------------------------------------------

class RolloutBot:
    def __init__(self, seed: int = 606):
        self.hand = []
        self.player_id = 0
        self.rng = random.Random(seed)
        self.tracker = TableTracker()

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self.tracker.reset_hand()

    def end_game(self, actions: list[PlayerAction], game_state):
        self.tracker.sync(actions, game_state, self.player_id, is_end=True)

    def run(self, actions: list[PlayerAction], game_state) -> PlayerAction:
        self.tracker.sync(actions, game_state, self.player_id, is_end=False)
        n_opp = max(1, len(game_state.players_in_hand) - 1)
        a = HandAnalysis(self.hand, game_state.board, n_opp, self.rng, num_samples=200)
        return self.decide(game_state, a)

    def decide(self, game_state, a: HandAnalysis) -> PlayerAction:
        me = self.player_id
        bb = max(1, game_state.big_blind)
        call_amt = game_state.call_amount
        pot = game_state.pot
        pot_odds = game_state.pot_odds
        hu = len(game_state.players_in_hand) == 2
        pct = a.hu_percentile if hu else a.mw_percentile
        ehs = a.ehs
        hs = a.multiway_equity

        if game_state.street == "preflop":
            eff_bb = game_state.effective_stack(me) / bb
            if eff_bb <= 14.0 and self.tracker.pf_raises == 0 and a.nash_push_bb >= eff_bb:
                return _safe_action(game_state, me, "raise", game_state.max_raise_to)
            if self.tracker.pf_raises == 0:
                if pct >= (0.22 if hu else 0.50):
                    return _safe_action(game_state, me, "raise", int(round(bb * (2.25 if hu else 2.5))))
                if call_amt == 0 or (hu and call_amt <= bb // 2 and pct >= 0.16):
                    return _safe_action(game_state, me, "call")
                return _safe_action(game_state, me, "fold")
            if pct >= (0.82 if hu else 0.88) and game_state.can_raise and self.tracker.pf_raises == 1 and game_state.current_bet <= 5 * bb:
                return _safe_action(game_state, me, "raise", int(round(game_state.current_bet * 3.0)))
            if pct >= 0.92 and game_state.can_raise:
                return _safe_action(game_state, me, "raise", game_state.max_raise_to)
            if hs >= pot_odds and pct >= (0.36 if hu else 0.55):
                return _safe_action(game_state, me, "call")
            return _safe_action(game_state, me, "fold")

        if a.is_playing_board and call_amt > 0:
            return _safe_action(game_state, me, "fold")

        protection_urgency = a.np * 0.55 if a.real_cat >= 1 else 0.0

        if call_amt == 0:
            if not game_state.can_raise:
                return _safe_action(game_state, me, "call")
            if a.real_cat >= 1 and ehs + protection_urgency >= 0.60:
                if game_state.street == "river" and a.real_cat == 1 and not a.is_top_pair and not a.is_overpair:
                    return _safe_action(game_state, me, "call")
                frac = min(0.95, 0.35 + 0.55 * (ehs ** 2) + protection_urgency)
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))
            if a.pp >= 0.22 and _deterministic_mix(self.hand, game_state.board, pot, salt=6) < 0.62:
                return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, 0.45))
            return _safe_action(game_state, me, "call")

        bet_to_pot = call_amt / max(1, pot - call_amt)
        opp_barrels = self.tracker.max_opp_barrels(me)
        cond_eq = a.conditioned_equity(True, bet_to_pot, opp_barrels)

        if a.real_cat >= 2 and cond_eq >= 0.74 and game_state.can_raise:
            frac = min(1.0, 0.55 + 0.50 * (cond_eq ** 2) + protection_urgency)
            if a.real_cat >= 3 and (game_state.spr(me) <= 1.7 or cond_eq >= 0.86):
                return _safe_action(game_state, me, "raise", game_state.max_raise_to)
            return _safe_action(game_state, me, "raise", _raise_by_pot_fraction(game_state, frac))

        if cond_eq >= pot_odds + 0.01:
            return _safe_action(game_state, me, "call")

        return _safe_action(game_state, me, "fold")


# ---------------------------------------------------------------------------
# BOT 7: EnsembleBot / Model ("apex_ensemble")
# Contextual Hedge / Multi-Expert Meta-Controller Combining All 6 Specialists
# ---------------------------------------------------------------------------

class EnsembleBot:
    def __init__(self, seed: int = 707):
        self.hand = []
        self.player_id = 0
        self.rng = random.Random(seed)
        self.tracker = TableTracker()
        self.gto = GTOBot(seed=seed + 1)
        self.bayesian = BayesianBot(seed=seed + 2)
        self.shark = SharkBot(seed=seed + 3)
        self.pressure = PressureBot(seed=seed + 4)
        self.fortress = FortressBot(seed=seed + 5)
        self.rollout = RolloutBot(seed=seed + 6)
        self.experts = [
            ("gto", self.gto),
            ("bayesian", self.bayesian),
            ("shark", self.shark),
            ("pressure", self.pressure),
            ("fortress", self.fortress),
            ("rollout", self.rollout),
        ]
        self.team_weights = {}
        self.last_expert_votes = {}
        self.start_stack = None

    def seed(self, hand: list[Card], player_id: int):
        self.hand = list(hand)
        self.player_id = player_id
        self.tracker.reset_hand()
        self.last_expert_votes = {}
        self.start_stack = None
        for _, bot in self.experts:
            bot.hand = list(hand)
            bot.player_id = player_id
            bot.tracker = self.tracker
        self.bayesian.range_weights = list(_COMBO_COUNTS)

    def end_game(self, actions: list[PlayerAction], game_state):
        self.tracker.sync(actions, game_state, self.player_id, is_end=True)
        for pid, cards in self.tracker.shown_cards.items():
            if pid == self.player_id or len(cards) != 2 or pid >= len(self.tracker.teams):
                continue
            tname = self.tracker.teams[pid]
            c_idx = _canonical_idx(_card_to_int(cards[0]), _card_to_int(cards[1]))
            pct = _HU_PERCENTILE[c_idx]
            old_l = self.bayesian.opp_looseness.get(tname, 1.0)
            self.bayesian.opp_looseness[tname] = 0.85 * old_l + 0.15 * (1.45 - 0.80 * pct)

        if self.start_stack is not None and self.last_expert_votes:
            me = self.player_id
            final_stack = game_state.stacks[me]
            net_bb = (final_stack - self.start_stack) / max(1, game_state.big_blind)
            clipped = max(-5.0, min(5.0, net_bb))
            lr = 0.015
            for pid, tname in enumerate(game_state.teams):
                if pid == me:
                    continue
                w_map = self.team_weights.setdefault(
                    tname, {name: 1.0 for name, _ in self.experts}
                )
                for name, (chosen_kind, was_chosen) in self.last_expert_votes.items():
                    if was_chosen:
                        w_map[name] = max(0.4, min(2.5, w_map[name] * math.exp(lr * clipped)))

    def _context_weights(self, game_state) -> dict[str, float]:
        me = self.player_id
        bb = max(1, game_state.big_blind)
        eff_bb = game_state.effective_stack(me) / bb
        n_in = len(game_state.players_in_hand)
        opps = [p for p in game_state.players_in_hand if p != me]

        w = {
            "gto": 1.35,
            "bayesian": 1.10,
            "shark": 1.20,
            "pressure": 1.00,
            "fortress": 1.25,
            "rollout": 1.30,
        }

        if eff_bb <= 15.0:
            w["fortress"] *= 1.55
            w["gto"] *= 1.30
            w["pressure"] *= 0.70

        if n_in >= 3:
            w["fortress"] *= 1.35
            w["rollout"] *= 1.25
            w["pressure"] *= 0.75

        if opps:
            arch_counts = collections.Counter(
                self.tracker.get_stats(game_state.teams[p]).archetype() for p in opps
            )
            if any(k in arch_counts for k in ("nit", "station", "maniac", "shover")):
                w["shark"] *= 1.65
                w["bayesian"] *= 1.25
            if "maniac" in arch_counts or "shover" in arch_counts:
                w["fortress"] *= 1.40
            if "nit" in arch_counts:
                w["pressure"] *= 1.35

            for p in opps:
                tname = game_state.teams[p]
                if tname in self.team_weights:
                    for k in w:
                        w[k] *= self.team_weights[tname].get(k, 1.0)

        return w

    def run(self, actions: list[PlayerAction], game_state) -> PlayerAction:
        me = self.player_id
        if self.start_stack is None:
            self.start_stack = game_state.stacks[me] + game_state.bets[me]
        self.tracker.sync(actions, game_state, me, is_end=False)

        n_opp = max(1, len(game_state.players_in_hand) - 1)
        analysis = HandAnalysis(self.hand, game_state.board, n_opp, self.rng, num_samples=180)
        weights = self._context_weights(game_state)

        kind_scores = {"fold": 0.0, "call": 0.0, "raise": 0.0}
        raise_weighted_sum = 0.0
        raise_weight_total = 0.0
        expert_actions = {}

        for name, bot in self.experts:
            act = bot.decide(game_state, analysis)
            w = weights[name]
            kind_scores[act.kind] += w
            expert_actions[name] = act.kind
            if act.kind == "raise":
                raise_weighted_sum += w * act.amount
                raise_weight_total += w

        if game_state.call_amount == 0:
            kind_scores["call"] += kind_scores["fold"]
            kind_scores["fold"] = 0.0

        total_continue = kind_scores["call"] + kind_scores["raise"]
        if total_continue >= kind_scores["fold"]:
            if kind_scores["raise"] >= kind_scores["call"] * 0.95 and game_state.can_raise and raise_weight_total > 0:
                best_kind = "raise"
                avg_raise = int(round(raise_weighted_sum / raise_weight_total))
            else:
                best_kind = "call"
                avg_raise = 0
        else:
            best_kind = "fold"
            avg_raise = 0

        self.last_expert_votes = {
            name: (k, k == best_kind) for name, k in expert_actions.items()
        }
        return _safe_action(game_state, me, best_kind, avg_raise)


Model = EnsembleBot

CHALLENGERS = {
    "gto_apex": GTOBot,
    "bayesian": BayesianBot,
    "hud_shark": SharkBot,
    "lag_pressure": PressureBot,
    "fortress": FortressBot,
    "mc_rollout": RolloutBot,
    "apex_ensemble": EnsembleBot,
}
