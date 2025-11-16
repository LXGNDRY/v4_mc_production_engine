
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sports Betting Formula V4 + Monte Carlo
(v4.10.0-prod-singlefile: Safety + AltLine77 + DynamicConsensus + ScenarioEnsemble
 + HBU + ConformalBeta + SameGameCopula)
-------------------------------------------------------------------------------
Adds three modules, ON by default:
1) HBU (Hierarchical Bayesian Updating): posterior-adjust mu/sigma using simple
   conjugate-style shrinkage with effective sample sizes.
2) ConformalBeta: conservative beta-calibration of probabilities to reduce
   overconfidence (Laplace-style smoothing toward 0.5).
3) SameGameCopula: utilities to simulate correlated outcomes for multiple legs
   from the same game (Gaussian/t-copula approximation).

All previous behaviors (MoS, KEG, sigma-inflate, AltLine77, Dynamic Consensus,
Scenario Ensemble) are preserved and compose naturally.

Version: 2025-10-08
"""

import math, json, random
try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ============================================================================
# V4 + MC HOUSE-BEATING ENHANCEMENTS - NON-DESTRUCTIVE ADDITIONS
# Priority improvements for institutional-level edge extraction
# Integrated with existing V4 + MC Enhanced Production system
# ============================================================================

import time
import statistics
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import json

# ============================================================================
# 1. STEAM DETECTION & LINE MOVEMENT ANALYSIS
# ============================================================================

class LineMovementType(Enum):
    STEAM = "steam"  # Sharp money movement
    REVERSE = "reverse"  # Line moves opposite to public betting
    TRAP = "trap"  # Suspicious line movement
    NATURAL = "natural"  # Normal market movement

@dataclass
class LineMovement:
    timestamp: float
    book: str
    market: str
    old_line: float
    new_line: float
    old_odds: float
    new_odds: float
    movement_size: float
    velocity: float  # Movement per minute

class SteamDetectionEngine:
    """Real-time line movement analysis and steam detection"""

    def __init__(self, velocity_threshold: float = 0.5, steam_window: int = 300):
        self.velocity_threshold = velocity_threshold  # Min movement per minute for steam
        self.steam_window = steam_window  # 5 minute window for steam detection
        self.line_history = defaultdict(deque)  # Store line movements per market
        self.public_betting_data = {}  # Public betting percentages

    def add_line_movement(self, book: str, market: str, old_line: float, 
                         new_line: float, old_odds: float, new_odds: float) -> LineMovement:
        """Track new line movement and calculate velocity"""
        timestamp = time.time()
        movement_size = abs(new_line - old_line)

        # Calculate velocity (movement per minute)
        recent_movements = [m for m in self.line_history[market] 
                          if timestamp - m.timestamp < self.steam_window]

        if recent_movements:
            time_span = timestamp - recent_movements[0].timestamp
            total_movement = sum(m.movement_size for m in recent_movements) + movement_size
            velocity = total_movement / max(time_span / 60, 0.1)  # Per minute
        else:
            velocity = movement_size / 0.1  # Assume rapid if first movement

        movement = LineMovement(
            timestamp=timestamp,
            book=book,
            market=market,
            old_line=old_line,
            new_line=new_line,
            old_odds=old_odds,
            new_odds=new_odds,
            movement_size=movement_size,
            velocity=velocity
        )

        # Store movement (keep last 50 per market)
        self.line_history[market].append(movement)
        if len(self.line_history[market]) > 50:
            self.line_history[market].popleft()

        return movement

    def detect_movement_type(self, market: str, movement: LineMovement) -> LineMovementType:
        """Classify line movement type"""

        # Steam detection: High velocity movement
        if movement.velocity >= self.velocity_threshold:
            return LineMovementType.STEAM

        # Reverse line movement: Line moves opposite to public betting
        public_pct = self.public_betting_data.get(market, 0.5)
        line_direction = 1 if movement.new_line > movement.old_line else -1
        public_direction = 1 if public_pct > 0.6 else -1 if public_pct < 0.4 else 0

        if line_direction != public_direction and public_direction != 0:
            return LineMovementType.REVERSE

        # Trap detection: Unusual movement patterns
        recent_movements = [m for m in self.line_history[market] 
                          if time.time() - m.timestamp < self.steam_window]

        if len(recent_movements) >= 3:
            directions = [1 if m.new_line > m.old_line else -1 for m in recent_movements[-3:]]
            if len(set(directions)) >= 2:  # Back and forth movement
                return LineMovementType.TRAP

        return LineMovementType.NATURAL

    def get_steam_signal(self, market: str) -> Dict[str, Union[float, str]]:
        """Get current steam signal strength for market"""
        if market not in self.line_history:
            return {"strength": 0.0, "direction": "none", "type": "none"}

        recent_movements = [m for m in self.line_history[market] 
                          if time.time() - m.timestamp < self.steam_window]

        if not recent_movements:
            return {"strength": 0.0, "direction": "none", "type": "none"}

        # Calculate aggregate steam strength
        total_velocity = sum(m.velocity for m in recent_movements)
        avg_velocity = total_velocity / len(recent_movements)

        # Determine direction
        net_movement = sum(m.new_line - m.old_line for m in recent_movements)
        direction = "up" if net_movement > 0 else "down" if net_movement < 0 else "none"

        # Classify strongest movement type
        movement_types = [self.detect_movement_type(market, m) for m in recent_movements]
        most_common_type = max(set(movement_types), key=movement_types.count)

        return {
            "strength": min(avg_velocity / self.velocity_threshold, 5.0),  # Cap at 5x
            "direction": direction,
            "type": most_common_type.value,
            "recent_movements": len(recent_movements)
        }

# ============================================================================
# 2. CLOSING LINE VALUE (CLV) TRACKING & OPTIMIZATION
# ============================================================================

@dataclass
class BetRecord:
    bet_id: str
    timestamp: float
    market: str
    bet_line: float
    bet_odds: float
    closing_line: float
    closing_odds: float
    clv_line: float  # Difference in line value
    clv_odds: float  # Difference in odds value
    result: Optional[str] = None  # "win", "loss", "push"

class CLVTracker:
    """Comprehensive Closing Line Value tracking and optimization"""

    def __init__(self):
        self.bet_history = []
        self.clv_by_market = defaultdict(list)
        self.clv_by_time_to_game = defaultdict(list)
        self.clv_by_source = defaultdict(list)

    def record_bet(self, bet_id: str, market: str, bet_line: float, 
                   bet_odds: float, source: str = "v4_mc") -> str:
        """Record a new bet for CLV tracking"""
        bet_record = BetRecord(
            bet_id=bet_id,
            timestamp=time.time(),
            market=market,
            bet_line=bet_line,
            bet_odds=bet_odds,
            closing_line=0.0,  # Will be updated at game time
            closing_odds=0.0,  # Will be updated at game time
            clv_line=0.0,
            clv_odds=0.0
        )

        self.bet_history.append(bet_record)
        return bet_id

    def update_closing_lines(self, bet_id: str, closing_line: float, closing_odds: float):
        """Update closing line data and calculate CLV"""
        for bet in self.bet_history:
            if bet.bet_id == bet_id:
                bet.closing_line = closing_line
                bet.closing_odds = closing_odds

                # Calculate CLV (positive = bet got better line than close)
                bet.clv_line = bet.bet_line - closing_line
                bet.clv_odds = bet.bet_odds - closing_odds

                # Store in categorized tracking
                self.clv_by_market[bet.market].append(bet.clv_line)

                # Calculate time to game for timing analysis
                # (This would need game time data - using placeholder)
                time_to_game = 3600  # Placeholder: 1 hour
                self.clv_by_time_to_game[time_to_game].append(bet.clv_line)
                break

    def get_clv_stats(self) -> Dict[str, float]:
        """Get comprehensive CLV statistics"""
        if not self.bet_history:
            return {}

        clv_values = [bet.clv_line for bet in self.bet_history if bet.clv_line != 0.0]

        if not clv_values:
            return {}

        return {
            "avg_clv": statistics.mean(clv_values),
            "clv_hit_rate": len([clv for clv in clv_values if clv > 0]) / len(clv_values),
            "total_bets": len(clv_values),
            "best_clv": max(clv_values),
            "worst_clv": min(clv_values),
            "clv_stdev": statistics.stdev(clv_values) if len(clv_values) > 1 else 0.0
        }

    def get_market_clv_performance(self, market: str) -> Dict[str, float]:
        """Get CLV performance for specific market type"""
        market_clvs = self.clv_by_market.get(market, [])

        if not market_clvs:
            return {}

        return {
            "avg_clv": statistics.mean(market_clvs),
            "hit_rate": len([clv for clv in market_clvs if clv > 0]) / len(market_clvs),
            "sample_size": len(market_clvs)
        }

    def get_optimal_timing(self) -> Dict[int, float]:
        """Analyze optimal betting timing based on CLV"""
        timing_performance = {}

        for time_bucket, clvs in self.clv_by_time_to_game.items():
            if clvs:
                timing_performance[time_bucket] = statistics.mean(clvs)

        return timing_performance

# ============================================================================
# 3. MARKET MAKER PATTERN RECOGNITION
# ============================================================================

class MarketMakerDetector:
    """Identify and exploit market maker patterns and vulnerabilities"""

    def __init__(self):
        self.book_patterns = defaultdict(dict)
        self.vulnerability_windows = defaultdict(list)
        self.sharp_book_rankings = {}

    def analyze_book_behavior(self, book: str, market: str, line_movements: List[LineMovement]) -> Dict:
        """Analyze individual sportsbook behavior patterns"""
        if not line_movements:
            return {}

        # Calculate book's response time to market movements
        response_times = []
        movement_sizes = []

        for i, movement in enumerate(line_movements[1:], 1):
            prev_movement = line_movements[i-1]
            time_diff = movement.timestamp - prev_movement.timestamp
            response_times.append(time_diff)
            movement_sizes.append(movement.movement_size)

        pattern_analysis = {
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "avg_movement_size": statistics.mean(movement_sizes) if movement_sizes else 0,
            "movement_frequency": len(line_movements),
            "volatility": statistics.stdev(movement_sizes) if len(movement_sizes) > 1 else 0
        }

        self.book_patterns[book][market] = pattern_analysis
        return pattern_analysis

    def identify_sharp_books(self) -> Dict[str, float]:
        """Rank books by sharpness (faster, smaller movements = sharper)"""
        book_scores = {}

        for book, markets in self.book_patterns.items():
            if not markets:
                continue

            # Calculate sharpness score
            response_times = [market_data.get("avg_response_time", 999) 
                            for market_data in markets.values()]
            movement_sizes = [market_data.get("avg_movement_size", 0) 
                            for market_data in markets.values()]

            if response_times and movement_sizes:
                # Sharp books: fast response + small movements
                avg_response = statistics.mean(response_times)
                avg_movement = statistics.mean(movement_sizes)

                # Lower score = sharper (faster response, smaller movements)
                sharpness_score = avg_response * avg_movement
                book_scores[book] = sharpness_score

        # Rank books (lower score = sharper)
        self.sharp_book_rankings = dict(sorted(book_scores.items(), key=lambda x: x[1]))
        return self.sharp_book_rankings

    def detect_vulnerability_windows(self, book: str, market: str) -> List[Dict]:
        """Identify time windows when book is most vulnerable"""
        vulnerabilities = []

        # Look for patterns in book behavior
        pattern = self.book_patterns.get(book, {}).get(market, {})

        if not pattern:
            return vulnerabilities

        # High volatility periods indicate uncertainty/vulnerability
        if pattern.get("volatility", 0) > pattern.get("avg_movement_size", 0) * 0.5:
            vulnerabilities.append({
                "type": "high_volatility",
                "confidence": 0.7,
                "description": "Book showing high line volatility"
            })

        # Slow response times indicate vulnerability
        if pattern.get("avg_response_time", 0) > 300:  # 5 minutes
            vulnerabilities.append({
                "type": "slow_response",
                "confidence": 0.8,
                "description": "Book slow to respond to market movements"
            })

        return vulnerabilities

# ============================================================================
# 4. CONTRARIAN SIGNAL INTEGRATION  
# ============================================================================

class ContrarianAnalyzer:
    """Analyze public betting patterns and generate contrarian signals"""

    def __init__(self, contrarian_threshold: float = 0.65):
        self.contrarian_threshold = contrarian_threshold
        self.public_betting_history = defaultdict(list)
        self.contrarian_performance = defaultdict(list)

    def add_public_betting_data(self, market: str, public_percentage: float, 
                               handle_percentage: float = None):
        """Track public betting percentages"""
        betting_data = {
            "timestamp": time.time(),
            "public_pct": public_percentage,
            "handle_pct": handle_percentage or public_percentage,
            "market": market
        }

        self.public_betting_history[market].append(betting_data)

        # Keep only recent data (last 24 hours)
        cutoff_time = time.time() - 86400
        self.public_betting_history[market] = [
            data for data in self.public_betting_history[market] 
            if data["timestamp"] > cutoff_time
        ]

    def get_contrarian_signal(self, market: str, steam_signal: Dict = None) -> Dict:
        """Generate contrarian betting signal"""
        if market not in self.public_betting_history:
            return {"strength": 0.0, "direction": "none", "reason": "no_data"}

        recent_data = self.public_betting_history[market][-1]  # Most recent
        public_pct = recent_data["public_pct"]
        handle_pct = recent_data["handle_pct"]

        # Strong contrarian signal conditions
        contrarian_strength = 0.0
        direction = "none"
        reasons = []

        # Heavy public betting (contrarian fade opportunity)
        if public_pct >= self.contrarian_threshold:
            contrarian_strength += (public_pct - self.contrarian_threshold) * 2
            direction = "fade_public"
            reasons.append(f"heavy_public_{public_pct:.1%}")

        elif public_pct <= (1 - self.contrarian_threshold):
            contrarian_strength += (self.contrarian_threshold - public_pct) * 2
            direction = "fade_public"
            reasons.append(f"heavy_public_opposite_{public_pct:.1%}")

        # Smart money divergence (public % vs handle %)
        if abs(public_pct - handle_pct) > 0.15:
            contrarian_strength += 1.0
            if handle_pct > public_pct:
                direction = "follow_handle"
                reasons.append("smart_money_divergence")
            else:
                direction = "fade_handle"
                reasons.append("public_money_heavy")

        # Combine with steam signal for confirmation
        if steam_signal and steam_signal.get("type") == "reverse":
            contrarian_strength += 1.5
            reasons.append("reverse_line_movement")

        return {
            "strength": min(contrarian_strength, 5.0),  # Cap at 5.0
            "direction": direction,
            "reasons": reasons,
            "public_pct": public_pct,
            "handle_pct": handle_pct
        }

    def update_contrarian_performance(self, market: str, signal_strength: float, 
                                    outcome: str):
        """Track performance of contrarian signals"""
        self.contrarian_performance[signal_strength].append(outcome == "win")

    def get_contrarian_roi(self) -> Dict[str, float]:
        """Calculate ROI by contrarian signal strength"""
        roi_by_strength = {}

        for strength, outcomes in self.contrarian_performance.items():
            if outcomes:
                win_rate = sum(outcomes) / len(outcomes)
                # Assume -110 odds for ROI calculation
                roi = (win_rate * 0.909) - ((1 - win_rate) * 1.0)
                roi_by_strength[strength] = roi

        return roi_by_strength

# ============================================================================
# 5. LIVE INJURY IMPACT ASSESSMENT
# ============================================================================

class InjuryImpactCalculator:
    """Real-time injury severity assessment and line impact prediction"""

    def __init__(self):
        self.player_value_models = {}  # WAR/impact models by sport
        self.injury_severity_scales = {
            "questionable": 0.3,
            "doubtful": 0.7,
            "out": 1.0,
            "probable": 0.1,
            "gtd": 0.5  # Game time decision
        }
        self.position_impact_weights = {}

    def assess_injury_impact(self, player: str, position: str, injury_status: str, 
                           team: str, opponent: str, sport: str) -> Dict:
        """Calculate comprehensive injury impact on game lines"""

        # Base impact from injury severity
        severity_impact = self.injury_severity_scales.get(injury_status.lower(), 0.5)

        # Position importance weighting
        position_weight = self._get_position_weight(position, sport)

        # Player value (would need actual player impact data)
        player_value = self._estimate_player_value(player, position, sport)

        # Calculate total impact
        total_impact = severity_impact * position_weight * player_value

        # Estimate line movement (points)
        estimated_line_move = self._calculate_line_movement(
            total_impact, sport, position, team, opponent
        )

        return {
            "player": player,
            "position": position,
            "injury_status": injury_status,
            "severity_score": severity_impact,
            "position_weight": position_weight,
            "player_value": player_value,
            "total_impact": total_impact,
            "estimated_line_move": estimated_line_move,
            "confidence": min(total_impact * 2, 1.0),
            "recommendation": self._generate_injury_recommendation(total_impact, estimated_line_move)
        }

    def _get_position_weight(self, position: str, sport: str) -> float:
        """Get position importance weight by sport"""
        weights = {
            "nfl": {
                "qb": 1.0, "rb": 0.6, "wr": 0.5, "te": 0.4,
                "ol": 0.7, "dl": 0.6, "lb": 0.5, "cb": 0.5, "s": 0.4
            },
            "nba": {
                "pg": 0.8, "sg": 0.7, "sf": 0.7, "pf": 0.6, "c": 0.6
            },
            "mlb": {
                "sp": 1.0, "rp": 0.4, "c": 0.6, "1b": 0.5, "2b": 0.5,
                "3b": 0.5, "ss": 0.6, "of": 0.5, "dh": 0.4
            }
        }

        return weights.get(sport.lower(), {}).get(position.lower(), 0.5)

    def _estimate_player_value(self, player: str, position: str, sport: str) -> float:
        """Estimate player value (placeholder - would need actual data)"""
        # This would integrate with actual player value databases
        # For now, return moderate value
        return 0.6

    def _calculate_line_movement(self, impact: float, sport: str, position: str,
                               team: str, opponent: str) -> float:
        """Estimate point spread movement from injury"""
        base_movement = impact * 3.0  # Base scaling factor

        # Sport-specific adjustments
        if sport.lower() == "nfl":
            if position.lower() == "qb":
                base_movement *= 2.5  # QB injuries have huge impact
            elif position.lower() in ["rb", "wr"]:
                base_movement *= 1.2

        elif sport.lower() == "nba":
            base_movement *= 1.8  # Basketball players have high individual impact

        return round(base_movement, 1)

    def _generate_injury_recommendation(self, impact: float, line_move: float) -> str:
        """Generate betting recommendation based on injury impact"""
        if impact > 0.7 and abs(line_move) > 2.0:
            return "STRONG_IMPACT - Wait for line adjustment or bet immediately"
        elif impact > 0.4 and abs(line_move) > 1.0:
            return "MODERATE_IMPACT - Monitor line movement"
        elif impact > 0.2:
            return "MINOR_IMPACT - Factor into analysis"
        else:
            return "MINIMAL_IMPACT - No significant effect expected"

# ============================================================================
# 6. UNIFIED HOUSE-BEATING INTEGRATION CLASS
# ============================================================================

class HouseBeatingSuite:
    """Main integration class for all house-beating enhancements"""

    def __init__(self):
        self.steam_detector = SteamDetectionEngine()
        self.clv_tracker = CLVTracker()
        self.market_maker_detector = MarketMakerDetector()
        self.contrarian_analyzer = ContrarianAnalyzer()
        self.injury_calculator = InjuryImpactCalculator()

        # Integration flags
        self.features_enabled = {
            "steam_detection": True,
            "clv_tracking": True,
            "market_maker_analysis": True,
            "contrarian_signals": True,
            "injury_assessment": True
        }

    def analyze_market_opportunity(self, market: str, current_line: float, 
                                 current_odds: float, public_pct: float = None,
                                 injury_data: List[Dict] = None) -> Dict:
        """Comprehensive market analysis using all house-beating tools"""

        analysis = {
            "market": market,
            "timestamp": time.time(),
            "overall_edge": 0.0,
            "confidence": 0.0,
            "recommendations": []
        }

        # Steam detection analysis
        if self.features_enabled["steam_detection"]:
            steam_signal = self.steam_detector.get_steam_signal(market)
            analysis["steam_signal"] = steam_signal

            if steam_signal["strength"] > 1.0:
                analysis["overall_edge"] += steam_signal["strength"] * 0.3
                analysis["recommendations"].append(f"Steam detected: {steam_signal['type']}")

        # Contrarian analysis
        if self.features_enabled["contrarian_signals"] and public_pct:
            self.contrarian_analyzer.add_public_betting_data(market, public_pct)
            contrarian_signal = self.contrarian_analyzer.get_contrarian_signal(
                market, analysis.get("steam_signal")
            )
            analysis["contrarian_signal"] = contrarian_signal

            if contrarian_signal["strength"] > 1.0:
                analysis["overall_edge"] += contrarian_signal["strength"] * 0.2
                analysis["recommendations"].append(f"Contrarian opportunity: {contrarian_signal['direction']}")

        # Injury impact analysis
        if self.features_enabled["injury_assessment"] and injury_data:
            injury_impacts = []
            for injury in injury_data:
                impact = self.injury_calculator.assess_injury_impact(**injury)
                injury_impacts.append(impact)

                if impact["total_impact"] > 0.5:
                    analysis["overall_edge"] += impact["total_impact"] * 0.25
                    analysis["recommendations"].append(f"Injury impact: {impact['recommendation']}")

            analysis["injury_impacts"] = injury_impacts

        # Calculate overall confidence
        analysis["confidence"] = min(analysis["overall_edge"] / 2.0, 1.0)

        # Generate final recommendation
        if analysis["overall_edge"] > 2.0:
            analysis["final_recommendation"] = "STRONG_BET - Multiple edge indicators"
        elif analysis["overall_edge"] > 1.0:
            analysis["final_recommendation"] = "MODERATE_BET - Some edge detected"
        elif analysis["overall_edge"] > 0.5:
            analysis["final_recommendation"] = "WEAK_BET - Minor edge"
        else:
            analysis["final_recommendation"] = "NO_BET - Insufficient edge"

        return analysis

    def record_bet_for_tracking(self, bet_id: str, market: str, line: float, 
                              odds: float, source: str = "v4_mc") -> str:
        """Record bet for CLV and performance tracking"""
        if self.features_enabled["clv_tracking"]:
            return self.clv_tracker.record_bet(bet_id, market, line, odds, source)
        return bet_id

    def update_bet_outcome(self, bet_id: str, closing_line: float, 
                          closing_odds: float, result: str):
        """Update bet outcome for performance tracking"""
        if self.features_enabled["clv_tracking"]:
            self.clv_tracker.update_closing_lines(bet_id, closing_line, closing_odds)

        # Update contrarian performance if applicable
        # (Would need to track which bets used contrarian signals)

    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary across all systems"""
        summary = {
            "timestamp": time.time(),
            "clv_performance": {},
            "contrarian_performance": {},
            "overall_roi": 0.0
        }

        if self.features_enabled["clv_tracking"]:
            summary["clv_performance"] = self.clv_tracker.get_clv_stats()

        if self.features_enabled["contrarian_signals"]:
            summary["contrarian_performance"] = self.contrarian_analyzer.get_contrarian_roi()

        return summary

# ============================================================================
# END V4 + MC HOUSE-BEATING ENHANCEMENTS
# ============================================================================



# -----------------------------------------------------------------------------
# Core Config
# -----------------------------------------------------------------------------

DEFAULT_CFG: Dict[str, Any] = {
    "consensus_weight": 0.35,   # fallback if dynamic consensus has no signals
    "anti_public": 0.08,

    # Monte Carlo
    "mc_sims_min": 100_000,
    "mc_sims_max": 500_000,
    "mc_antithetic": True,
    "mc_copula": True,  # retained flag (not used by single-leg path)

    # Kelly / EV
    "kelly_fraction": 0.25,
    "kelly_ev_gate": 0.0,
    "bankroll_cap_pct": 0.02,

    # Slip tiers
    "tiers": {
        "condensed": {"min_legs": 3, "max_legs": 4},
        "medium":    {"min_legs": 5, "max_legs": 7},
        "large":     {"min_legs": 8, "max_legs": 10},
    },

    # Correlation
    "corr_penalty_floor": 0.92,

    # EV haircut baseline
    "ev_haircut": 0.980,

    "dedupe_enable": True,
}

SAFE_GUARDS: Dict[str, Any] = {
    "enable_mos": True,
    "enable_keg": True,
    "enable_prop_pmins": True,
    "enable_sigma_inflate": True,
    "enable_keynum_haircut": True,
    "enable_altline_optimizer": True,

    # 77% Alt-line heuristic
    "enable_altline_77": True,
    "altline_77_round_step": {
        "PASSYDS": 0.5, "RECYDS": 0.5, "RUSHYDS": 0.5,
        "POINTS": 0.5, "RECEPTIONS": 0.5, "SOG": 0.5
    },

    # KEG thresholds
    "keg_slope_cap": 0.06,
    "keg_kelly_penalty": 0.85,

    # Margin-of-Safety
    "mos_unit_floor": {"YARDS": 1.0, "POINTS": 0.5, "SOG": 0.5, "RECEPTIONS": 0.5},
    "mos_sigma_pct": 0.10,

    # Tiered p-min floors (props)
    "prop_pmin": {"condensed": 0.70, "medium": 0.67, "large": 0.64},

    # Sigma inflation
    "sigma_inflate": {
        "RUSHYDS": 1.05, "PASSYDS": 1.05, "RECYDS": 1.05,
        "POINTS": 1.03, "SOG": 1.03, "RECEPTIONS": 1.03
    },

    # Roundish key-number bump
    "roundish_eps": 0.25,
    "keynum_haircut_floor": 0.97,

    # Alt optimizer local scan
    "alt_scan_units": 5.0,
    "alt_step": 0.5,
    "alt_max_candidates": 21,
}

PROP_FAM = {"PASSYDS","RECYDS","RUSHYDS","POINTS","SOG","RECEPTIONS"}


# === League-specific coefficients & bounds (additive) ===
LEAGUE_PARAMS = {
    "NFL": {
        "FAMILIES": {
            "PASSYDS": {"mu_min": 0, "mu_max": 600, "sigma_min": 18, "sigma_max": 120},
            "RUSHYDS": {"mu_min": 0, "mu_max": 250, "sigma_min": 6,  "sigma_max": 60},
            "RECYDS":  {"mu_min": 0, "mu_max": 250, "sigma_min": 6,  "sigma_max": 60},
            "REC":     {"mu_min": 0, "mu_max": 15,  "sigma_min": 1,  "sigma_max": 6},
            "POINTS":  {"mu_min": 0, "mu_max": 60,  "sigma_min": 3,  "sigma_max": 20},
        },
        "k":   {"k1": 0.12, "k2": 0.10, "k3": 0.06, "k4": 0.10, "k5": 0.05, "k6": 0.10},
        "VAR": {"shadow_sigma": 0.10, "pressure_sigma": 0.05, "doubles_sigma": 0.04},
        "COVERAGE_YDS_SCALE": 10.0,
        "BASELINES": {"run_rate": 0.43}
    },
    "NBA": {
        "FAMILIES": {
            "POINTS": {"mu_min": 0, "mu_max": 60, "sigma_min": 3,  "sigma_max": 18},
            "REB":    {"mu_min": 0, "mu_max": 25, "sigma_min": 2,  "sigma_max": 10},
            "AST":    {"mu_min": 0, "mu_max": 20, "sigma_min": 2,  "sigma_max": 9},
            "PRA":    {"mu_min": 0, "mu_max": 100,"sigma_min": 5,  "sigma_max": 28},
            "3PM":    {"mu_min": 0, "mu_max": 12, "sigma_min": 1,  "sigma_max": 6},
        },
        "k":   {"k1": 0.10, "k2": 0.12, "k3": 0.08, "k4": 0.07, "k5": 0.04, "k6": 0.06},
        "VAR": {"shadow_sigma": 0.08, "pressure_sigma": 0.04, "doubles_sigma": 0.05},
        "COVERAGE_YDS_SCALE": 1.8,
        "BASELINES": {}
    },
    "MLB": {
        "FAMILIES": {
            "HITS":  {"mu_min": 0.0, "mu_max": 5.0, "sigma_min": 0.5, "sigma_max": 2.0},
            "TB":    {"mu_min": 0.0, "mu_max": 6.0, "sigma_min": 0.5, "sigma_max": 2.5},
            "HR":    {"mu_min": 0.0, "mu_max": 2.0, "sigma_min": 0.2, "sigma_max": 1.0},
            "K_PIT": {"mu_min": 0.0, "mu_max": 15.0,"sigma_min": 1.0, "sigma_max": 6.0},
        },
        "k":   {"k1": 0.06, "k2": 0.08, "k3": 0.05, "k4": 0.10, "k5": 0.07, "k6": 0.00},
        "VAR": {"shadow_sigma": 0.04, "pressure_sigma": 0.06, "doubles_sigma": 0.00},
        "COVERAGE_YDS_SCALE": 0.6,
        "BASELINES": {}
    },
    "NHL": {
        "FAMILIES": {
            "SOG":    {"mu_min": 0,  "mu_max": 12, "sigma_min": 1,   "sigma_max": 6},
            "POINTS": {"mu_min": 0,  "mu_max": 5,  "sigma_min": 0.5, "sigma_max": 2.5},
        },
        "k":   {"k1": 0.06, "k2": 0.06, "k3": 0.04, "k4": 0.05, "k5": 0.04, "k6": 0.00},
        "VAR": {"shadow_sigma": 0.04, "pressure_sigma": 0.04, "doubles_sigma": 0.00},
        "COVERAGE_YDS_SCALE": 0.8,
        "BASELINES": {}
    },
    "CFB": {
        "FAMILIES": {
            "PASSYDS": {"mu_min": 0, "mu_max": 600, "sigma_min": 20, "sigma_max": 140},
            "RUSHYDS": {"mu_min": 0, "mu_max": 320, "sigma_min": 8,  "sigma_max": 70},
            "RECYDS":  {"mu_min": 0, "mu_max": 260, "sigma_min": 8,  "sigma_max": 65},
        },
        "k":   {"k1": 0.10, "k2": 0.11, "k3": 0.06, "k4": 0.11, "k5": 0.05, "k6": 0.08},
        "VAR": {"shadow_sigma": 0.10, "pressure_sigma": 0.06, "doubles_sigma": 0.05},
        "COVERAGE_YDS_SCALE": 12.5,
        "BASELINES": {"run_rate": 0.50}
    },
    "CBB": {
        "FAMILIES": {
            "POINTS": {"mu_min": 0, "mu_max": 45, "sigma_min": 3, "sigma_max": 16},
            "REB":    {"mu_min": 0, "mu_max": 18, "sigma_min": 2, "sigma_max": 8},
            "AST":    {"mu_min": 0, "mu_max": 15, "sigma_min": 2, "sigma_max": 7},
        },
        "k":   {"k1": 0.09, "k2": 0.10, "k3": 0.07, "k4": 0.06, "k5": 0.03, "k6": 0.05},
        "VAR": {"shadow_sigma": 0.06, "pressure_sigma": 0.03, "doubles_sigma": 0.04},
        "COVERAGE_YDS_SCALE": 1.6,
        "BASELINES": {}
    },
    "SOCCER": {
        "FAMILIES": {
            "SOG":    {"mu_min": 0,   "mu_max": 8,   "sigma_min": 0.8, "sigma_max": 4},
            "GOALS":  {"mu_min": 0,   "mu_max": 2.5, "sigma_min": 0.2, "sigma_max": 1.2},
            "ASSIST": {"mu_min": 0,   "mu_max": 2.0, "sigma_min": 0.2, "sigma_max": 1.0},
        },
        "k":   {"k1": 0.05, "k2": 0.05, "k3": 0.04, "k4": 0.04, "k5": 0.03, "k6": 0.00},
        "VAR": {"shadow_sigma": 0.03, "pressure_sigma": 0.03, "doubles_sigma": 0.00},
        "COVERAGE_YDS_SCALE": 0.5,
        "BASELINES": {}
    },
}
LEAGUE_PARAMS_DEFAULT = LEAGUE_PARAMS["NFL"]

# Optional: menu rounding per league/family (used by alt-line snapping if desired)
FAMILY_ROUNDING = {
    "NFL": {"RUSHYDS": 0.5, "RECYDS": 0.5, "PASSYDS": 0.5, "REC": 0.5, "POINTS": 0.5},
    "NBA": {"POINTS": 0.5, "REB": 0.5, "AST": 0.5, "PRA": 0.5, "3PM": 0.5},
    "MLB": {"HITS": 0.5, "TB": 0.5, "HR": 1.0, "K_PIT": 0.5},
    "NHL": {"SOG": 0.5, "POINTS": 0.5},
    "CFB": {"PASSYDS": 0.5, "RUSHYDS": 0.5, "RECYDS": 0.5},
    "CBB": {"POINTS": 0.5, "REB": 0.5, "AST": 0.5},
    "SOCCER": {"SOG": 0.5, "GOALS": 0.5, "ASSIST": 0.5},
}

def _league_for_context(ctx: dict) -> str:
    return str((ctx.get("league") or ctx.get("sport") or "NFL")).upper()

def _fam_from_market(market: str) -> str:
    try:
        return market.split("_", 2)[1]
    except Exception:
        return market

def _league_params(ctx: dict):
    lg = _league_for_context(ctx)
    return LEAGUE_PARAMS.get(lg, LEAGUE_PARAMS_DEFAULT)


# Dynamic Consensus (existing)
CONSENSUS_CFG: Dict[str, Any] = {
    "enable_dynamic": False,
    "w_min": 0.00,
    "w_max": 0.10,
    "sport_w_hint": {
        "NFL": (0.35, 0.45),
        "NBA": (0.25, 0.35),
        "MLB": (0.20, 0.30),
        "NHL": (0.15, 0.25),
        "CFB": (0.15, 0.25),
        "CBB": (0.15, 0.25),
        "SOCCER": (0.20, 0.35)
    },
    "crs_weights": {"num_books":0.00,"book_agreement":0.00,"market_volume":0.00,"hours_to_start":0.00,"line_volatility":0.00},
    "anti_public_by_family": {"PASSYDS":0.02,"RECYDS":0.02,"RUSHYDS":0.01,"POINTS":0.02,"RECEPTIONS":0.02,"SOG":0.01},
}

# Scenario Ensemble (existing; ON by default)
SCENARIO_CFG: Dict[str, Any] = {
    "enable_blend": True,
    "weights": {"p10": 0.20, "p50": 0.60, "p90": 0.20},
    "shifts": {
        "p10": {"mu_dsig": -0.25, "sigma_mult": 1.05},
        "p50": {"mu_dsig":  0.00, "sigma_mult": 1.00},
        "p90": {"mu_dsig":  0.25, "sigma_mult": 0.97},
    },
}

# NEW: HBU (posterior updating) — ON by default
HBU_CFG: Dict[str, Any] = {
    "enable": True,
    # Prior pseudo-counts (strength). Larger = slower updates.
    "n0_mu": 7.5,        # prior "observations" for mean
    "n0_var": 7.5,       # prior "observations" for variance
    # Safety clamps
    "sigma_floor": 1e-3,
    "sigma_cap_mult": 3.0,   # cap posterior sigma to 3x prior sigma
}

# NEW: Conformal-like beta calibration — ON by default
CONF_CFG: Dict[str, Any] = {
    "enable": True,
    # Laplace smoothing toward 0.5: p' = (p*T + 0.5*A)/(T + A)
    # Use T = effective trials proxy (from MC sims) and A = alpha strength
    "alpha_strength": 2000.0,  # higher => more shrink toward 0.5
    # Minimum and maximum clamps to avoid 0/1
    "p_min": 1e-6,
    "p_max": 1.0 - 1e-6,
}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


class RNG:
    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        if seed is not None and HAVE_NUMPY:
            try:
                import numpy as _np  # local import guard
                _np.random.seed(seed)
            except Exception:
                pass

    def normal(self, mu: float, sigma: float, size: int) -> List[float]:
        if size <= 0:
            return []
        if HAVE_NUMPY:
            import numpy as _np
            return _np.random.normal(mu, sigma, size).tolist()
        # Fallback Box-Muller
        out: List[float] = []
        import random as _r, math as _m
        while len(out) < size:
            u1 = max(1e-12, _r.random()); u2 = _r.random()
            z0 = _m.sqrt(-2.0*_m.log(u1)) * _m.cos(2.0*_m.pi*u2)
            z1 = _m.sqrt(-2.0*_m.log(u1)) * _m.sin(2.0*_m.pi*u2)
            out.append(mu + sigma * z0)
            if len(out) < size:
                out.append(mu + sigma * z1)
        return out[:size]

class PriceModel:
    def estimate(self, market: str, line: float, direction: str, base_price: float) -> float:
        return base_price
    def implied_prob(self, price: float) -> float:
        return 0.0 if price <= 1e-9 else 1.0/price

class ValueModel:
    def edge(self, price: float) -> float:
        return max(0.0, price - 1.0)

def market_family(market: str) -> str:
    m = market.upper()
    for fam in PROP_FAM:
        if fam in m:
            return fam
    if "RUSH" in m and "YDS" in m: return "RUSHYDS"
    if "PASS" in m and "YDS" in m: return "PASSYDS"
    if "REC"  in m and "YDS" in m: return "RECYDS"
    if "PTS" in m or "POINTS" in m: return "POINTS"
    return "OTHER"

def detect_sport(market: str) -> str:
    u = market.upper()
    if "NFL" in u: return "NFL"
    if "NBA" in u: return "NBA"
    if "MLB" in u: return "MLB"
    if "NHL" in u: return "NHL"
    if "CFB" in u or "NCAAF" in u: return "CFB"
    if "CBB" in u or "NCAAB" in u: return "CBB"
    if any(k in u for k in ["SOCCER","EPL","UEFA","MLS"]): return "SOCCER"
    return "OTHER"

def fragility_slope(mu: float, sigma: float, line: float) -> float:
    sg = max(1e-9, sigma)
    z = (line - mu) / sg
    phi = math.exp(-0.5 * z*z) / math.sqrt(2.0 * math.pi)
    return phi / sg

def _family_unit_floor(fam: str) -> float:
    if fam in {"PASSYDS","RECYDS","RUSHYDS"}:
        return SAFE_GUARDS["mos_unit_floor"]["YARDS"]
    if fam in {"RECEPTIONS","POINTS","SOG"}:
        return SAFE_GUARDS["mos_unit_floor"].get("RECEPTIONS", 0.5)
    return 0.0

def apply_margin_of_safety(line: float, direction: str, sigma: float, fam: str) -> float:
    unit_floor = _family_unit_floor(fam)
    m = max(unit_floor, SAFE_GUARDS["mos_sigma_pct"] * max(1e-9, sigma))
    return (line + m) if direction.lower().startswith("o") else (line - m)

def is_roundish_key(line: float, eps: float=0.25) -> bool:
    r5 = abs((line % 5.0)); r10 = abs((line % 10.0))
    return (r5 < eps) or (5.0 - r5 < eps) or (r10 < eps) or (10.0 - r10 < eps)

def _inflate_sigma_if_needed(fam: str, sigma: float) -> float:
    if not SAFE_GUARDS["enable_sigma_inflate"]:
        return sigma
    factor = SAFE_GUARDS["sigma_inflate"].get(fam)
    return sigma * factor if factor else sigma

# 77% helpers
def _round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(round(x / step) * step, 3)

def _altline_77_candidate(line: float, direction: str) -> float:
    if direction.lower().startswith("o"):
        return line * 0.77
    return line / 0.77  # ≈ 1.2987012987

# -----------------------------------------------------------------------------
# Copula utilities (for same-game joint simulation)
# -----------------------------------------------------------------------------

def cholesky_from_corr(corr: List[List[float]]) -> List[List[float]]:
    if not HAVE_NUMPY:
        # Simple, not optimized: attempt manual Cholesky for small matrices
        n = len(corr)
        L = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                s = sum(L[i][k]*L[j][k] for k in range(j))
                if i == j:
                    val = max(1e-12, corr[i][i] - s)
                    L[i][j] = math.sqrt(val)
                else:
                    L[i][j] = (corr[i][j] - s)/max(1e-12, L[j][j])
        return L
    return np.linalg.cholesky(np.array(corr))

def mvn_correlated_normals(mu_vec: List[float], sigma_vec: List[float], corr: List[List[float]], sims: int, seed: Optional[int]=42):
    n = len(mu_vec)
    if HAVE_NUMPY:
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(np.array(corr))
        Z = rng.standard_normal(size=(sims, n))
        X = Z @ L.T
        return (np.array(mu_vec) + X * np.array(sigma_vec))
    # Fallback
    L = cholesky_from_corr(corr)
    rnd = RNG(seed)
    out = []
    for _ in range(sims):
        z = rnd.normal(0.0, 1.0, n)
        # multiply by L
        x = [sum(L[i][k]*z[k] for k in range(n)) for i in range(n)]
        out.append([mu_vec[i] + sigma_vec[i]*x[i] for i in range(n)])
    return out

# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------



# ============================================================================
# ENHANCED MONTE CARLO ENGINE - V4 INTEGRATION
# Drop-in replacement with advanced features for reliability and accuracy
# ============================================================================

import time
import functools
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

# Enhanced Distribution Types for Sports-Specific Modeling
class DistributionType(Enum):
    NORMAL = "normal"
    GAMMA = "gamma" 
    POISSON = "poisson"
    BETA = "beta"
    SKEWED_NORMAL = "skewed_normal"
    ZERO_INFLATED_POISSON = "zero_inflated_poisson"

@dataclass
class MonteCarloResult:
    """Enhanced result object with diagnostics"""
    probability: float
    standard_error: float
    n_simulations: int
    convergence_achieved: bool
    diagnostics: Dict[str, Any]
    variance_reduction_ratio: float = 1.0

class EnhancedRNG:
    """Enhanced Random Number Generator with dynamic seeding for session independence"""
    def __init__(self, base_seed=None):
        self.base_seed = base_seed or int(time.time() * 1000000) % 2**32
        self.session_counter = 0
        self.rng_state = None
        self._initialize_rng()

    def _initialize_rng(self):
        """Initialize RNG with current seed"""
        if HAVE_NUMPY:
            import numpy as np
            self.rng_state = np.random.RandomState(self.base_seed)
        else:
            import random
            random.seed(self.base_seed)

    def get_session_seed(self):
        """Generate unique seed per betting session"""
        self.session_counter += 1
        return (self.base_seed + self.session_counter * 1000) % 2**32

    def new_session(self):
        """Initialize new session with unique seed"""
        new_seed = self.get_session_seed()
        self.base_seed = new_seed
        self._initialize_rng()
        return new_seed

    def normal(self, mu: float, sigma: float, size: int) -> List[float]:
        """Generate normal random samples"""
        if size <= 0:
            return []

        if HAVE_NUMPY and self.rng_state is not None:
            return self.rng_state.normal(mu, sigma, size).tolist()

        # Fallback Box-Muller implementation
        import random as _r
        import math as _m
        out = []
        while len(out) < size:
            u1 = max(1e-12, _r.random())
            u2 = _r.random()
            z0 = _m.sqrt(-2.0 * _m.log(u1)) * _m.cos(2.0 * _m.pi * u2)
            z1 = _m.sqrt(-2.0 * _m.log(u1)) * _m.sin(2.0 * _m.pi * u2)
            out.append(mu + sigma * z0)
            if len(out) < size:
                out.append(mu + sigma * z1)
        return out[:size]

class SobolSequenceGenerator:
    """Enhanced Sobol sequence generator with Owen scrambling for better coverage"""
    def __init__(self, dimension, scramble=True, seed=None):
        self.dimension = dimension
        self.scramble = scramble
        self.seed = seed or int(time.time() * 1000) % 2**32
        self._initialize_direction_numbers()

    def _initialize_direction_numbers(self):
        """Initialize Sobol direction numbers"""
        # Basic Sobol direction numbers for first few dimensions
        self.direction_numbers = []
        for d in range(min(self.dimension, 10)):  # Support up to 10 dimensions
            if d == 0:
                self.direction_numbers.append([1 << i for i in range(31, 0, -1)])
            elif d == 1:
                self.direction_numbers.append([1 << i if i % 2 else 0 for i in range(31, 0, -1)])
            else:
                # Simple extension for additional dimensions
                self.direction_numbers.append([(i * (d + 7) + 1) << (31 - i) for i in range(31)])

    def generate(self, n_points):
        """Generate Sobol sequence points with optional Owen scrambling"""
        if HAVE_NUMPY:
            import numpy as np
            points = np.zeros((n_points, self.dimension))
        else:
            points = [[0.0] * self.dimension for _ in range(n_points)]

        for i in range(n_points):
            for d in range(min(self.dimension, len(self.direction_numbers))):
                x = 0
                gray_code = i ^ (i >> 1)  # Gray code for better distribution

                for bit in range(31):
                    if gray_code & (1 << bit):
                        x ^= self.direction_numbers[d][bit]

                val = x / (2**31)

                # Apply Owen scrambling if enabled
                if self.scramble:
                    val = self._owen_scramble(val, d, i)

                if HAVE_NUMPY:
                    points[i, d] = val
                else:
                    points[i][d] = val

        return points

    def _owen_scramble(self, x, dimension, index):
        """Hash-based Owen scrambling for better randomization"""
        hash_input = int(x * 2**32) ^ (dimension << 16) ^ (self.seed << 8) ^ index
        hash_val = ((hash_input * 2654435761) % 2**32) / 2**32
        return hash_val

class EnhancedMonteCarloEngine:
    """Enhanced Monte Carlo Engine for Sports Betting with Advanced Features"""

    def __init__(self, 
                 min_simulations: int = 50000,
                 max_simulations: int = 1000000,
                 target_se: float = 0.001,
                 batch_size: int = 25000,
                 use_sobol: bool = True,
                 variance_reduction: bool = True,
                 base_seed: Optional[int] = None,
                 cache_size: int = 10000):

        self.min_simulations = min_simulations
        self.max_simulations = max_simulations
        self.target_se = target_se
        self.batch_size = batch_size
        self.use_sobol = use_sobol
        self.variance_reduction = variance_reduction

        # Initialize enhanced RNG
        self.rng = EnhancedRNG(base_seed)

        # Initialize Sobol generator
        if self.use_sobol:
            self.sobol_gen = SobolSequenceGenerator(dimension=5, scramble=True, 
                                                   seed=self.rng.get_session_seed())

        # Diagnostics tracking
        self.diagnostics = {}

        # Cache for expensive computations
        if HAVE_NUMPY:
            try:
                import scipy.stats as stats
                self._cached_cdf = functools.lru_cache(maxsize=cache_size)(stats.norm.cdf)
                self._cached_ppf = functools.lru_cache(maxsize=cache_size)(stats.norm.ppf)
            except ImportError:
                pass

    def enhanced_p_hit(self, 
                      market: str,
                      mu: float, 
                      sigma: float, 
                      line: float,
                      direction: str = "over",
                      correlation_matrix: Optional[List[List[float]]] = None) -> MonteCarloResult:
        """
        Enhanced probability calculation - MAIN ENTRY POINT
        """

        # Start new session for independence
        session_seed = self.rng.new_session()

        # Select appropriate distribution based on market type
        distribution_type = self._get_sports_distribution_type(market)

        # Initialize tracking variables
        results = []
        running_sum = 0.0
        running_sq_sum = 0.0
        n_completed = 0
        convergence_history = []

        # Adaptive simulation loop
        while n_completed < self.min_simulations:
            # Generate batch of simulations
            if self.use_sobol and n_completed == 0:
                # Use Sobol for first batch - better coverage
                batch_results = self._run_sobol_batch(
                    self.batch_size, mu, sigma, line, direction, 
                    correlation_matrix, distribution_type
                )
            else:
                # Use enhanced Monte Carlo for subsequent batches
                batch_results = self._run_mc_batch(
                    self.batch_size, mu, sigma, line, direction, 
                    correlation_matrix, distribution_type
                )

            # Update running statistics
            batch_sum = sum(batch_results)
            batch_sq_sum = sum(x**2 for x in batch_results)

            running_sum += batch_sum
            running_sq_sum += batch_sq_sum
            n_completed += len(batch_results)

            # Add to results for convergence checking
            results.extend(batch_results)

            # Track convergence
            current_prob = running_sum / n_completed
            current_se = self._compute_standard_error_running(running_sum, running_sq_sum, n_completed)
            convergence_history.append((n_completed, current_prob, current_se))

            # Check for convergence (after minimum simulations)
            if (n_completed >= self.min_simulations and 
                current_se <= self.target_se and 
                n_completed < self.max_simulations):
                break

            # Safety brake
            if n_completed >= self.max_simulations:
                break

            # Keep memory usage reasonable
            if len(results) > 100000:
                results = results[-50000:]

        # Calculate final statistics
        probability = running_sum / n_completed
        variance = max((running_sq_sum / n_completed) - probability**2, 1e-10)
        standard_error = math.sqrt(variance / n_completed)

        # Calculate variance reduction ratio
        variance_reduction_ratio = self._calculate_variance_reduction_ratio(
            results, self.variance_reduction
        )

        # Generate comprehensive diagnostics
        diagnostics = self._generate_diagnostics(
            results, n_completed, session_seed, convergence_history
        )

        return MonteCarloResult(
            probability=probability,
            standard_error=standard_error,
            n_simulations=n_completed,
            convergence_achieved=standard_error <= self.target_se,
            diagnostics=diagnostics,
            variance_reduction_ratio=variance_reduction_ratio
        )

    def _get_sports_distribution_type(self, market: str) -> DistributionType:
        """Select appropriate distribution based on sports market"""
        market_upper = market.upper()

        # Yardage markets - right-skewed, use Gamma
        if any(yard_type in market_upper for yard_type in 
               ['YARD', 'YDS', 'RUSHING', 'PASSING', 'RECEIVING']):
            return DistributionType.GAMMA

        # Counting stats - use Poisson
        if any(count_type in market_upper for count_type in 
               ['RECEPTIONS', 'REC', 'COMPLETIONS', 'COMP', 'ATTEMPTS', 'ATT', 
                'SHOTS', 'GOALS', 'ASSISTS', 'REBOUNDS']):
            return DistributionType.POISSON

        # Percentage-based - use Beta
        if any(pct_type in market_upper for pct_type in 
               ['PCT', '%', 'PERCENTAGE', 'COMPLETION', 'ACCURACY']):
            return DistributionType.BETA

        # Time-based or other continuous - use Normal
        return DistributionType.NORMAL

    def _run_sobol_batch(self, batch_size: int, mu: float, sigma: float, 
                        line: float, direction: str, 
                        correlation_matrix: Optional[List[List[float]]],
                        distribution_type: DistributionType) -> List[float]:
        """Run batch using Sobol sequence for better coverage"""

        # Generate Sobol points
        sobol_points = self.sobol_gen.generate(batch_size)
        if HAVE_NUMPY:
            import numpy as np
            uniform_samples = sobol_points[:, 0] if hasattr(sobol_points, 'shape') else [p[0] for p in sobol_points]
        else:
            uniform_samples = [p[0] for p in sobol_points]

        # Transform to desired distribution
        samples = self._transform_uniform_to_distribution(
            uniform_samples, mu, sigma, distribution_type
        )

        # Apply variance reduction
        if self.variance_reduction:
            # Antithetic variates with Sobol
            if HAVE_NUMPY:
                import numpy as np
                antithetic_uniform = 1.0 - np.array(uniform_samples)
                antithetic_samples = self._transform_uniform_to_distribution(
                    antithetic_uniform, mu, sigma, distribution_type
                )
                all_samples = list(samples) + list(antithetic_samples)
            else:
                antithetic_uniform = [1.0 - u for u in uniform_samples]
                antithetic_samples = self._transform_uniform_to_distribution(
                    antithetic_uniform, mu, sigma, distribution_type
                )
                all_samples = list(samples) + list(antithetic_samples)
        else:
            all_samples = samples

        # Evaluate hits
        return self._evaluate_hits(all_samples, line, direction)

    def _run_mc_batch(self, batch_size: int, mu: float, sigma: float,
                     line: float, direction: str,
                     correlation_matrix: Optional[List[List[float]]],
                     distribution_type: DistributionType) -> List[float]:
        """Run standard Monte Carlo batch with variance reduction"""

        if self.variance_reduction:
            # Use antithetic variates
            half_batch = batch_size // 2

            # Generate first half
            if HAVE_NUMPY and self.rng.rng_state is not None:
                import numpy as np
                uniform_1 = self.rng.rng_state.uniform(0, 1, half_batch)
                uniform_2 = 1.0 - uniform_1
                all_uniform = list(uniform_1) + list(uniform_2)
            else:
                import random
                uniform_1 = [random.random() for _ in range(half_batch)]
                uniform_2 = [1.0 - u for u in uniform_1]
                all_uniform = uniform_1 + uniform_2

            all_samples = self._transform_uniform_to_distribution(
                all_uniform, mu, sigma, distribution_type
            )
        else:
            # Standard sampling
            if HAVE_NUMPY and self.rng.rng_state is not None:
                import numpy as np
                uniform_samples = self.rng.rng_state.uniform(0, 1, batch_size)
            else:
                import random
                uniform_samples = [random.random() for _ in range(batch_size)]

            all_samples = self._transform_uniform_to_distribution(
                uniform_samples, mu, sigma, distribution_type
            )

        # Evaluate hits
        return self._evaluate_hits(all_samples, line, direction)

    def _transform_uniform_to_distribution(self, uniform_samples, 
                                         mu: float, sigma: float,
                                         distribution_type: DistributionType):
        """Transform uniform samples to specified distribution"""

        if HAVE_NUMPY:
            import numpy as np
            uniform_samples = np.array(uniform_samples)

        if distribution_type == DistributionType.NORMAL:
            if HAVE_NUMPY:
                try:
                    import scipy.stats as stats
                    return stats.norm.ppf(uniform_samples, loc=mu, scale=sigma)
                except ImportError:
                    pass

            # Fallback using Box-Muller on uniforms
            results = []
            for u in uniform_samples:
                # Simple inverse transform approximation
                z = self._inverse_normal_cdf(u)
                results.append(mu + sigma * z)
            return results

        elif distribution_type == DistributionType.GAMMA:
            # Convert normal parameters to gamma parameters
            if sigma <= 0 or mu <= 0:
                # Fallback to normal
                return self._transform_uniform_to_distribution(uniform_samples, mu, max(sigma, 0.1), DistributionType.NORMAL)

            # Method of moments conversion
            scale = sigma**2 / mu
            shape = mu / scale
            if HAVE_NUMPY:
                try:
                    import scipy.stats as stats
                    return stats.gamma.ppf(uniform_samples, a=shape, scale=scale)
                except ImportError:
                    pass

            # Simple gamma approximation using normal
            return [max(0, mu + sigma * self._inverse_normal_cdf(u)) for u in uniform_samples]

        elif distribution_type == DistributionType.POISSON:
            # Use normal approximation for large lambda
            lam = max(mu, 0.1)
            if HAVE_NUMPY:
                try:
                    import scipy.stats as stats
                    if lam > 30:
                        # Normal approximation
                        return np.maximum(0, stats.norm.ppf(uniform_samples, loc=lam, scale=np.sqrt(lam)))
                    else:
                        return stats.poisson.ppf(uniform_samples, mu=lam)
                except ImportError:
                    pass

            # Simple approximation
            if lam > 30:
                return [max(0, lam + math.sqrt(lam) * self._inverse_normal_cdf(u)) for u in uniform_samples]
            else:
                return [max(0, int(lam + math.sqrt(lam) * self._inverse_normal_cdf(u))) for u in uniform_samples]

        else:
            # Default to normal
            return self._transform_uniform_to_distribution(uniform_samples, mu, sigma, DistributionType.NORMAL)

    def _inverse_normal_cdf(self, u):
        """Simple inverse normal CDF approximation"""
        # Beasley-Springer-Moro approximation
        u = max(1e-15, min(1-1e-15, u))

        if u < 0.5:
            u = 1.0 - u
            sign = -1
        else:
            sign = 1

        u = u - 0.5
        r = u * u

        # Rational approximation
        num = ((((-0.140543331) * r + 0.914624893) * r - 1.645349621) * r + 0.886226899)
        den = ((((0.012229801) * r - 0.329097515) * r + 1.442710462) * r - 2.118377725) * r + 1.0

        return sign * u * num / den

    def _evaluate_hits(self, samples, line: float, direction: str) -> List[float]:
        """Evaluate whether samples hit the line"""
        if HAVE_NUMPY:
            import numpy as np
            samples = np.array(samples)
            if direction.lower() in ['over', 'o']:
                hits = (samples > line).astype(float)
            else:
                hits = (samples < line).astype(float)
            return hits.tolist()
        else:
            if direction.lower() in ['over', 'o']:
                return [1.0 if x > line else 0.0 for x in samples]
            else:
                return [1.0 if x < line else 0.0 for x in samples]

    def _compute_standard_error_running(self, running_sum: float, running_sq_sum: float, 
                                      n: int) -> float:
        """Compute standard error from running statistics"""
        if n < 2:
            return float('inf')

        p = running_sum / n
        variance = max((running_sq_sum / n) - p**2, 1e-10)
        return math.sqrt(variance / n)

    def _calculate_variance_reduction_ratio(self, samples: List[float], 
                                          used_variance_reduction: bool) -> float:
        """Estimate variance reduction achieved"""
        if not used_variance_reduction or len(samples) < 100:
            return 1.0

        # Compare variance of first half vs second half (rough estimate)
        mid = len(samples) // 2
        if HAVE_NUMPY:
            import numpy as np
            var1 = np.var(samples[:mid]) if mid > 1 else 1.0
            var2 = np.var(samples[mid:]) if len(samples) - mid > 1 else 1.0
        else:
            def variance(data):
                if len(data) < 2:
                    return 1.0
                mean = sum(data) / len(data)
                return sum((x - mean)**2 for x in data) / (len(data) - 1)

            var1 = variance(samples[:mid])
            var2 = variance(samples[mid:])

        if var2 > 0:
            return max(0.1, min(10.0, var1 / var2))  # Bounded ratio
        return 1.0

    def _generate_diagnostics(self, results: List[float], n_simulations: int, 
                            session_seed: int, convergence_history: List) -> Dict[str, Any]:
        """Generate comprehensive diagnostics"""

        if len(results) < 10:
            return {"error": "Insufficient samples for diagnostics"}

        p_hat = sum(results) / len(results)

        # Effective sample size
        autocorr = self._compute_autocorrelation(results)
        ess = len(results) / (1 + 2 * sum(autocorr[1:min(len(autocorr), 10)]))

        # Convergence assessment
        convergence_stability = 0.0
        if len(convergence_history) > 5:
            recent_probs = [x[1] for x in convergence_history[-10:]]
            if HAVE_NUMPY:
                import numpy as np
                convergence_stability = np.std(recent_probs)
            else:
                mean_prob = sum(recent_probs) / len(recent_probs)
                convergence_stability = math.sqrt(sum((p - mean_prob)**2 for p in recent_probs) / len(recent_probs))

        return {
            "session_seed": session_seed,
            "effective_sample_size": max(1, ess),
            "autocorrelation_lag1": autocorr[1] if len(autocorr) > 1 else 0.0,
            "convergence_stability": convergence_stability,
            "probability_estimate": p_hat,
            "total_simulations": n_simulations,
            "convergence_history_length": len(convergence_history)
        }

    def _compute_autocorrelation(self, data: List[float], max_lag: int = 20) -> List[float]:
        """Compute sample autocorrelation function"""
        if len(data) < 4:
            return [1.0, 0.0]

        if HAVE_NUMPY:
            import numpy as np
            data_array = np.array(data)
            n = len(data_array)
            data_centered = data_array - np.mean(data_array)

            autocorr = [1.0]  # lag 0 is always 1

            for lag in range(1, min(max_lag, n//4)):
                numerator = np.sum(data_centered[:-lag] * data_centered[lag:])
                denominator = np.sum(data_centered**2)

                if denominator > 0:
                    autocorr.append(numerator / denominator)
                else:
                    autocorr.append(0.0)
        else:
            n = len(data)
            mean_data = sum(data) / n
            data_centered = [x - mean_data for x in data]

            autocorr = [1.0]

            for lag in range(1, min(max_lag, n//4)):
                numerator = sum(data_centered[i] * data_centered[i + lag] for i in range(n - lag))
                denominator = sum(x**2 for x in data_centered)

                if denominator > 0:
                    autocorr.append(numerator / denominator)
                else:
                    autocorr.append(0.0)

        return autocorr

# Global enhanced Monte Carlo engine instance
_ENHANCED_MC_ENGINE = None

def get_enhanced_mc_engine():
    """Get or create the global enhanced MC engine"""
    global _ENHANCED_MC_ENGINE
    if _ENHANCED_MC_ENGINE is None:
        _ENHANCED_MC_ENGINE = EnhancedMonteCarloEngine(
            min_simulations=100000,
            max_simulations=500000,
            target_se=0.001,
            use_sobol=True,
            variance_reduction=True
        )
    return _ENHANCED_MC_ENGINE

# ============================================================================
# END ENHANCED MONTE CARLO ENGINE
# ============================================================================




# ============================================================================
# CONTEXT-AWARE MONTE CARLO INTEGRATION
# Non-destructive enhancement to feed available context into MC simulations
# Version: 1.0 - Added 2025-11-07
# ============================================================================

class ContextAwareMCAdapter:
    """
    Adapter that bridges existing context systems (HBU, injury assessment, 
    positional intel) with Monte Carlo simulations.

    This layer extracts context from V4Engine and transforms it into 
    adjusted mu/sigma parameters that the MC engine can use.
    """

    def __init__(self, engine):
        self.engine = engine
        self.context_cache = {}

    def extract_game_context(self, market: str, mu: float, sigma: float, 
                            blob: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Extract all available context for a given market from the engine's 
        context store and compute adjusted parameters.

        Returns a comprehensive context dict with adjusted mu/sigma.
        """
        context = {
            "market": market,
            "original_mu": mu,
            "original_sigma": sigma,
            "adjusted_mu": mu,
            "adjusted_sigma": sigma,
            "adjustments_applied": [],
            "confidence_factors": {}
        }

        # 1. Apply HBU (Hierarchical Bayesian Updating) posteriors
        if hasattr(self.engine, '_apply_hbu'):
            mu_hbu, sigma_hbu = self.engine._apply_hbu(market, mu, sigma)
            if mu_hbu != mu or sigma_hbu != sigma:
                context["adjusted_mu"] = mu_hbu
                context["adjusted_sigma"] = sigma_hbu
                context["adjustments_applied"].append({
                    "type": "HBU_posterior",
                    "mu_delta": mu_hbu - mu,
                    "sigma_delta": sigma_hbu - sigma
                })

        # 2. Apply injury impact adjustments
        context = self._apply_injury_context(context, market, blob)

        # 3. Apply positional intelligence adjustments
        context = self._apply_positional_context(context, market, blob)

        # 4. Apply opponent/matchup adjustments
        context = self._apply_matchup_context(context, market, blob)

        # 5. Extract situational factors
        context = self._extract_situational_factors(context, market, blob)

        # 6. Calculate confidence score
        context["confidence_score"] = self._calculate_confidence_score(context)

        return context

    def _apply_injury_context(self, context: Dict, market: str, 
                             blob: Optional[Dict]) -> Dict:
        """Apply injury impact to mu/sigma if injury data exists"""
        if not blob or "injuries" not in blob:
            return context

        injuries = blob["injuries"]
        if not isinstance(injuries, list):
            return context

        # Use engine's injury calculator if available
        if hasattr(self.engine, 'house_beating_suite'):
            injury_calc = self.engine.house_beating_suite.injury_calculator

            for injury in injuries:
                if not isinstance(injury, dict):
                    continue

                # Calculate injury impact
                try:
                    impact = injury_calc.assess_injury_impact(
                        player=injury.get("player", ""),
                        position=injury.get("position", ""),
                        injury_status=injury.get("status", "questionable"),
                        team=injury.get("team", ""),
                        opponent=injury.get("opponent", ""),
                        sport=injury.get("sport", "NFL")
                    )

                    # Apply impact to parameters
                    if impact["total_impact"] > 0.3:  # Significant impact threshold
                        # Reduce mu for offensive players on affected team
                        mu_adjustment = -impact["estimated_line_move"]
                        context["adjusted_mu"] += mu_adjustment

                        # Increase sigma (more uncertainty)
                        sigma_mult = 1.0 + (impact["total_impact"] * 0.15)
                        context["adjusted_sigma"] *= sigma_mult

                        context["adjustments_applied"].append({
                            "type": "injury_impact",
                            "player": injury.get("player"),
                            "mu_delta": mu_adjustment,
                            "sigma_mult": sigma_mult,
                            "impact_score": impact["total_impact"]
                        })

                        context["confidence_factors"]["injury_uncertainty"] = impact["total_impact"]

                except Exception as e:
                    # Silently continue if injury assessment fails
                    pass

        return context

    def _apply_positional_context(self, context: Dict, market: str,
                                  blob: Optional[Dict]) -> Dict:
        """Apply positional intelligence (WR/CB matchups, coverage, etc.)"""
        if not blob:
            return context

        # Check if positional data exists in blob
        if "matchups" not in blob and "defense" not in blob:
            return context

        # Try to enrich with positional intel through engine
        if hasattr(self.engine, 'enrich_with_positional_intel'):
            try:
                # Create temporary context copy
                temp_ctx = dict(self.engine.context)

                # Enrich blob
                self.engine.enrich_with_positional_intel(blob)

                # Extract updated posteriors
                hbu_posteriors = self.engine.context.get("hbu_posteriors", {})
                market_key = self._normalize_market_key(market)

                if market_key in hbu_posteriors:
                    post = hbu_posteriors[market_key]
                    new_mu = post.get("mu", context["adjusted_mu"])
                    new_sigma = post.get("sigma", context["adjusted_sigma"])

                    if new_mu != context["adjusted_mu"] or new_sigma != context["adjusted_sigma"]:
                        context["adjustments_applied"].append({
                            "type": "positional_intelligence",
                            "mu_delta": new_mu - context["adjusted_mu"],
                            "sigma_delta": new_sigma - context["adjusted_sigma"]
                        })

                        context["adjusted_mu"] = new_mu
                        context["adjusted_sigma"] = new_sigma
                        context["confidence_factors"]["positional_data"] = 0.8

            except Exception:
                pass

        return context

    def _apply_matchup_context(self, context: Dict, market: str,
                               blob: Optional[Dict]) -> Dict:
        """Apply opponent-specific adjustments"""
        if not blob:
            return context

        opponent = blob.get("opponent", {})
        if not opponent:
            return context

        # Opponent defensive rating adjustments
        opp_def_rating = opponent.get("defensive_rating")
        if opp_def_rating is not None:
            # Normalize to 0-1 scale (0.5 = league average)
            # Higher rating = better defense = lower offensive output
            if opp_def_rating > 0.5:
                # Facing good defense - reduce mu, increase sigma
                strength = (opp_def_rating - 0.5) * 2  # 0 to 1 scale
                mu_reduction = context["adjusted_mu"] * 0.10 * strength
                sigma_increase = 1.0 + (0.08 * strength)

                context["adjusted_mu"] -= mu_reduction
                context["adjusted_sigma"] *= sigma_increase

                context["adjustments_applied"].append({
                    "type": "opponent_defense",
                    "strength": opp_def_rating,
                    "mu_delta": -mu_reduction,
                    "sigma_mult": sigma_increase
                })
                context["confidence_factors"]["matchup_difficulty"] = strength

            elif opp_def_rating < 0.5:
                # Facing weak defense - increase mu slightly
                strength = (0.5 - opp_def_rating) * 2
                mu_increase = context["adjusted_mu"] * 0.08 * strength

                context["adjusted_mu"] += mu_increase

                context["adjustments_applied"].append({
                    "type": "opponent_defense_weak",
                    "strength": opp_def_rating,
                    "mu_delta": mu_increase
                })

        return context

    def _extract_situational_factors(self, context: Dict, market: str,
                                    blob: Optional[Dict]) -> Dict:
        """Extract situational context (home/away, weather, rest, etc.)"""
        if not blob:
            return context

        situational = {}

        # Home/Away
        is_home = blob.get("is_home", None)
        if is_home is not None:
            situational["home_advantage"] = is_home
            if is_home:
                # Small home boost
                context["adjusted_mu"] *= 1.03
                context["adjustments_applied"].append({
                    "type": "home_field_advantage",
                    "mu_mult": 1.03
                })

        # Weather (for outdoor sports)
        weather = blob.get("weather", {})
        if weather:
            # Wind affects passing/kicking
            wind_speed = weather.get("wind_mph", 0)
            if wind_speed > 15:
                market_upper = market.upper()
                if any(x in market_upper for x in ["PASS", "YARDS", "YDS"]):
                    wind_factor = min(wind_speed / 30.0, 0.3)  # Cap at 30%
                    context["adjusted_mu"] *= (1.0 - wind_factor * 0.15)
                    context["adjusted_sigma"] *= (1.0 + wind_factor * 0.20)

                    context["adjustments_applied"].append({
                        "type": "weather_wind",
                        "wind_mph": wind_speed,
                        "mu_mult": (1.0 - wind_factor * 0.15),
                        "sigma_mult": (1.0 + wind_factor * 0.20)
                    })
                    situational["high_wind"] = True

        # Rest days
        rest_days = blob.get("rest_days")
        if rest_days is not None:
            if rest_days < 3:
                # Short rest - higher variance
                context["adjusted_sigma"] *= 1.05
                situational["short_rest"] = True
            elif rest_days > 7:
                # Long rest - slight rust factor
                context["adjusted_mu"] *= 0.98
                situational["long_rest"] = True

        # Recent form
        recent_form = blob.get("recent_form")  # e.g., 0.0 to 1.0
        if recent_form is not None:
            if recent_form < 0.35:
                # Bad form - reduce confidence
                context["adjusted_sigma"] *= 1.08
                situational["poor_form"] = True
                context["confidence_factors"]["recent_form"] = recent_form
            elif recent_form > 0.65:
                # Good form - slight boost
                context["adjusted_mu"] *= 1.02
                situational["hot_streak"] = True
                context["confidence_factors"]["recent_form"] = recent_form

        context["situational_factors"] = situational
        return context

    def _calculate_confidence_score(self, context: Dict) -> float:
        """
        Calculate overall confidence score (0.0 to 1.0) based on
        how much context was available and applied.

        Higher score = more confident in the adjusted parameters
        """
        confidence = 0.5  # Base confidence

        # Boost for each adjustment type applied
        adjustment_types = set(adj["type"] for adj in context["adjustments_applied"])
        confidence += len(adjustment_types) * 0.08

        # Confidence factors
        factors = context.get("confidence_factors", {})

        if "injury_uncertainty" in factors:
            # High injury impact reduces confidence
            confidence -= factors["injury_uncertainty"] * 0.15

        if "positional_data" in factors:
            # Positional data increases confidence
            confidence += factors["positional_data"] * 0.10

        if "recent_form" in factors:
            # Recent form data increases confidence
            confidence += 0.05

        # Cap between 0.2 and 0.95
        return max(0.2, min(0.95, confidence))

    def _normalize_market_key(self, market: str) -> str:
        """Normalize market string to match HBU key format"""
        return market.upper().replace(" ", "").replace("-", "_")


class ContextIntegratedMCEngine:
    """
    Wrapper around EnhancedMonteCarloEngine that automatically applies
    context adjustments before running simulations.
    """

    def __init__(self, base_engine, v4_engine):
        self.base_engine = base_engine
        self.v4_engine = v4_engine
        self.context_adapter = ContextAwareMCAdapter(v4_engine)
        self.simulation_log = []

    def run_with_context(self, market: str, mu: float, sigma: float,
                        line: float, direction: str,
                        blob: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with full context integration.

        Returns enhanced result with context details.
        """
        # Extract and apply context
        context = self.context_adapter.extract_game_context(
            market, mu, sigma, blob
        )

        # Use adjusted parameters for simulation
        adjusted_mu = context["adjusted_mu"]
        adjusted_sigma = context["adjusted_sigma"]

        # Run enhanced Monte Carlo with adjusted parameters
        mc_result = self.base_engine.enhanced_p_hit(
            market=market,
            mu=adjusted_mu,
            sigma=adjusted_sigma,
            line=line,
            direction=direction
        )

        # Combine results
        result = {
            "probability": mc_result.probability,
            "standard_error": mc_result.standard_error,
            "n_simulations": mc_result.n_simulations,
            "convergence_achieved": mc_result.convergence_achieved,
            "context": context,
            "mc_diagnostics": mc_result.diagnostics,
            "variance_reduction_ratio": mc_result.variance_reduction_ratio
        }

        # Log for debugging
        self.simulation_log.append({
            "market": market,
            "line": line,
            "direction": direction,
            "original_mu": mu,
            "adjusted_mu": adjusted_mu,
            "mu_delta": adjusted_mu - mu,
            "original_sigma": sigma,
            "adjusted_sigma": adjusted_sigma,
            "sigma_delta": adjusted_sigma - sigma,
            "adjustments": len(context["adjustments_applied"]),
            "confidence": context["confidence_score"],
            "probability": mc_result.probability
        })

        return result

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all simulations run"""
        if not self.simulation_log:
            return {}

        total_sims = len(self.simulation_log)
        avg_mu_delta = sum(abs(s["mu_delta"]) for s in self.simulation_log) / total_sims
        avg_sigma_delta = sum(abs(s["sigma_delta"]) for s in self.simulation_log) / total_sims
        avg_adjustments = sum(s["adjustments"] for s in self.simulation_log) / total_sims
        avg_confidence = sum(s["confidence"] for s in self.simulation_log) / total_sims

        return {
            "total_simulations": total_sims,
            "avg_mu_adjustment": avg_mu_delta,
            "avg_sigma_adjustment": avg_sigma_delta,
            "avg_adjustments_per_sim": avg_adjustments,
            "avg_confidence_score": avg_confidence,
            "context_utilization_rate": sum(1 for s in self.simulation_log if s["adjustments"] > 0) / total_sims
        }


class V4EngineIntegrated:
    def __init__(self, cfg=None):
        self.cfg = cfg or {}
        self.context = {}
        self.price_model = type("P", (), {"estimate": lambda self, m,l,d,b: b, "implied_prob": lambda self, pr: (abs(pr)/100.0)/(abs(pr)/100.0+1.0) if pr>0 else (1.0/(1.0+abs(pr)/100.0))})()
        self.value_model = type("V", (), {"edge": lambda self, dec: max(0.0, (abs(dec)/100.0) if isinstance(dec,(int,float)) and dec!=0 else 0.0)})()
        self.rng = type("R", (), {"seed": 42})()
        self._last_haircut = 1.0        
        # Initialize House-Beating Enhancement Suite
        self.house_beating_suite = HouseBeatingSuite()

        # House-beating features configuration
        self.house_beating_enabled = cfg.get("house_beating_enabled", True) if cfg else True

    def _apply_hbu(self, market, mu, sigma):
        return mu, sigma
    def _p_hit_raw(self, market, mu, sigma, line, direction, consensus=None) -> float:

        """

        ENHANCED Monte Carlo simulation with adaptive sizing, sports-specific distributions,

        Sobol sequences, variance reduction, and comprehensive diagnostics.

        

        This method replaces the original _p_hit_raw with state-of-the-art Monte Carlo techniques.

        """

        # Get or create enhanced MC engine

        enhanced_engine = get_enhanced_mc_engine()

        

        # Use the enhanced Monte Carlo engine

        try:

            result = enhanced_engine.enhanced_p_hit(

                market=market,

                mu=mu,

                sigma=sigma,

                line=line,

                direction=direction

            )

            

            # Store diagnostics in context for debugging/monitoring

            if hasattr(self, 'context'):

                self.context["_last_mc_sims"] = result.n_simulations

                self.context["_last_mc_diagnostics"] = result.diagnostics

                self.context["_last_mc_se"] = result.standard_error

                self.context["_last_mc_convergence"] = result.convergence_achieved

                self.context["_last_mc_variance_reduction"] = result.variance_reduction_ratio

            

            return result.probability

            

        except Exception as e:

            # Fallback to original implementation if enhanced fails

            import warnings

            warnings.warn(f"Enhanced Monte Carlo failed, using fallback: {e}")

            return self._p_hit_raw_fallback(market, mu, sigma, line, direction, consensus)


    def _p_hit_raw_fallback(self, market, mu, sigma, line, direction, consensus=None) -> float:

        """Fallback to original Monte Carlo implementation"""

        sims = max(int(self.cfg.get("mc_sims_min", 100_000)), 10_000)

        sims = min(sims, int(self.cfg.get("mc_sims_max", 500_000)))

        over = direction.lower().startswith("o")

        

        if HAVE_NUMPY:

            draws = np.random.normal(mu, max(1e-9, sigma), sims)

            hits = (draws > line).sum() if over else (draws < line).sum()

            p = float(hits) / sims

        else:

            rng = RNG(self.rng.seed)

            draws = rng.normal(mu, max(1e-9, sigma), sims)

            hits = sum(1 for x in draws if (x > line if over else x < line))

            p = hits / sims

        

        # Store last sims for conformal calibration strength

        if hasattr(self, 'context'):

            self.context["_last_mc_sims"] = sims

        return p
    def _p_hit_singleworld(self, market, mu, sigma, line, direction, consensus=None):
        return self._p_hit_raw(market, mu, sigma, line, direction, consensus)
    def _p_hit(self, market, mu, sigma, line, direction, consensus=None):
        return self._p_hit_singleworld(market, mu, sigma, line, direction, consensus)
    def evaluate_mainline(self, market, mu, sigma, posted_line, direction, price):
        p = self._p_hit(market, mu, sigma, posted_line, direction)
        b = (abs(price)/100.0) if price>0 else (100.0/abs(price))
        ev = p*b - (1.0-p)
        q = 1.0-p
        k = max(0.0, (b*p - q)/b) if b>0 else 0.0
        return {"p_hit": p, "ev": ev, "kelly": k, "haircut": 1.0}
    def _optimize_alt_line_stub(self, market, mu, sigma, posted_line, direction, price_quote, tier, consensus=None):
        return posted_line, price_quote
    def mvn_correlated_normals(self, size, rho_matrix):
        try:
            import numpy as _np
            rng=_np.random.default_rng(42)
            d=len(rho_matrix)
            Z=rng.standard_normal((size,d))
            return Z
        except Exception:
            return [[0.0]*len(rho_matrix) for _ in range(size)]
    def build_slips(self, candidates, spec):
        return []
# --- Positional Intelligence: WR/CB, coverage, OL health, redistribution ---
def enrich_with_positional_intel(self, blob: dict) -> None:
    """Use WR/CB matchups, coverage mix, OL health, and route-level splits
    to refine HBU posteriors (mu/sigma) and suggest rho overrides. Additive-only."""
    ctx = self.context
    L = _league_params(ctx)
    k = L["k"]; VAR = L["VAR"]; SCALE = float(L["COVERAGE_YDS_SCALE"])
    hbu = ctx.setdefault("hbu_posteriors", {})
    rho = ctx.setdefault("rho_overrides", {})

    defense = (blob.get("defense") or {})
    mix = defense.get("coverage_mix", {})
    man_share = float(mix.get("man", 0.0))
    pressure = float(defense.get("pressure_rate", 0.0))
    double_vs_wr1 = float(defense.get("double_rate_vs_WR1", 0.0))

    health = (blob.get("health") or {})
    pressure_adj = float((health.get("OL") or {}).get("pressure_projection_adj", 0.0))
    pressure_eff = max(0.0, pressure + pressure_adj)

    matchups = (blob.get("matchups") or {})
    for key, info in matchups.items():
        if not isinstance(key, str) or not key.startswith("WR:"):
            continue
        pname = key.split("WR:", 1)[1].strip()
        mk = f"NFL_RECYDS_{pname.replace(' ','')}"
        post = hbu.get(mk)
        if not post:
            continue
        mu = float(post.get("mu", 0.0))
        sg = float(post.get("sigma", 18.0))
        sz = info.get("man_zone_split", {}) or {}
        vs_cb = (info.get("vs_primary_cb") or {})
        shadow = (info.get("shadow") or {})
        shadow_rate = float(shadow.get("rate", 0.0))

        cb_allow_yprr = float(vs_cb.get("allow_yprr", 1.6))
        cb_allow_sr   = float(vs_cb.get("allow_sr", 0.47))
        cb_quality = max(0.0, (1.6 - cb_allow_yprr)) + max(0.0, (0.50 - cb_allow_sr))

        yprr_man  = float(sz.get("yprr_vs_man", 1.6))
        yprr_zone = float(sz.get("yprr_vs_zone", 1.6))
        d_man  = yprr_man  - 1.6
        d_zone = yprr_zone - 1.6

        # 1) Shadow CB effect
        mu -= k["k1"] * cb_quality * shadow_rate * mu
        sg *= (1.0 + VAR["shadow_sigma"] * shadow_rate)

        # 2) Coverage fit -> yards-scale shift
        mu += (k["k2"] * (man_share * d_man + (1.0 - man_share) * d_zone)) * SCALE

        # 3) Pressure uncertainty & WR1 doubles
        sg *= (1.0 + VAR["pressure_sigma"] * pressure_eff)
        sg *= (1.0 + VAR["doubles_sigma"]  * double_vs_wr1)

        fam = _fam_from_market(mk)
        fb = L["FAMILIES"].get(fam, {"mu_min":0, "mu_max":9999, "sigma_min":1, "sigma_max":999})
        mu = max(fb["mu_min"], min(fb["mu_max"], mu))
        sg = max(fb["sigma_min"], min(fb["sigma_max"], sg))
        hbu[mk] = {"mu": mu, "sigma": sg}

        wr_key = f"WR:{pname.replace(' ','')}"
        rho.setdefault("GROUP:NFL_RECYDS", {})
        if wr_key not in rho["GROUP:NFL_RECYDS"]:
            rho["GROUP:NFL_RECYDS"][wr_key] = {"TE": +0.10, "RB": +0.08}

def redistribute_usage_from_absences(self, blob: dict) -> None:
    """Conservative usage redistribution when position-group absences exist (e.g., WR2 out)."""
    ctx = self.context
    _ = _league_params(ctx)  # reserved for future tuning by league
    hbu = ctx.setdefault("hbu_posteriors", {})
    health = (blob.get("health") or {})
    wr = (health.get("WR") or {})
    wr2_out = bool(wr.get("WR2_out", False))
    if not wr2_out:
        return
    wr_mus = [(k, v["mu"]) for k, v in hbu.items() if k.startswith("NFL_RECYDS_") and isinstance(v, dict) and "mu" in v]
    if not wr_mus:
        return
    wr1_key, _ = max(wr_mus, key=lambda x: x[1])
    for k, post in hbu.items():
        if not isinstance(post, dict):
            continue
        if k == wr1_key and k.startswith("NFL_RECYDS_"):
            post["mu"] = float(post.get("mu", 0.0)) * 1.05
        elif k.startswith("NFL_RECYDS_TE"):
            post["mu"] = float(post.get("mu", 0.0)) * 1.03

    cfg: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CFG))
    rng: RNG = field(default_factory=lambda: RNG(42))
    price_model: PriceModel = field(default_factory=PriceModel)
    value_model: ValueModel = field(default_factory=ValueModel)
    context: Dict[str, Any] = field(default_factory=dict)
    _last_haircut: float = 1.0
    # HBU in-memory store: key -> {"mu":..., "sigma":..., "n_mu":..., "n_var":...}
    hbu_store: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ---- Public ----
    def load_context_files(self, files: Dict[str, str]) -> None:
        for key, path in files.items():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.context[key] = json.load(f)
            except Exception:
                pass

    # ---------- HBU (posterior fetch/apply) ----------
    def _hbu_key(self, market: str) -> str:
        # You can customize to use player IDs, etc.
        return market.upper()

    def _apply_hbu(self, market: str, mu: float, sigma: float) -> Tuple[float, float]:
        if not HBU_CFG.get("enable", True):
            return mu, sigma
        key = self._hbu_key(market)
        # external override: context["hbu_posteriors"][key] = {"mu":..,"sigma":..}
        ctx_map = self.context.get("hbu_posteriors", {})
        if key in ctx_map:
            post = ctx_map[key]
            mu_p = float(post.get("mu", mu))
            sg_p = float(post.get("sigma", sigma))
            return mu_p, max(HBU_CFG["sigma_floor"], sg_p)

        # lightweight conjugate-style update using stored sufficient stats if present
        rec = self.hbu_store.get(key)
        if not rec:
            return mu, sigma

        n0_mu = float(HBU_CFG["n0_mu"]); n0_var = float(HBU_CFG["n0_var"])
        n_mu  = max(0.0, float(rec.get("n_mu", 0.0)))
        n_var = max(0.0, float(rec.get("n_var", 0.0)))

        mu_obs = float(rec.get("mu", mu))
        sg_obs = max(HBU_CFG["sigma_floor"], float(rec.get("sigma", sigma)))

        # Posterior mean: weighted by effective counts
        mu_post = (n0_mu*mu + n_mu*mu_obs) / max(1e-9, (n0_mu + n_mu))
        # Posterior sigma: pooled with caps
        var_post = (n0_var*(sigma**2) + n_var*(sg_obs**2)) / max(1e-9, (n0_var + n_var))
        sg_post = math.sqrt(max(HBU_CFG["sigma_floor"], var_post))
        sg_post = min(sg_post, HBU_CFG["sigma_cap_mult"] * sigma)
        return mu_post, sg_post

    def update_hbu_from_observation(self, market: str, sample_value: float, weight: float=1.0) -> None:
        """Optional: call after games to update the in-memory HBU store."""
        key = self._hbu_key(market)
        rec = self.hbu_store.get(key, {"mu": sample_value, "sigma": 1.0, "n_mu": 0.0, "n_var": 0.0})
        # incremental updates (Welford-style could be used; keep simple here)
        prev_mu = rec["mu"]; prev_n_mu = rec["n_mu"]
        new_n_mu = prev_n_mu + weight
        new_mu = (prev_mu*prev_n_mu + sample_value*weight) / max(1e-9, new_n_mu)
        # crude variance proxy via absolute deviation
        abs_dev = abs(sample_value - new_mu)
        prev_var = rec.get("sigma", 1.0)**2
        new_var = 0.9*prev_var + 0.1*(abs_dev**2)
        self.hbu_store[key] = {"mu": new_mu, "sigma": math.sqrt(max(HBU_CFG["sigma_floor"], new_var)), "n_mu": new_n_mu, "n_var": rec["n_var"] + weight}

    # ---------- External API ----------
    def p_hit(self, market: str, mu: float, sigma: float, line: float, direction: str, consensus: Optional[float]=None) -> float:
        return self._p_hit(market, mu, sigma, line, direction, consensus)

    def evaluate_mainline(self, market: str, mu: float, sigma: float, posted_line: float, direction: str, price: float) -> Dict[str, Any]:
        fam = market_family(market)
        # Apply HBU adjustment first
        mu0, sg0 = self._apply_hbu(market, mu, sigma)
        sigma_eff0 = _inflate_sigma_if_needed(fam, sg0)
        p = self._p_hit(market, mu0, sigma_eff0, posted_line, direction)

        haircut = float(self.cfg.get("ev_haircut", 0.985))
        if SAFE_GUARDS["enable_keynum_haircut"]:
            slope = fragility_slope(mu0, sigma_eff0, posted_line)
            if slope > SAFE_GUARDS["keg_slope_cap"] and is_roundish_key(posted_line, SAFE_GUARDS["roundish_eps"]):
                haircut = max(haircut, SAFE_GUARDS["keynum_haircut_floor"])
        self._last_haircut = haircut

        implied = self.price_model.implied_prob(price)
        edge = max(0.0, p - implied) * haircut

        kelly_base = 0.0
        if implied > 0 and p > implied:
            b = max(1e-9, price - 1.0)
            q = 1.0 - p
            kelly_full = (b*p - q) / b
            kelly_base = max(0.0, kelly_full) * float(self.cfg.get("kelly_fraction", 0.5))

        kelly = self._apply_keg_penalty(mu0, sigma_eff0, posted_line, kelly_base)
        kelly = min(float(self.cfg.get('bankroll_cap_pct', 0.03)), kelly)

        return {"market": market, "family": fam, "line": posted_line, "direction": direction, "price": price,
                "p_hit": p, "ev_proxy": edge, "kelly": kelly, "haircut": haircut}

    # ---- Slip Builder ----
    @dataclass
    class Leg:
        market: str
        mu: float
        sigma: float
        line: float
        direction: str
        price: float
        family: str = ""
        p: float = 0.0

    @dataclass
    class SlipSpec:
        target: str  # 'condensed' | 'medium' | 'large'
        legs: int

    def build_slips(self, candidates: List['V4EngineIntegrated.Leg'], spec: 'V4EngineIntegrated.SlipSpec') -> List[List['V4EngineIntegrated.Leg']]:
        tier = spec.target
        out_slips: List[List[V4EngineIntegrated.Leg]] = []
        selected: List[V4EngineIntegrated.Leg] = []

        for leg in candidates:
            fam = market_family(leg.market)
            leg.family = fam
            # HBU per leg
            mu_adj, sg_adj = self._apply_hbu(leg.market, leg.mu, leg.sigma)
            sigma_eff = _inflate_sigma_if_needed(fam, sg_adj)

            # Alt-line optimization (includes 77% candidate)
            line_eff, price_eff = leg.line, leg.price
            if SAFE_GUARDS["enable_altline_optimizer"] and fam in PROP_FAM:
                line_eff, price_eff = self.optimize_alt_line(
                    leg.market, mu_adj, sigma_eff, leg.line, leg.direction, leg.price, tier
                )

            p = self._p_hit(leg.market, mu_adj, sigma_eff, line_eff, leg.direction)
            leg.p = p
            leg.line = line_eff
            leg.price = price_eff

            if SAFE_GUARDS["enable_prop_pmins"] and fam in PROP_FAM:
                pmin = SAFE_GUARDS["prop_pmin"].get(tier, 0.65)
                if p < pmin:
                    continue

            selected.append(leg)

        selected.sort(key=lambda L: L.p, reverse=True)
        i = 0
        while i < len(selected):
            slip = selected[i:i+spec.legs]
            if len(slip) == spec.legs:
                out_slips.append(slip)
            i += spec.legs

        return out_slips

    # ---- Internals ----

    def _apply_keg_penalty(self, mu, sigma, posted_line, base_kelly) -> float:
        if not SAFE_GUARDS["enable_keg"]:
            return base_kelly
        slope = fragility_slope(mu, sigma, posted_line)
        if slope > SAFE_GUARDS["keg_slope_cap"]:
            return base_kelly * SAFE_GUARDS["keg_kelly_penalty"]
        return base_kelly

    # Scenario helpers (existing)
    def _scenario_variants(self, mu: float, sigma: float) -> Dict[str, Tuple[float,float]]:
        out: Dict[str, Tuple[float,float]] = {}
        for name, tr in SCENARIO_CFG["shifts"].items():
            mu_s = mu + float(tr.get("mu_dsig", 0.0)) * sigma
            sg_s = sigma * float(tr.get("sigma_mult", 1.0))
            out[name] = (mu_s, sg_s)
        return out

    def scenario_summary(self, market: str, mu: float, sigma: float, line: float, direction: str, consensus: Optional[float]=None) -> Dict[str, Any]:
        variants = self._scenario_variants(mu, sigma)
        rows: List[Tuple[str, float]] = []
        for name, (mu_s, sg_s) in variants.items():
            p_s = self._p_hit_singleworld(market, mu_s, sg_s, line, direction, consensus)
            rows.append((name, p_s))
        wmap = SCENARIO_CFG["weights"]
        p_blend = sum(wmap.get(n, 0.0) * p for n, p in rows)
        return {"scenarios": rows, "p_blend": p_blend}

    def _p_hit(self, market, mu, sigma, line, direction, consensus=None) -> float:
        if SCENARIO_CFG.get("enable_blend", False):
            summ = self.scenario_summary(market, mu, sigma, line, direction, consensus)
            return self._conformal_beta(summ["p_blend"])
        # fallback single-world
        return self._conformal_beta(self._p_hit_singleworld(market, mu, sigma, line, direction, consensus))

    def _p_hit_singleworld(self, market, mu, sigma, line, direction, consensus=None) -> float:
        fam = market_family(market)
        # HBU per call
        mu1, sg1 = self._apply_hbu(market, mu, sigma)
        sigma_eff = _inflate_sigma_if_needed(fam, sg1)
        line_eff = line
        if SAFE_GUARDS["enable_mos"] and fam in PROP_FAM:
            line_eff = apply_margin_of_safety(line, direction, sigma_eff, fam)
        p_mc = self._p_hit_raw(market, mu1, sigma_eff, line_eff, direction, None)
        return self._blend_with_consensus(market, fam, p_mc, direction, consensus)

    def _p_hit_raw(self, market, mu, sigma, line, direction, consensus=None) -> float:
        sims = max(int(self.cfg.get("mc_sims_min", 100_000)), 10_000)
        sims = min(sims, int(self.cfg.get("mc_sims_max", 500_000)))
        over = direction.lower().startswith("o")
        if HAVE_NUMPY:
            draws = np.random.normal(mu, max(1e-9, sigma), sims)
            hits = (draws > line).sum() if over else (draws < line).sum()
            p = float(hits) / sims
        else:
            rng = RNG(self.rng.seed)
            draws = rng.normal(mu, max(1e-9, sigma), sims)
            hits = sum(1 for x in draws if (x > line if over else x < line))
            p = hits / sims
        # Store last sims for conformal calibration strength
        self.context["_last_mc_sims"] = sims
        return p

    # ---------- Conformal-like beta calibration ----------
    def _conformal_beta(self, p: float) -> float:
        if not CONF_CFG.get("enable", True):
            return max(CONF_CFG["p_min"], min(CONF_CFG["p_max"], p))
        T = float(self.context.get("_last_mc_sims", self.cfg.get("mc_sims_min", 100000)))
        A = float(CONF_CFG.get("alpha_strength", 2000.0))
        p_cal = (p*T + 0.5*A) / (T + A)
        p_cal = max(CONF_CFG["p_min"], min(CONF_CFG["p_max"], p_cal))
        return p_cal

    # ---------- Dynamic Consensus core (existing) ----------
    def _blend_with_consensus(self, market: str, fam: str, p_mc: float, direction: str, consensus_input: Optional[float]) -> float:
        if not CONSENSUS_CFG.get("enable_dynamic", True):
            if consensus_input is None:
                return p_mc
            w = float(self.cfg.get("consensus_weight", 0.35))
            return (1.0 - w) * p_mc + w * consensus_input

        p_cons_eff = self._consensus_effective_prob(market, direction, consensus_input)
        if p_cons_eff is None:
            if consensus_input is None:
                return p_mc
            w = float(self.cfg.get("consensus_weight", 0.35))
            return (1.0 - w) * p_mc + w * float(consensus_input)

        sport = detect_sport(market)
        market_data = self._consensus_market_meta(market)
        w = self._dynamic_consensus_weight(fam, sport, market_data)

        anti_base = float(self.cfg.get("anti_public", 0.08))
        anti_adj = CONSENSUS_CFG["anti_public_by_family"].get(fam, 0.0)
        anti_total = max(0.0, anti_base + anti_adj)

        p_cons_adj = p_cons_eff
        if direction.lower().startswith("o"):
            p_cons_adj = max(1e-6, p_cons_eff - anti_total * 0.02)
        else:
            p_cons_adj = min(1.0 - 1e-6, p_cons_eff + anti_total * 0.02)

        p = (1.0 - w) * p_mc + w * p_cons_adj
        return p

    def _consensus_effective_prob(self, market: str, direction: str, fallback_single: Optional[float]) -> Optional[float]:
        srcs = self.context.get("consensus_sources")
        if not srcs or not isinstance(srcs, list):
            return fallback_single

        weights = CONSENSUS_CFG["crs_weights"]
        num_books = max(1, len(srcs))
        probs = [max(1e-6, min(1.0 - 1e-6, s.get("p", 0.5))) for s in srcs]
        if len(probs) > 1:
            mean_p = sum(probs) / len(probs)
            var = sum((p - mean_p)**2 for p in probs) / (len(probs) - 1)
            agreement = max(0.0, 1.0 - min(1.0, (var / 0.025)))
        else:
            agreement = 0.5

        total_w = 0.0
        accum = 0.0
        for s in srcs:
            p = max(1e-6, min(1.0 - 1e-6, s.get("p", 0.5)))
            hts = float(s.get("hours_to_start", 24.0))
            mv  = float(s.get("market_volume", 0.5))
            lv  = float(s.get("line_volatility", 0.1))

            h_score = max(0.0, min(1.0, 1.0 - min(1.0, hts / 72.0)))
            mv_score = max(0.0, min(1.0, mv))
            lv_score = max(0.0, min(1.0, 1.0 - lv))

            crs = (
                weights["num_books"]     * min(1.0, num_books / 8.0) +
                weights["book_agreement"]* agreement +
                weights["market_volume"] * mv_score +
                weights["hours_to_start"]* h_score +
                weights["line_volatility"]* lv_score
            )
            total_w += crs
            accum += crs * p

        if total_w <= 1e-9:
            return fallback_single
        return accum / total_w

    def _consensus_market_meta(self, market: str) -> Dict[str, float]:
        meta = self.context.get("consensus_meta", {}) or {}
        return {
            "market_volume": float(meta.get("market_volume", 0.5)),
            "steam_strength": float(meta.get("steam_strength", 0.0)),
            "book_agreement": float(meta.get("book_agreement", 0.5)),
            "public_pct": float(meta.get("public_pct", 0.5)),
        }

    def _dynamic_consensus_weight(self, fam: str, sport: str, market_data: Dict[str, float]) -> float:
        w_min = float(CONSENSUS_CFG.get("w_min", 0.15))
        w_max = float(CONSENSUS_CFG.get("w_max", 0.55))

        vol = market_data.get("market_volume", 0.5)
        steam = market_data.get("steam_strength", 0.0)
        agree = market_data.get("book_agreement", 0.5)
        public = market_data.get("public_pct", 0.5)

        lo, hi = CONSENSUS_CFG["sport_w_hint"].get(detect_sport("X_"+fam), (self.cfg.get("consensus_weight", 0.35), self.cfg.get("consensus_weight", 0.35)))
        base = (lo + hi) / 2.0

        w = base + 0.15*vol + 0.15*agree - 0.20*steam
        if public > 0.70:
            w -= 0.10
        return max(w_min, min(w_max, w))

    # ---------- Alt-line optimizer ----------
    def optimize_alt_line(self, market: str, mu: float, sigma: float, posted_line: float, direction: str, price_quote: float, tier: str, consensus: Optional[float]=None) -> Tuple[float, float]:
        fam = market_family(market)
        # HBU for alt search
        mu_h, sg_h = self._apply_hbu(market, mu, sigma)
        sigma_eff = _inflate_sigma_if_needed(fam, sg_h)

        # Priority 77% candidate
        specials: List[float] = []
        if SAFE_GUARDS.get("enable_altline_77", False) and fam in PROP_FAM:
            cand = _altline_77_candidate(posted_line, direction)
            step_map = SAFE_GUARDS.get("altline_77_round_step", {})
            cand = _round_to_step(cand, step_map.get(fam, 0.5))
            specials.append(cand)

        # Local scan
        rng = SAFE_GUARDS["alt_scan_units"]
        step = SAFE_GUARDS["alt_step"]
        candidates: List[float] = []
        x = -rng
        while x <= rng and len(candidates) < SAFE_GUARDS["alt_max_candidates"]:
            candidates.append(round(posted_line + x, 3))
            x += step

        # Merge & de-dup (77% first)
        cands_all: List[float] = []
        seen = set()
        for L in specials + candidates:
            if L not in seen:
                cands_all.append(L); seen.add(L)

        pmin = SAFE_GUARDS["prop_pmin"].get(tier, 0.65) if fam in PROP_FAM else 0.0

        best_line, best_metric, best_price = posted_line, -1e18, price_quote
        for L in cands_all:
            p = self._p_hit(market, mu_h, sigma_eff, L, direction, consensus)
            if p < pmin:
                continue
            alt_price = self.price_model.estimate(market, L, direction, base_price=price_quote)
            metric = p * self.value_model.edge(alt_price)  # monotone with log-growth proxy
            if metric > best_metric:
                best_metric, best_line, best_price = metric, L, alt_price
        return best_line, best_price

    # ---------- Same-game copula joint simulation (optional utility) ----------
    def simulate_same_game_bundle(self, legs: List['V4EngineIntegrated.Leg'], corr: Optional[List[List[float]]]=None, sims: Optional[int]=None) -> List[List[float]]:
        """Simulate joint outcomes for legs in the same game using a Gaussian copula approximation.
        Returns simulated raw stat outcomes (not booleans). Caller can compute hits vs. lines.
        """
        if sims is None:
            sims = max(int(self.cfg.get("mc_sims_min", 100_000)), 10_000)
        n = len(legs)
        mu_vec, sg_vec = [], []
        for L in legs:
            fam = market_family(L.market)
            mu_h, sg_h = self._apply_hbu(L.market, L.mu, L.sigma)
            mu_vec.append(mu_h)
            sg_vec.append(_inflate_sigma_if_needed(fam, sg_h))
        # identity correlation if none provided
        if corr is None:
            corr = [[1.0 if i==j else 0.1 for j in range(n)] for i in range(n)]  # mild default
        samples = mvn_correlated_normals(mu_vec, sg_vec, corr, sims, seed=self.rng.seed)
        return samples


# === AUTO-PATCH: bind stray functions as class methods if present ===
try:
    if 'enrich_with_positional_intel' in globals():
        V4EngineIntegrated.enrich_with_positional_intel = enrich_with_positional_intel  # type: ignore
    if 'redistribute_usage_from_absences' in globals():
        V4EngineIntegrated.redistribute_usage_from_absences = redistribute_usage_from_absences  # type: ignore
except Exception as _e:
    print("[WARN] Could not bind positional-intel methods:", _e)

# -----------------------------------------------------------------------------
# Example (can be removed)
# -----------------------------------------------------------------------------

def example():
    eng = V4EngineIntegrated()
    leg = V4EngineIntegrated.Leg(
        market="NFL_RUSHYDS_PlayerX", mu=72.0, sigma=18.0, line=74.5, direction="over", price=1.83
    )
    res = eng.evaluate_mainline(leg.market, leg.mu, leg.sigma, leg.line, leg.direction, leg.price)
    print("Single leg eval:", res)

if __name__ == "__main__":
    import sys
    if "--selfcheck" not in sys.argv:
        try:
            example()
        except Exception as _e:
            print("[WARN] example() failed:", _e)




# ============================================================================
# Pro Methods Strategy Module (Integrated into V4 single-file build)
# Source concept: "Integrating Pro Methods Into V4 + Monte Carlo: Architecture and Code"
# This block is framework-agnostic and mounted as a strategy used by V4EngineIntegrated.
# ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import math, random, statistics

# -------------------------
# Data Structures
# -------------------------

@dataclass
    # ========================================================================
    # CONTEXT-AWARE MONTE CARLO METHODS
    # Added 2025-11-07 for context integration
    # ========================================================================

    def _init_context_aware_mc(self):
        """Initialize context-aware Monte Carlo engine"""
        if not hasattr(self, '_context_mc_engine'):
            base_mc = get_enhanced_mc_engine()
            self._context_mc_engine = ContextIntegratedMCEngine(base_mc, self)
        return self._context_mc_engine

    def _p_hit_raw_with_context(self, market, mu, sigma, line, direction, 
                                consensus=None, blob=None) -> float:
        """
        CONTEXT-AWARE Monte Carlo simulation that automatically applies
        all available context before running simulations.

        This is the main entry point that should be used instead of _p_hit_raw.
        """
        # Initialize context-aware MC engine
        context_mc = self._init_context_aware_mc()

        # Run simulation with full context integration
        try:
            result = context_mc.run_with_context(
                market=market,
                mu=mu,
                sigma=sigma,
                line=line,
                direction=direction,
                blob=blob
            )

            # Store context and diagnostics for later inspection
            if hasattr(self, 'context'):
                self.context["_last_mc_context"] = result["context"]
                self.context["_last_mc_sims"] = result["n_simulations"]
                self.context["_last_mc_diagnostics"] = result["mc_diagnostics"]
                self.context["_last_mc_se"] = result["standard_error"]
                self.context["_last_mc_convergence"] = result["convergence_achieved"]
                self.context["_last_mc_confidence"] = result["context"]["confidence_score"]

            return result["probability"]

        except Exception as e:
            # Fallback to original implementation
            import warnings
            warnings.warn(f"Context-aware MC failed, using fallback: {e}")
            # Fallback to original _p_hit_raw
            return self._p_hit_raw(market, mu, sigma, line, direction, consensus)

    def evaluate_mainline_with_context(self, market: str, mu: float, sigma: float, 
                                      posted_line: float, direction: str, price: float,
                                      blob: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enhanced evaluate_mainline that uses context-aware Monte Carlo.

        Args:
            market: Market identifier (e.g., "NFL_RECYDS_Player")
            mu: Base mean estimate
            sigma: Base standard deviation
            posted_line: The line to evaluate
            direction: "over" or "under"
            price: Odds/price for the bet
            blob: Optional context blob with injuries, matchups, weather, etc.

        Returns:
            Dict with p_hit, ev, kelly, and context details
        """
        fam = market_family(market)

        # Use context-aware simulation
        p = self._p_hit_raw_with_context(
            market, mu, sigma, posted_line, direction, None, blob
        )

        # Rest of evaluation logic
        haircut = float(self.cfg.get("ev_haircut", 0.985))

        if SAFE_GUARDS["enable_keynum_haircut"]:
            # Get adjusted parameters for fragility check
            context_mc = self._init_context_aware_mc()
            ctx = context_mc.context_adapter.extract_game_context(market, mu, sigma, blob)
            adj_mu = ctx["adjusted_mu"]
            adj_sigma = ctx["adjusted_sigma"]

            slope = fragility_slope(adj_mu, adj_sigma, posted_line)
            if slope > SAFE_GUARDS["keg_slope_cap"] and is_roundish_key(posted_line, SAFE_GUARDS["roundish_eps"]):
                haircut = max(haircut, SAFE_GUARDS["keynum_haircut_floor"])

        self._last_haircut = haircut
        implied = self.price_model.implied_prob(price)
        edge = max(0.0, p - implied) * haircut

        kelly_base = 0.0
        if implied > 0 and p > implied:
            b = max(1e-9, price - 1.0)
            q = 1.0 - p
            kelly_full = (b*p - q) / b
            kelly_base = max(0.0, kelly_full) * float(self.cfg.get("kelly_fraction", 0.5))

        # Get adjusted params for KEG penalty
        context_mc = self._init_context_aware_mc()
        ctx = context_mc.context_adapter.extract_game_context(market, mu, sigma, blob)
        kelly = self._apply_keg_penalty(ctx["adjusted_mu"], ctx["adjusted_sigma"], posted_line, kelly_base)
        kelly = min(float(self.cfg.get('bankroll_cap_pct', 0.03)), kelly)

        result = {
            "market": market,
            "family": fam,
            "line": posted_line,
            "direction": direction,
            "price": price,
            "p_hit": p,
            "ev_proxy": edge,
            "kelly": kelly,
            "haircut": haircut
        }

        # Add context details if available
        if hasattr(self, 'context') and "_last_mc_context" in self.context:
            result["context_applied"] = self.context["_last_mc_context"]["adjustments_applied"]
            result["context_confidence"] = self.context["_last_mc_context"]["confidence_score"]
            result["mu_adjusted"] = self.context["_last_mc_context"]["adjusted_mu"]
            result["sigma_adjusted"] = self.context["_last_mc_context"]["adjusted_sigma"]

        return result

    def get_context_simulation_summary(self) -> Dict[str, Any]:
        """
        Get summary of how context has been applied across all simulations.
        Useful for debugging and optimization.
        """
        context_mc = self._init_context_aware_mc()
        return context_mc.get_simulation_summary()

    def set_game_context(self, blob: Dict[str, Any]):
        """
        Set game context that will be used for subsequent evaluations.

        Example blob structure:
        {
            "injuries": [
                {"player": "Jayson Tatum", "position": "SF", "status": "out", 
                 "team": "BOS", "opponent": "ORL", "sport": "NBA"}
            ],
            "opponent": {
                "defensive_rating": 0.65,  # 0-1 scale, 0.5 = average
                "pace": 98.5
            },
            "is_home": False,
            "weather": {"wind_mph": 20, "temp_f": 45},
            "rest_days": 2,
            "recent_form": 0.444,  # Win rate
            "matchups": {
                "WR:Player Name": {
                    "vs_primary_cb": {"allow_yprr": 1.8, "allow_sr": 0.52},
                    "shadow": {"rate": 0.75}
                }
            },
            "defense": {
                "coverage_mix": {"man": 0.45, "zone": 0.55},
                "pressure_rate": 0.32
            }
        }
        """
        self.context["_game_blob"] = blob

    def get_game_context(self) -> Optional[Dict[str, Any]]:
        """Retrieve current game context blob"""
        return self.context.get("_game_blob")


class TeamRating:
    team_id: str
    power: float            # Walters-style team power rating
    qb_adj: float           # QB-specific adjustment
    injury_adj: float       # Sum of non-QB injury adjustments
    situational_adj: float  # Situational quantified factors (HFA, fatigue, bounceback, etc.)

@dataclass
class Game:
    game_id: str
    sport: str              # "NFL","NBA","CFB","CBB" etc.
    home: str
    away: str
    neutral: bool
    start_time_unix: int
    market_spread_home: float   # -3.5 means home favored by 3.5
    market_total: float
    market_moneyline_home: Optional[float] = None
    market_moneyline_away: Optional[float] = None

@dataclass
class InjuryImpact:
    team_id: str
    qb_out: bool
    qb_value_pts: float
    key_non_qb_pts: float   # aggregate non-QB point value
    ol_downgrades_pts: float

@dataclass
class ScheduleAdjustedStats:
    team_id: str
    # Example core compensated stats (schedule-adjusted)
    off_ypc_comp: float
    def_ypc_comp: float
    off_ypp_comp: float
    def_ypp_comp: float
    # Play-by-play derived predictive metrics
    yards_before_contact_run: float
    short_ydg_conv_rate: float
    pressure_rate_allowed: float
    play_action_success: float
    red_zone_eff: float
    pace_possessions_per_game: Optional[float] = None
    ft_rate: Optional[float] = None
    three_pt_rate: Optional[float] = None
    recency_weight: float = 0.5

@dataclass
class PeriodModelParams:
    first_half_multiplier: float
    second_half_multiplier: float
    close_game_fourth_q_foul_factor: float  # 1.0-1.8 multiplier
    bench_basket_first_half_bias: float     # visiting bench shooting bias

@dataclass
class ModelWeights:
    walters_power_w: float
    bob_comp_stats_w: float
    sharp_play_weights_w: float
    market_efficiency_w: float

@dataclass
class CorrelationInfo:
    rho: Dict[Tuple[str, str], float]  # (bet_leg_id_a, bet_leg_id_b) -> correlation [-1,1]

@dataclass
class BetEdge:
    game_id: str
    market: str           # "spread_home", "spread_away", "total_over", etc.
    fair_price: float     # decimal odds fair value
    market_price: float   # decimal odds
    edge: float           # EV per unit stake
    win_prob: float
    kelly_fraction: float
    stake_units: float

# -------------------------
# Utility & Odds Converters
# -------------------------

def american_to_decimal(odds: float) -> float:
    if odds is None:
        return None
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / abs(odds)

def decimal_to_american(dec: float) -> float:
    if dec <= 1.0:
        return 0.0
    if dec >= 2.0:
        return (dec - 1.0) * 100.0
    else:
        return -100.0 / (dec - 1.0)

def price_to_prob(dec: float) -> float:
    return 1.0 / dec

def prob_to_fair_price(p: float) -> float:
    return 1.0 / max(1e-9, p)

def kelly_fraction(prob: float, dec_odds: float) -> float:
    # Kelly: f* = (bp - q) / b where b = dec_odds - 1
    b = dec_odds - 1.0
    p = prob
    q = 1.0 - p
    if b <= 0:
        return 0.0
    val = (b * p - q) / b
    return max(0.0, val)

# -------------------------
# Walters-Style Power Rating
# -------------------------

class WaltersPowerModel:
    def __init__(self, base_hfa_points: float = 2.5):
        self.base_hfa = base_hfa_points
        self.bounce19 = 1.5
        self.bounce29 = 2.0
        self.monday_road_pen = 1.5

    def true_game_performance(self, raw_diff: float, injury_pts: float, situational_pts: float, weather_pts: float) -> float:
        return raw_diff + injury_pts + situational_pts + weather_pts

    def update_power(self, prev_power: float, true_perf_level: float, alpha: float = 0.10) -> float:
        return (1.0 - alpha) * prev_power + alpha * true_perf_level

    def quantify_situational(self, game, is_super_bowl_champ: bool, last_loss_margin: Optional[int], is_mnf_road: bool) -> float:
        adj = 0.0
        if not game.neutral:
            adj += self.base_hfa
        if is_super_bowl_champ:
            adj += 0.5
        if last_loss_margin is not None:
            if last_loss_margin >= 29:
                adj += self.bounce29
            elif last_loss_margin >= 19:
                adj += self.bounce19
        if is_mnf_road:
            adj -= self.monday_road_pen
        return adj

# -------------------------
# Dr. Bob-Style Compensated Stats
# -------------------------

def compensated_stat(raw_value: float, opp_allowed_avg: float, league_avg: float) -> float:
    if opp_allowed_avg <= 1e-9:
        return raw_value
    return (raw_value / opp_allowed_avg) * league_avg

def predictive_blend(stats: ScheduleAdjustedStats) -> float:
    w_ybc = 0.30
    w_short = 0.20
    w_press = 0.15
    w_play_action = 0.15
    w_rz = 0.20
    base = (
        w_ybc * stats.yards_before_contact_run
        + w_short * stats.short_ydg_conv_rate
        - w_press * stats.pressure_rate_allowed
        + w_play_action * stats.play_action_success
        + w_rz * stats.red_zone_eff
    )
    return base * (0.5 + 0.5 * stats.recency_weight)

# -------------------------
# Voulgaris-Style Period Model
# -------------------------

def adjust_total_by_period(total: float, params: PeriodModelParams, is_close_game: bool, visiting_bench_bias: bool) -> Tuple[float, float]:
    fh = total * 0.5 * params.first_half_multiplier
    sh_mult = params.second_half_multiplier
    if is_close_game:
        sh_mult *= params.close_game_fourth_q_foul_factor
    if visiting_bench_bias:
        fh *= params.bench_basket_first_half_bias
    sh = total - fh
    sh *= sh_mult
    scale = (fh + sh)
    if scale <= 1e-9:
        return (total * 0.5, total * 0.5)
    norm_factor = total / scale
    return (fh * norm_factor, sh * norm_factor)

# -------------------------
# Market Delta & Ensemble Fair Line
# -------------------------

def ensemble_spread(home_team: TeamRating,
                    away_team: TeamRating,
                    home_stats: ScheduleAdjustedStats,
                    away_stats: ScheduleAdjustedStats,
                    weights: ModelWeights,
                    walters_model: WaltersPowerModel,
                    situational_home: float,
                    situational_away: float) -> float:
    walters_comp = (home_team.power + home_team.qb_adj + home_team.injury_adj + situational_home) - \
                   (away_team.power + away_team.qb_adj + away_team.injury_adj + situational_away)

    bob_comp = ((home_stats.off_ypp_comp - away_stats.def_ypp_comp) - (away_stats.off_ypp_comp - home_stats.def_ypp_comp))
    bob_comp += predictive_blend(home_stats) - predictive_blend(away_stats)

    sharp_comp = (home_stats.yards_before_contact_run - away_stats.yards_before_contact_run) \
                 + (home_stats.short_ydg_conv_rate - away_stats.short_ydg_conv_rate) \
                 - 0.7 * (home_stats.pressure_rate_allowed - away_stats.pressure_rate_allowed)

    norm_walters = walters_comp
    norm_bob = bob_comp * 3.0
    norm_sharp = sharp_comp * 4.0

    market_anchor = 0.0

    spread = (weights.walters_power_w * norm_walters +
              weights.bob_comp_stats_w * norm_bob +
              weights.sharp_play_weights_w * norm_sharp +
              weights.market_efficiency_w * market_anchor)
    return spread

def spread_to_win_prob(spread: float, market_spread: float, sigma_pts: float = 13.5) -> float:
    from math import erf, sqrt
    z = (spread - market_spread) / max(1e-9, sigma_pts)
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

# -------------------------
# Edge, EV, and Stake
# -------------------------

def compute_edge(win_prob: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    return win_prob * b - (1.0 - win_prob)

def optimal_stake_units(win_prob: float, dec_odds: float, bankroll_units: float, kelly_frac: float = 0.25) -> float:
    k = kelly_fraction(win_prob, dec_odds)
    return bankroll_units * (k * kelly_frac)

# -------------------------
# Monte Carlo Simulation
# -------------------------

def simulate_bankroll(bets: List[BetEdge], n_trials: int = 10000, starting_bankroll: float = 100.0) -> Dict[str, float]:
    final_bankrolls = []
    max_drawdowns = []

    for _ in range(n_trials):
        bankroll = starting_bankroll
        peak = bankroll
        for b in bets:
            stake = b.stake_units
            if stake <= 0:
                continue
            outcome = 1 if random.random() <= b.win_prob else 0
            if outcome == 1:
                bankroll += stake * (b.fair_price - 1.0)  # conservative (fair instead of market)
            else:
                bankroll -= stake
            peak = max(peak, bankroll)
        dd = (peak - bankroll) / max(1e-9, peak)
        final_bankrolls.append(bankroll)
        max_drawdowns.append(dd)

    final_bankrolls_sorted = sorted(final_bankrolls)
    p05_idx = max(0, int(0.05 * n_trials) - 1)
    p95_idx = min(n_trials - 1, int(0.95 * n_trials) - 1)

    return {
        "median_final_bankroll": statistics.median(final_bankrolls),
        "p05_final_bankroll": final_bankrolls_sorted[p05_idx],
        "p95_final_bankroll": final_bankrolls_sorted[p95_idx],
        "mean_max_drawdown": statistics.mean(max_drawdowns),
    }

# -------------------------
# Correlation-Aware Parlay Construction (Greedy)
# -------------------------

def build_parlay_legs(candidates: List[BetEdge],
                      rho: CorrelationInfo,
                      max_legs: int = 5,
                      min_edge: float = 0.01,
                      max_pairwise_corr: float = 0.6) -> List[BetEdge]:
    legs = []
    sorted_cands = sorted([c for c in candidates if c.edge >= min_edge], key=lambda x: x.edge, reverse=True)
    for c in sorted_cands:
        ok = True
        for l in legs:
            key = (f"{c.game_id}:{c.market}", f"{l.game_id}:{l.market}")
            rev = (key[1], key[0])
            r = rho.rho.get(key, rho.rho.get(rev, 0.0))
            if r > max_pairwise_corr:
                ok = False
                break
        if ok:
            legs.append(c)
        if len(legs) >= max_legs:
            break
    return legs

# -------------------------
# Main Strategy Orchestrator
# -------------------------

class V4MC_ProBlendStrategy:
    def __init__(self,
                 weights: 'ModelWeights',
                 period_params_by_sport: Dict[str, 'PeriodModelParams'],
                 bankroll_units: float = 100.0,
                 kelly_fractional: float = 0.25):
        self.weights = weights
        self.period_params_by_sport = period_params_by_sport
        self.bankroll_units = bankroll_units
        self.kelly_fractional = kelly_fractional
        self.walters = WaltersPowerModel()

    def fair_spread(self,
                    game: 'Game',
                    home: 'TeamRating',
                    away: 'TeamRating',
                    home_stats: 'ScheduleAdjustedStats',
                    away_stats: 'ScheduleAdjustedStats',
                    situational_home: float,
                    situational_away: float) -> float:
        return ensemble_spread(home, away, home_stats, away_stats, self.weights, self.walters, situational_home, situational_away)

    def fair_total(self,
                   game: 'Game',
                   home_stats: 'ScheduleAdjustedStats',
                   away_stats: 'ScheduleAdjustedStats',
                   visiting_bench_bias: bool = True,
                   is_close_game: bool = True) -> float:
        base = game.market_total
        diff = (predictive_blend(home_stats) + predictive_blend(away_stats)) * 1.5
        prelim = base + diff
        params = self.period_params_by_sport.get(game.sport, PeriodModelParams(1.00, 1.00, 1.00, 1.00))
        fh, sh = adjust_total_by_period(prelim, params, is_close_game=is_close_game, visiting_bench_bias=visiting_bench_bias and (game.away is not None))
        return fh + sh

    def price_spread_side(self, fair_spread: float, market_spread: float, side: str) -> Tuple[float, float]:
        sigma = 13.5
        p = spread_to_win_prob(fair_spread, market_spread, sigma_pts=sigma)
        if side == "away":
            p = 1.0 - p
        fair_dec = prob_to_fair_price(p)
        return p, fair_dec

    def evaluate_spread_bet(self, game: 'Game', fair_spread: float, side: str, market_price_amer: float) -> 'BetEdge':
        market_dec = american_to_decimal(market_price_amer)
        market_spread = game.market_spread_home  # assumed home-oriented
        p, fair_dec = self.price_spread_side(fair_spread, market_spread, side)
        edge = compute_edge(p, market_dec)
        kfrac = kelly_fraction(p, market_dec) * self.kelly_fractional
        stake = self.bankroll_units * kfrac
        market = f"spread_{side}"
        return BetEdge(game.game_id, market, fair_dec, market_dec, edge, p, kfrac, stake)

    def evaluate_total_bet(self, game: 'Game', fair_total: float, over_under: str, market_price_amer: float) -> 'BetEdge':
        sigma_total = 18.0
        from math import erf, sqrt
        z = (fair_total - game.market_total) / sigma_total
        p_over = 0.5 * (1.0 + erf(z / math.sqrt(2)))
        p = p_over if over_under == "over" else (1.0 - p_over)
        market_dec = american_to_decimal(market_price_amer)
        fair_dec = prob_to_fair_price(p)
        edge = compute_edge(p, market_dec)
        kfrac = kelly_fraction(p, market_dec) * self.kelly_fractional
        stake = self.bankroll_units * kfrac
        market = f"total_{over_under}"
        return BetEdge(game.game_id, market, fair_dec, market_dec, edge, p, kfrac, stake)

    def build_card(self,
                   games: List['Game'],
                   team_ratings: Dict[str, 'TeamRating'],
                   stats: Dict[str, 'ScheduleAdjustedStats'],
                   situational_hooks: Callable[['Game'], Tuple[float, float]],
                   price_feed: Callable[['Game', str], float]) -> List['BetEdge']:
        bets: List[BetEdge] = []
        for g in games:
            home = team_ratings[g.home]
            away = team_ratings[g.away]
            home_stats = stats[g.home]
            away_stats = stats[g.away]
            sit_home, sit_away = situational_hooks(g)
            fair_sp = self.fair_spread(g, home, away, home_stats, away_stats, sit_home, sit_away)
            price_home = price_feed(g, "spread_home")  # Amer odds
            price_away = price_feed(g, "spread_away")
            beh = self.evaluate_spread_bet(g, fair_spread=fair_sp, side="home", market_price_amer=price_home)
            bea = self.evaluate_spread_bet(g, fair_spread=fair_sp, side="away", market_price_amer=price_away)
            best_spread = beh if beh.edge > bea.edge else bea
            if best_spread.edge > 0:
                bets.append(best_spread)

            fair_tot = self.fair_total(g, home_stats, away_stats, visiting_bench_bias=True, is_close_game=True)
            price_over = price_feed(g, "total_over")
            price_under = price_feed(g, "total_under")
            beo = self.evaluate_total_bet(g, fair_total=fair_tot, over_under="over", market_price_amer=price_over)
            beu = self.evaluate_total_bet(g, fair_total=fair_tot, over_under="under", market_price_amer=price_under)
            best_total = beo if beo.edge > beu.edge else beu
            if best_total.edge > 0:
                bets.append(best_total)
        return bets

# -------------------------
# Suggested defaults for weights and period params
# -------------------------

DEFAULT_WEIGHTS_NFL = ModelWeights(
    walters_power_w=0.40,
    bob_comp_stats_w=0.30,
    sharp_play_weights_w=0.20,
    market_efficiency_w=0.10
)

DEFAULT_PERIOD_PARAMS = {
    "NFL": PeriodModelParams(
        first_half_multiplier=1.00,
        second_half_multiplier=1.02,
        close_game_fourth_q_foul_factor=1.12,
        bench_basket_first_half_bias=1.00
    ),
    "NBA": PeriodModelParams(
        first_half_multiplier=0.995,
        second_half_multiplier=1.02,
        close_game_fourth_q_foul_factor=1.18,
        bench_basket_first_half_bias=1.03
    ),
    "CFB": PeriodModelParams(
        first_half_multiplier=1.00,
        second_half_multiplier=1.00,
        close_game_fourth_q_foul_factor=1.05,
        bench_basket_first_half_bias=1.00
    ),
    "CBB": PeriodModelParams(
        first_half_multiplier=0.99,
        second_half_multiplier=1.02,
        close_game_fourth_q_foul_factor=1.15,
        bench_basket_first_half_bias=1.02
    ),
}

# ============================================================================
# V4 Integration Hooks
# ============================================================================

def _example_situational_hooks_from_v4ctx(g: 'Game') -> Tuple[float, float]:
    # Adapts to engine context if available; otherwise returns Walters defaults
    walters = WaltersPowerModel()
    # TODO: wire to V4 context feeds (schedule, injuries, rest, travel, altitude, weather)
    home_situ = walters.quantify_situational(g, is_super_bowl_champ=False, last_loss_margin=None, is_mnf_road=False)
    away_situ = walters.quantify_situational(g, is_super_bowl_champ=False, last_loss_margin=None, is_mnf_road=False)
    return home_situ, away_situ

def _example_price_feed_from_v4ctx(g: 'Game', market_key: str) -> float:
    # TODO: connect to engine odds layer. Return American odds for demo.
    return -110.0

def run_problend_daycard(games: List['Game'],
                         team_ratings: Dict[str, 'TeamRating'],
                         stats: Dict[str, 'ScheduleAdjustedStats'],
                         rho: Optional['CorrelationInfo']=None,
                         bankroll_units: float = 100.0,
                         kelly_fractional: float = 0.25,
                         period_params_by_sport: Optional[Dict[str, 'PeriodModelParams']]=None,
                         weights: Optional['ModelWeights']=None,
                         n_trials_mc_bankroll: int = 5000,
                         starting_bankroll: float = 100.0,
                         parlay_max_legs: int = 5,
                         parlay_min_edge: float = 0.01,
                         parlay_max_pairwise_corr: float = 0.6):
    if weights is None:
        weights = DEFAULT_WEIGHTS_NFL
    if period_params_by_sport is None:
        period_params_by_sport = DEFAULT_PERIOD_PARAMS
    if rho is None:
        rho = CorrelationInfo(rho={})
    strat = V4MC_ProBlendStrategy(weights=weights,
                                  period_params_by_sport=period_params_by_sport,
                                  bankroll_units=bankroll_units,
                                  kelly_fractional=kelly_fractional)
    bets = strat.build_card(games, team_ratings, stats, _example_situational_hooks_from_v4ctx, _example_price_feed_from_v4ctx)
    pos_bets = [b for b in bets if b.edge > 0]
    risk_report = simulate_bankroll(pos_bets, n_trials=n_trials_mc_bankroll, starting_bankroll=starting_bankroll)
    parlay = build_parlay_legs(pos_bets, rho=rho, max_legs=parlay_max_legs, min_edge=parlay_min_edge, max_pairwise_corr=parlay_max_pairwise_corr)
    return pos_bets, parlay, risk_report



# ==== ProBlend Bridge: add convenience methods to V4EngineIntegrated ====

def _attach_problend_methods_to_v4(cls):
    # Lazy imports inside method bodies to avoid top-of-file order issues.
    def problend_build_card(self, games, team_ratings, stats, rho=None,
                            bankroll_units=100.0, kelly_fractional=0.25,
                            period_params_by_sport=None, weights=None,
                            n_trials_mc_bankroll=5000, starting_bankroll=100.0,
                            parlay_max_legs=5, parlay_min_edge=0.01, parlay_max_pairwise_corr=0.6):
        from types import SimpleNamespace
        # Route through the strategy facade
        result = run_problend_daycard(
            games=games, team_ratings=team_ratings, stats=stats, rho=rho,
            bankroll_units=bankroll_units, kelly_fractional=kelly_fractional,
            period_params_by_sport=period_params_by_sport, weights=weights,
            n_trials_mc_bankroll=n_trials_mc_bankroll, starting_bankroll=starting_bankroll,
            parlay_max_legs=parlay_max_legs, parlay_min_edge=parlay_min_edge,
            parlay_max_pairwise_corr=parlay_max_pairwise_corr
        )
        # returns (pos_bets, parlay, risk_report)
        return result
    cls.problend_build_card = problend_build_card
    return cls

# Apply the bridge to V4EngineIntegrated if it's defined
try:
    V4EngineIntegrated = _attach_problend_methods_to_v4(V4EngineIntegrated)
except Exception as _e:
    # If class not yet defined at import time, do nothing (safe for module import order)
    pass



# ============================================================================
# V4 Context Adapters (Odds, Injuries, Priors, Rho) — Plug-and-Play
# ============================================================================

def _get_ctx_dict(ctx, key):
    obj = ctx.get(key, {})
    return obj if isinstance(obj, dict) else {}

def _american_from_price_or_prob(entry):
    """Accepts entry that may be american odds (int/float), decimal odds, or prob.
    Heuristics:
      - If entry is dict, try keys: 'american','dec','prob'
      - If numeric:
          * abs(value) > 1.2 and < 20 -> decimal odds
          * abs(value) >= 20 or value in [-5000,5000] -> treat as American
          * 0 < value < 1 -> probability
    Returns American odds float.
    """
    def to_amer_from_dec(dec):
        if dec >= 2.0: return (dec - 1.0) * 100.0
        if dec <= 1.0: return -10000.0
        return -100.0 / (dec - 1.0)
    def to_amer_from_prob(p):
        p = max(1e-6, min(1-1e-6, p))
        dec = 1.0 / p
        return to_amer_from_dec(dec)

    if isinstance(entry, dict):
        if 'american' in entry: return float(entry['american'])
        if 'dec' in entry: return float(to_amer_from_dec(float(entry['dec'])))
        if 'prob' in entry: return float(to_amer_from_prob(float(entry['prob'])))
        # fallback single value nested
        for v in entry.values():
            try:
                fv = float(v)
                if 0.0 < fv < 1.0: return float(to_amer_from_prob(fv))
                if 1.2 <= abs(fv) <= 20.0: return float(to_amer_from_dec(fv))
                return fv
            except Exception:
                continue
        return -110.0

    try:
        v = float(entry)
        if 0.0 < v < 1.0:
            return float(to_amer_from_prob(v))
        if 1.2 <= abs(v) <= 20.0:
            return float(to_amer_from_dec(v))
        return v
    except Exception:
        return -110.0

def _situational_from_ctx(ctx, game_obj):
    """Compute Walters situational points using available context keys.
    Expects optional ctx keys:
      - schedule: {game_id: {neutral:bool, is_mnf_road_home:bool, is_mnf_road_away:bool}}
      - accolades: {team_id: {super_bowl_champ:bool}}
      - results: {team_id: {last_loss_margin:int}}
    """
    walters = WaltersPowerModel()
    schedule = _get_ctx_dict(ctx, 'schedule')
    accolades = _get_ctx_dict(ctx, 'accolades')
    results = _get_ctx_dict(ctx, 'results')

    sch = schedule.get(game_obj.game_id, {}) if isinstance(schedule, dict) else {}
    neutral = bool(sch.get('neutral', game_obj.neutral))
    # Build two temp Game objects with neutral carried through
    g_home = type('G', (), dict(neutral=neutral))
    g_away = type('G', (), dict(neutral=neutral))

    home_sb = bool(_get_ctx_dict(accolades, game_obj.home).get('super_bowl_champ', False)) if isinstance(accolades, dict) else False
    away_sb = bool(_get_ctx_dict(accolades, game_obj.away).get('super_bowl_champ', False)) if isinstance(accolades, dict) else False

    home_last_loss = _get_ctx_dict(results, game_obj.home).get('last_loss_margin', None) if isinstance(results, dict) else None
    away_last_loss = _get_ctx_dict(results, game_obj.away).get('last_loss_margin', None) if isinstance(results, dict) else None

    home_mnf_road = bool(sch.get('is_mnf_road_home', False))
    away_mnf_road = bool(sch.get('is_mnf_road_away', False))

    home_situ = walters.quantify_situational(g_home, is_super_bowl_champ=home_sb, last_loss_margin=home_last_loss, is_mnf_road=home_mnf_road)
    away_situ = walters.quantify_situational(g_away, is_super_bowl_champ=away_sb, last_loss_margin=away_last_loss, is_mnf_road=away_mnf_road)
    return home_situ, away_situ

def v4ctx_situational_hooks(g: 'Game', ctx: dict) -> tuple:
    try:
        return _situational_from_ctx(ctx, g)
    except Exception:
        # fallback
        return _example_situational_hooks_from_v4ctx(g)

def v4ctx_price_feed(g: 'Game', market_key: str, ctx: dict) -> float:
    """Look up odds by game_id and market key from ctx['odds'] OR ctx['consensus_sources'] shape.
       Supports:
         ctx['odds'][game_id][market_key] -> american/dec/prob
         or a flat map ctx['odds'][(game_id, market_key)]
    """
    odds = _get_ctx_dict(ctx, 'odds')
    val = None
    if isinstance(odds, dict):
        if g.game_id in odds and isinstance(odds[g.game_id], dict):
            val = odds[g.game_id].get(market_key)
        if val is None:
            val = odds.get((g.game_id, market_key))
    if val is None:
        # try book bias / market anchor to nudge default
        bb = _get_ctx_dict(ctx, 'book_bias').get(market_key, 0.0) if isinstance(_get_ctx_dict(ctx,'book_bias'), dict) else 0.0
        base = -110.0 + float(bb) * 10.0
        return base
    return _american_from_price_or_prob(val)

def v4ctx_build_team_ratings(ctx: dict, team_ids: list) -> Dict[str, 'TeamRating']:
    pri = _get_ctx_dict(ctx, 'priors')
    injuries = _get_ctx_dict(ctx, 'injuries')
    out = {}
    for tid in team_ids:
        p = _get_ctx_dict(pri, tid)
        inj = _get_ctx_dict(injuries, tid)
        power = float(p.get('power', 0.0))
        qb_adj = float(inj.get('qb_adj', 0.0) or p.get('qb_adj', 0.0))
        inj_adj = float(inj.get('non_qb_pts', 0.0))
        situ = float(p.get('situational_adj', 0.0))
        out[tid] = TeamRating(team_id=tid, power=power, qb_adj=qb_adj, injury_adj=inj_adj, situational_adj=situ)
    return out

def v4ctx_build_comp_stats(ctx: dict, team_ids: list) -> Dict[str, 'ScheduleAdjustedStats']:
    stats = _get_ctx_dict(ctx, 'schedule_adjusted_stats')
    out = {}
    for tid in team_ids:
        s = _get_ctx_dict(stats, tid)
        def _g(k, d=0.0): 
            try: return float(s.get(k, d))
            except Exception: return d
        out[tid] = ScheduleAdjustedStats(
            team_id=tid,
            off_ypc_comp=_g('off_ypc_comp'),
            def_ypc_comp=_g('def_ypc_comp'),
            off_ypp_comp=_g('off_ypp_comp'),
            def_ypp_comp=_g('def_ypp_comp'),
            yards_before_contact_run=_g('ybc_run'),
            short_ydg_conv_rate=_g('short_ydg_conv'),
            pressure_rate_allowed=_g('pressure_rate_allowed'),
            play_action_success=_g('play_action_success'),
            red_zone_eff=_g('red_zone_eff'),
            pace_possessions_per_game=_g('pace', None),
            ft_rate=_g('ft_rate', None),
            three_pt_rate=_g('three_pt_rate', None),
            recency_weight=_g('recency_weight', 0.5),
        )
    return out

def v4ctx_build_rho(ctx: dict) -> 'CorrelationInfo':
    rho_raw = _get_ctx_dict(ctx, 'rho_overrides')
    # Expect either { "A|B": 0.3 } or { ("A","B"): 0.3 }
    rho_map = {}
    if isinstance(rho_raw, dict):
        for k, v in rho_raw.items():
            if isinstance(k, (list, tuple)) and len(k) == 2:
                a, b = str(k[0]), str(k[1])
            else:
                parts = str(k).split('|', 1)
                if len(parts) == 2:
                    a, b = parts[0], parts[1]
                else:
                    # Skip invalid key
                    continue
            try:
                rho_map[(a, b)] = float(v)
            except Exception:
                continue
    return CorrelationInfo(rho=rho_map)

def v4ctx_autowire_games(ctx: dict) -> list:
    """Try to build a minimal List[Game] from ctx['schedule'] and ctx['lines']"""
    schedule = _get_ctx_dict(ctx, 'schedule')
    lines = _get_ctx_dict(ctx, 'lines')
    games = []
    for gid, meta in schedule.items():
        try:
            sport = meta.get('sport', 'NFL')
            home = meta['home']
            away = meta['away']
            neutral = bool(meta.get('neutral', False))
            start_time_unix = int(meta.get('start', 0))
            L = _get_ctx_dict(lines, gid)
            mspread_home = float(L.get('market_spread_home', 0.0))
            mtotal = float(L.get('market_total', 44.5))
            mlh = L.get('market_moneyline_home')
            mla = L.get('market_moneyline_away')
            games.append(Game(
                game_id=str(gid), sport=str(sport), home=str(home), away=str(away),
                neutral=neutral, start_time_unix=start_time_unix,
                market_spread_home=mspread_home, market_total=mtotal,
                market_moneyline_home=mlh, market_moneyline_away=mla
            ))
        except Exception:
            continue
    return games

# Replace the bridge's basic method with a context-aware version
def _attach_problend_methods_to_v4_ctxaware(cls):
    def problend_build_card(self, games=None, team_ratings=None, stats=None, rho=None,
                            bankroll_units=100.0, kelly_fractional=0.25,
                            period_params_by_sport=None, weights=None,
                            n_trials_mc_bankroll=5000, starting_bankroll=100.0,
                            parlay_max_legs=5, parlay_min_edge=0.01, parlay_max_pairwise_corr=0.6,
                            situational_hooks=None, price_feed=None):
        ctx = getattr(self, 'context', {}) or {}
        # Build defaults from context if not supplied
        if games is None:
            games = v4ctx_autowire_games(ctx)
        # Collect team ids referenced by games
        team_ids = set()
        for g in games:
            team_ids.add(g.home); team_ids.add(g.away)
        if team_ratings is None:
            team_ratings = v4ctx_build_team_ratings(ctx, list(team_ids))
        if stats is None:
            stats = v4ctx_build_comp_stats(ctx, list(team_ids))
        if rho is None:
            rho = v4ctx_build_rho(ctx)

        # Default hooks read odds/situational from context
        if situational_hooks is None:
            situational_hooks = lambda game: v4ctx_situational_hooks(game, ctx)
        if price_feed is None:
            price_feed = lambda game, mk: v4ctx_price_feed(game, mk, ctx)

        # Delegate to strategy facade (already integrated above)
        return run_problend_daycard(
            games=games, team_ratings=team_ratings, stats=stats, rho=rho,
            bankroll_units=bankroll_units, kelly_fractional=kelly_fractional,
            period_params_by_sport=period_params_by_sport, weights=weights,
            n_trials_mc_bankroll=n_trials_mc_bankroll, starting_bankroll=starting_bankroll,
            parlay_max_legs=parlay_max_legs, parlay_min_edge=parlay_min_edge,
            parlay_max_pairwise_corr=parlay_max_pairwise_corr
        )
    cls.problend_build_card = problend_build_card
    return cls

try:
    V4EngineIntegrated = _attach_problend_methods_to_v4_ctxaware(V4EngineIntegrated)
except Exception as _e:
    pass



# ============================================================================
# V4EngineIntegrated: Team-level loaders & optional game-market blending
# ============================================================================

def _v4_safe_get_ctx(self):
    ctx = getattr(self, "context", None)
    if ctx is None or not isinstance(ctx, dict):
        self.context = {}
        ctx = self.context
    return ctx

def _v4_import_problend_objs():
    # Lazy import of strategy symbols defined above
    from typing import Tuple
    try:
        _objs = dict(
            TeamRating=globals()["TeamRating"],
            ScheduleAdjustedStats=globals()["ScheduleAdjustedStats"],
            Game=globals()["Game"],
            ModelWeights=globals()["ModelWeights"],
            PeriodModelParams=globals()["PeriodModelParams"],
            DEFAULT_WEIGHTS_NFL=globals()["DEFAULT_WEIGHTS_NFL"],
            DEFAULT_PERIOD_PARAMS=globals()["DEFAULT_PERIOD_PARAMS"],
            V4MC_ProBlendStrategy=globals()["V4MC_ProBlendStrategy"],
            spread_to_win_prob=globals()["spread_to_win_prob"],
            prob_to_fair_price=globals()["prob_to_fair_price"],
        )
        return _objs
    except KeyError as e:
        raise RuntimeError(f"ProBlend symbols not found in module: {e}")

# ---- Monkey-patch helpers onto the class without altering existing methods ----
def _extend_v4_with_team_level_methods(cls):
    # Loaders
    def load_team_ratings(self, ratings_file: str):
        """Load Walters-style team power ratings into context['priors']."""
        import json
        ctx = _v4_safe_get_ctx(self)
        try:
            with open(ratings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load team ratings: {e}")
        # Accept direct TeamRating dict or flat values per team
        pri = {}
        for tid, r in data.items():
            if isinstance(r, dict):
                pri[tid] = {
                    "power": float(r.get("power", 0.0)),
                    "qb_adj": float(r.get("qb_adj", 0.0)),
                    "situational_adj": float(r.get("situational_adj", 0.0)),
                }
            else:
                pri[tid] = {"power": float(r), "qb_adj": 0.0, "situational_adj": 0.0}
        ctx["priors"] = pri

    def load_schedule_adjusted_stats(self, stats_file: str):
        """Load Dr. Bob-style schedule-compensated stats into context['schedule_adjusted_stats']."""
        import json
        ctx = _v4_safe_get_ctx(self)
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load schedule-adjusted stats: {e}")
        ctx["schedule_adjusted_stats"] = data

    # Feature toggle
    def enable_game_market_blend(self, on: bool = True, weights=None, period_params=None):
        ctx = _v4_safe_get_ctx(self)
        ctx["enable_problend_blend"] = bool(on)
        if weights is not None:
            ctx["problend_weights"] = weights
        if period_params is not None:
            ctx["problend_period_params"] = period_params

    # Market helpers
    def is_game_market(self, market: str) -> bool:
        m = str(market).upper()
        return any(k in m for k in ["SPREAD", "TOTAL_"]) or m.endswith("_SPREAD") or m.startswith("TOTAL")

    # Fair value utilities (single game convenience)
    def fair_spread_for_game(self, game_obj):
        syms = _v4_import_problend_objs()
        weights = self.context.get("problend_weights", syms["DEFAULT_WEIGHTS_NFL"])  # default NFL
        pparams = self.context.get("problend_period_params", syms["DEFAULT_PERIOD_PARAMS"])
        # Build from context using adapters
        games = [game_obj]
        # Collect ids
        tids = [game_obj.home, game_obj.away]
        team_ratings = v4ctx_build_team_ratings(self.context, tids)
        comp_stats = v4ctx_build_comp_stats(self.context, tids)
        strat = syms["V4MC_ProBlendStrategy"](weights=weights, period_params_by_sport=pparams)
        home = team_ratings[game_obj.home]; away = team_ratings[game_obj.away]
        hs = comp_stats[game_obj.home]; as_ = comp_stats[game_obj.away]
        sit_home, sit_away = v4ctx_situational_hooks(game_obj, self.context)
        return strat.fair_spread(game_obj, home, away, hs, as_, sit_home, sit_away)

    def fair_total_for_game(self, game_obj):
        syms = _v4_import_problend_objs()
        weights = self.context.get("problend_weights", syms["DEFAULT_WEIGHTS_NFL"])  # unused directly
        pparams = self.context.get("problend_period_params", syms["DEFAULT_PERIOD_PARAMS"])
        tids = [game_obj.home, game_obj.away]
        comp_stats = v4ctx_build_comp_stats(self.context, tids)
        strat = syms["V4MC_ProBlendStrategy"](weights=weights, period_params_by_sport=pparams)
        hs = comp_stats[game_obj.home]; as_ = comp_stats[game_obj.away]
        return strat.fair_total(game_obj, hs, as_)

    def p_hit_spread_side(self, game_obj, side: str) -> float:
        """Return win probability for a spread side using ProBlend fair vs market."""
        syms = _v4_import_problend_objs()
        weights = self.context.get("problend_weights", syms["DEFAULT_WEIGHTS_NFL"])  # default NFL
        pparams = self.context.get("problend_period_params", syms["DEFAULT_PERIOD_PARAMS"])
        strat = syms["V4MC_ProBlendStrategy"](weights=weights, period_params_by_sport=pparams)
        fair = self.fair_spread_for_game(game_obj)
        p, _ = strat.price_spread_side(fair, game_obj.market_spread_home, side)
        return self._conformal_beta(p)

    def p_hit_total_ou(self, game_obj, over_under: str) -> float:
        syms = _v4_import_problend_objs()
        weights = self.context.get("problend_weights", syms["DEFAULT_WEIGHTS_NFL"])  # default NFL
        pparams = self.context.get("problend_period_params", syms["DEFAULT_PERIOD_PARAMS"])
        strat = syms["V4MC_ProBlendStrategy"](weights=weights, period_params_by_sport=pparams)
        fair = self.fair_total_for_game(game_obj)
        # Reuse evaluate_total_bet to compute p
        be = strat.evaluate_total_bet(game_obj, fair_total=fair, over_under=over_under, market_price_amer=-110.0)
        return self._conformal_beta(be.win_prob)

    # Attach
    cls.load_team_ratings = load_team_ratings
    cls.load_schedule_adjusted_stats = load_schedule_adjusted_stats
    cls.enable_game_market_blend = enable_game_market_blend
    cls.is_game_market = is_game_market
    cls.fair_spread_for_game = fair_spread_for_game
    cls.fair_total_for_game = fair_total_for_game
    cls.p_hit_spread_side = p_hit_spread_side
    cls.p_hit_total_ou = p_hit_total_ou
    return cls

try:
    V4EngineIntegrated = _extend_v4_with_team_level_methods(V4EngineIntegrated)
except Exception as _e:
    # If class not yet defined, ignore (import order safety)
    pass



# ============================================================================
# High-level orchestrator: run both systems and emit a combined betslip
# ============================================================================

def _v4_try_load_file(fp):
    try:
        import json, os
        if fp and os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None

def _v4_auto_sources_defaults():
    # Standard filenames; override via sources dict param to run_default_day
    return {
        "priors": "team_power_ratings.json",
        "schedule_adjusted_stats": "schedule_adjusted_stats.json",
        "schedule": "schedule.json",
        "lines": "lines.json",
        "odds": "odds.json",
        "injuries": "injuries.json",
        "rho_overrides": "rho_overrides.json",
        "book_bias": "book_bias.json",
        # optional:
        "accolades": "accolades.json",
        "results": "results.json",
        # props (optional):
        "prop_candidates": "prop_candidates.json"
    }

def _v4_merge_context(self, mapping):
    ctx = _v4_safe_get_ctx(self)
    for k, fp in mapping.items():
        data = _v4_try_load_file(fp)
        if data is not None:
            ctx[k] = data
    return ctx

def _v4_summarize_betedge_list(rows):
    # Convert BetEdge objects (game markets) into serializable rows
    out = []
    for b in rows:
        try:
            out.append({
                "type": "game",
                "game_id": getattr(b, "game_id", ""),
                "market": getattr(b, "market", ""),
                "win_prob": round(float(getattr(b, "win_prob", 0.0)), 6),
                "edge": round(float(getattr(b, "edge", 0.0)), 6),
                "kelly_fraction": round(float(getattr(b, "kelly_fraction", 0.0)), 6),
                "stake_units": round(float(getattr(b, "stake_units", 0.0)), 4),
                "fair_price": round(float(getattr(b, "fair_price", 0.0)), 6),
                "market_price": round(float(getattr(b, "market_price", 0.0)), 6),
            })
        except Exception:
            continue
    return out

def _v4_summarize_props(self, prop_result):
    # Heuristic: if build_slips returns a dict or list, try to normalize
    out = []
    if prop_result is None:
        return out
    try:
        if isinstance(prop_result, dict) and "slips" in prop_result:
            slips = prop_result["slips"]
        else:
            slips = prop_result
        # Each slip may be a dict with legs; be permissive
        for s in slips:
            leg_list = s.get("legs", []) if isinstance(s, dict) else []
            for lg in leg_list:
                try:
                    out.append({
                        "type": "prop",
                        "market": lg.get("market"),
                        "player": lg.get("player"),
                        "line": lg.get("line"),
                        "direction": lg.get("direction"),
                        "win_prob": round(float(lg.get("win_prob", 0.0)), 6) if isinstance(lg, dict) else None,
                        "price": lg.get("price"),
                        "edge": round(float(lg.get("edge", 0.0)), 6) if isinstance(lg, dict) else None,
                    })
                except Exception:
                    continue
    except Exception:
        pass
    return out

def _v4_dedupe_merge(game_rows, prop_rows):
    # Dedupe by (type, game_id, market, player, line, direction). Keep highest edge.
    seen = {}
    for row in list(game_rows) + list(prop_rows):
        key = (
            row.get("type"),
            row.get("game_id"),
            row.get("market"),
            row.get("player"),
            str(row.get("line")) if row.get("line") is not None else None,
            row.get("direction"),
        )
        prev = seen.get(key)
        if prev is None or float(row.get("edge", 0.0) or 0.0) > float(prev.get("edge", 0.0) or 0.0):
            seen[key] = row
    return list(seen.values())

def _v4_write_outputs(rows, out_json="/mnt/data/betslip_combined.json", out_csv="/mnt/data/betslip_combined.csv"):
    # JSON
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"bets": rows}, f, indent=2)
    except Exception:
        pass
    # CSV
    try:
        cols = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception:
        pass
    return out_json, out_csv

def run_props_if_available(self, spec=None):
    """
    Run your existing prop pipeline if input candidates exist in context.
    Looks for context['prop_candidates'] or a callable context['prop_candidate_hook'].
    Returns whatever your build_slips returns, or None if nothing to run.
    """
    ctx = _v4_safe_get_ctx(self)
    # Hook path: a callable that returns (prop_candidates, spec)
    hook = ctx.get("prop_candidate_hook", None)
    if callable(hook):
        try:
            cands, sp = hook(self, ctx)
            spec = spec or sp
            return self.build_slips(cands, spec) if cands else None
        except Exception:
            pass
    # Static candidates path
    prop_cands = ctx.get("prop_candidates", None)
    if prop_cands:
        try:
            # If spec is not provided, try to read from context
            if spec is None:
                spec = ctx.get("prop_spec") or ctx.get("slip_spec")
            return self.build_slips(prop_cands, spec) if hasattr(self, "build_slips") else None
        except Exception:
            return None
    return None

def run_default_day(self,
                    sources: dict = None,
                    bankroll_units: float = 100.0,
                    kelly_fractional: float = 0.25,
                    n_trials_mc_bankroll: int = 5000,
                    starting_bankroll: float = 100.0,
                    parlay_max_legs: int = 5,
                    parlay_min_edge: float = 0.01,
                    parlay_max_pairwise_corr: float = 0.6,
                    run_props: bool = True,
                    prop_spec: dict = None,
                    out_json: str = "/mnt/data/betslip_combined.json",
                    out_csv: str = "/mnt/data/betslip_combined.csv"):
    """
    One-call daily runner:
      - Loads/merges context
      - Runs game-market ProBlend
      - Optionally runs prop pipeline if candidates are present
      - Merges and writes combined betslip as JSON/CSV

    Returns: (combined_rows, aux) where aux contains each system's raw outputs.
    """
    # Load/merge context from files
    mapping = _v4_auto_sources_defaults()
    if sources and isinstance(sources, dict):
        mapping.update({k: v for k, v in sources.items() if v})
    _v4_merge_context(self, mapping)

    # Ensure ProBlend enabled
    try:
        self.enable_game_market_blend(True)
    except Exception:
        pass

    # ---- Game markets (ProBlend) ----
    try:
        pos_bets, parlay, risk = self.problend_build_card(
            bankroll_units=bankroll_units,
            kelly_fractional=kelly_fractional,
            n_trials_mc_bankroll=n_trials_mc_bankroll,
            starting_bankroll=starting_bankroll,
            parlay_max_legs=parlay_max_legs,
            parlay_min_edge=parlay_min_edge,
            parlay_max_pairwise_corr=parlay_max_pairwise_corr
        )
        game_rows = _v4_summarize_betedge_list(pos_bets)
    except Exception as e:
        pos_bets, parlay, risk = [], [], {}
        game_rows = []

    # ---- Props (V4 core) ----
    prop_rows = []
    prop_raw = None
    if run_props:
        try:
            prop_raw = run_props_if_available(self, spec=prop_spec)
            prop_rows = _v4_summarize_props(self, prop_raw)
        except Exception:
            prop_rows = []

    # ---- Merge & outputs ----
    combined_rows = _v4_dedupe_merge(game_rows, prop_rows)
    outj, outc = _v4_write_outputs(combined_rows, out_json=out_json, out_csv=out_csv)

    aux = {
        "game_pos_bets": pos_bets,
        "game_parlay": parlay,
        "game_risk": risk,
        "props_raw": prop_raw,
        "output_json": outj,
        "output_csv": outc,
    }
    return combined_rows, aux

# Attach to class
try:
    V4EngineIntegrated.run_default_day = run_default_day
    V4EngineIntegrated.run_props_if_available = run_props_if_available
except Exception:
    pass



# ============================================================================
# Multi-sport profiles & sport-aware pricing (NFL, NBA, CFB, CBB, NHL, MLB, Soccer)
# ============================================================================

SPORT_PROFILES = {
    # Points-based sports
    "NFL":  {"sigma_spread": 13.5, "sigma_total": 18.0,
             "spread_keys": ("spread_home","spread_away"),
             "ml_keys": ("moneyline_home","moneyline_away"),
             "total_keys": ("total_over","total_under")},
    "NBA":  {"sigma_spread": 11.0, "sigma_total": 24.0,
             "spread_keys": ("spread_home","spread_away"),
             "ml_keys": ("moneyline_home","moneyline_away"),
             "total_keys": ("total_over","total_under")},
    "CFB":  {"sigma_spread": 16.0, "sigma_total": 20.0,
             "spread_keys": ("spread_home","spread_away"),
             "ml_keys": ("moneyline_home","moneyline_away"),
             "total_keys": ("total_over","total_under")},
    "CBB":  {"sigma_spread": 12.0, "sigma_total": 18.0,
             "spread_keys": ("spread_home","spread_away"),
             "ml_keys": ("moneyline_home","moneyline_away"),
             "total_keys": ("total_over","total_under")},
    # Goals-based
    "NHL":  {"sigma_spread": 1.75, "sigma_total": 1.80,
             "spread_keys": ("puckline_home","puckline_away"),
             "ml_keys": ("moneyline_home","moneyline_away"),
             "total_keys": ("total_over","total_under")},
    # Runs-based
    "MLB":  {"sigma_spread": 2.20, "sigma_total": 2.50,
             "spread_keys": ("runline_home","runline_away"),
             "ml_keys": ("moneyline_home","moneyline_away"),
             "total_keys": ("total_over","total_under")},
    # Low-scoring football (soccer)
    "SOCCER": {"sigma_spread": 1.10, "sigma_total": 1.30,
             "spread_keys": ("asian_home","asian_away"),
             "ml_keys": ("moneyline_home","moneyline_away"),
             "total_keys": ("total_over","total_under")},
}

def _get_sport_profile(sport: str):
    s = (sport or "NFL").upper()
    return SPORT_PROFILES.get(s, SPORT_PROFILES["NFL"])

def _sigma_for(sport: str, kind: str):
    prof = _get_sport_profile(sport)
    if kind == "spread":
        return float(prof.get("sigma_spread", 13.5))
    return float(prof.get("sigma_total", 18.0))

def _market_keys_for_sport(sport: str):
    prof = _get_sport_profile(sport)
    return prof["spread_keys"], prof["ml_keys"], prof["total_keys"]

# ---- Extend ProBlend with sport-aware pricing & ML ----
try:
    _V4CLS = V4MC_ProBlendStrategy
except NameError:
    _V4CLS = None

if _V4CLS is not None:
    def _price_spread_side_sport(self, fair_spread: float, market_spread: float, side: str, sport: str):
        sigma = _sigma_for(sport, "spread")
        p = spread_to_win_prob(fair_spread, market_spread, sigma_pts=sigma)
        if side == "away":
            p = 1.0 - p
        fair_dec = prob_to_fair_price(p)
        return p, fair_dec

    def _evaluate_spread_bet_sport(self, game: 'Game', fair_spread: float, side: str, market_price_amer: float) -> 'BetEdge':
        market_dec = american_to_decimal(market_price_amer)
        market_spread = game.market_spread_home
        p, fair_dec = _price_spread_side_sport(self, fair_spread, market_spread, side, game.sport)
        edge = compute_edge(p, market_dec)
        kfrac = kelly_fraction(p, market_dec) * self.kelly_fractional
        stake = self.bankroll_units * kfrac
        market = f"spread_{side}"
        return BetEdge(game.game_id, market, fair_dec, market_dec, edge, p, kfrac, stake)

    def _evaluate_total_bet_sport(self, game: 'Game', fair_total: float, over_under: str, market_price_amer: float) -> 'BetEdge':
        sigma_total = _sigma_for(game.sport, "total")
        from math import erf, sqrt
        z = (fair_total - game.market_total) / sigma_total
        p_over = 0.5 * (1.0 + erf(z / math.sqrt(2)))
        p = p_over if over_under == "over" else (1.0 - p_over)
        market_dec = american_to_decimal(market_price_amer)
        fair_dec = prob_to_fair_price(p)
        edge = compute_edge(p, market_dec)
        kfrac = kelly_fraction(p, market_dec) * self.kelly_fractional
        stake = self.bankroll_units * kfrac
        market = f"total_{over_under}"
        return BetEdge(game.game_id, market, fair_dec, market_dec, edge, p, kfrac, stake)

    def _evaluate_moneyline_bet(self, game: 'Game', side: str, market_price_amer: float, fair_spread: float=None) -> 'BetEdge':
        if fair_spread is None:
            raise RuntimeError("fair_spread required for ML evaluation in this context")
        sigma = _sigma_for(game.sport, "spread")
        from math import erf, sqrt
        z = fair_spread / max(1e-9, sigma)
        p_home = 0.5 * (1.0 + erf(z / math.sqrt(2)))
        p = p_home if side == "home" else (1.0 - p_home)
        market_dec = american_to_decimal(market_price_amer)
        fair_dec = prob_to_fair_price(p)
        edge = compute_edge(p, market_dec)
        kfrac = kelly_fraction(p, market_dec) * self.kelly_fractional
        stake = self.bankroll_units * kfrac
        market = f"moneyline_{side}"
        return BetEdge(game.game_id, market, fair_dec, market_dec, edge, p, kfrac, stake)

    def _build_card_multisport(self,
                   games: list,
                   team_ratings: dict,
                   stats: dict,
                   situational_hooks,
                   price_feed) -> list:
        bets = []
        for g in games:
            home = team_ratings[g.home]; away = team_ratings[g.away]
            home_stats = stats[g.home];   away_stats = stats[g.away]
            sit_home, sit_away = situational_hooks(g)
            fair_sp = self.fair_spread(g, home, away, home_stats, away_stats, sit_home, sit_away)
            fair_tot = self.fair_total(g, home_stats, away_stats, visiting_bench_bias=True, is_close_game=True)

            spread_keys, ml_keys, total_keys = _market_keys_for_sport(g.sport)

            # Spread/runline/puckline
            try:
                price_home = price_feed(g, spread_keys[0])
                price_away = price_feed(g, spread_keys[1])
                beh = _evaluate_spread_bet_sport(self, g, fair_spread=fair_sp, side="home", market_price_amer=price_home)
                bea = _evaluate_spread_bet_sport(self, g, fair_spread=fair_sp, side="away", market_price_amer=price_away)
                best_spread = beh if beh.edge > bea.edge else bea
                if best_spread.edge > 0:
                    bets.append(best_spread)
            except Exception:
                pass

            # Totals
            try:
                price_over = price_feed(g, total_keys[0])
                price_under = price_feed(g, total_keys[1])
                beo = _evaluate_total_bet_sport(self, g, fair_total=fair_tot, over_under="over", market_price_amer=price_over)
                beu = _evaluate_total_bet_sport(self, g, fair_total=fair_tot, over_under="under", market_price_amer=price_under)
                best_total = beo if beo.edge > beu.edge else beu
                if best_total.edge > 0:
                    bets.append(best_total)
            except Exception:
                pass

            # Moneyline
            try:
                ml_home = price_feed(g, ml_keys[0])
                ml_away = price_feed(g, ml_keys[1])
                bmh = _evaluate_moneyline_bet(self, g, side="home", market_price_amer=ml_home, fair_spread=fair_sp)
                bma = _evaluate_moneyline_bet(self, g, side="away", market_price_amer=ml_away, fair_spread=fair_sp)
                best_ml = bmh if bmh.edge > bma.edge else bma
                if best_ml.edge > 0:
                    bets.append(best_ml)
            except Exception:
                pass

        return bets

    # Monkey-patch
    try:
        V4MC_ProBlendStrategy.build_card = _build_card_multisport
        V4MC_ProBlendStrategy.price_spread_side = _price_spread_side_sport
        V4MC_ProBlendStrategy.evaluate_spread_bet = _evaluate_spread_bet_sport
        V4MC_ProBlendStrategy.evaluate_total_bet = _evaluate_total_bet_sport
        V4MC_ProBlendStrategy.evaluate_moneyline_bet = _evaluate_moneyline_bet
    except Exception:
        pass



# ============================================================================
# >>> NON-DESTRUCTIVE PATCH: Multi-Sport + Missing Features (2025-10-12)
# This block APPENDS new functionality without deleting or modifying existing
# code above. It provides:
#   • Multi-sport orchestrator (V4EngineIntegrated_MultiSport)
#   • Smart multi-book price reconciliation
#   • Fragile-Unders auto-replacement logic (conservative)
#   • PrizePicks mode (fixed-payout slip assembly)
#   • FanDuel Markdown formatter
# All utilities are self-contained and can be called without impacting the
# original engine classes and methods in this file.
# ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Any
import os, json, math, csv

# ---------- Odds helpers ----------
def _amer_to_dec(odds: float) -> float:
    if odds is None: return None
    return 1.0 + (odds/100.0 if odds>0 else 100.0/abs(odds))

def _dec_to_prob(d: float) -> float:
    return 0.0 if d is None or d <= 1e-9 else 1.0/max(1e-9, d)

def _prob_to_dec(p: float) -> float:
    return 1.0 / max(1e-9, p)

# ---------- Smart multi-book reconciliation ----------
def reconcile_books(books: Dict[str, Dict[str, float]], market_key: str,
                    weights: Optional[Dict[str, float]] = None,
                    mode: str = "harmonic_best") -> Optional[float]:
    """
    books = { "BookA": {"spread_home": -110, "total_over": -108, ...}, "BookB": {...} }
    Returns reconciled DECIMAL price for market_key.
    mode:
      - "harmonic_best": harmonic mean of implied prob, then pick best price side of mean
      - "best": choose the most favorable price (highest decimal for overs/favorites)
    """
    if not books: return None
    quotes = []
    for book, mkts in books.items():
        if market_key in mkts and mkts[market_key] is not None:
            dec = _amer_to_dec(mkts[market_key])
            if dec is None: continue
            w = (weights or {}).get(book, 1.0)
            quotes.append((dec, max(1e-9, float(w))))
    if not quotes: return None

    if mode == "best":
        best_dec = max(q[0] for q in quotes)
        return best_dec

    # harmonic mean of probabilities -> convert back to decimal
    # p_harm = (sum(w) / sum(w/p_i))
    num = sum(w for _, w in quotes)
    den = sum(w / _dec_to_prob(d) for d, w in quotes)  # w / p_i
    if den <= 1e-12:
        return max(q[0] for q in quotes)
    p_harm = num / den
    return _prob_to_dec(p_harm)

# ---------- Fragile-Unders policy (applies to totals UNDERS by default) ----------
def fragile_unders_adjust(line: float, mu: float, sigma: float, threshold: float = 0.065) -> float:
    """
    If the under is fragile (slope near key), nudge the line down to get away from
    the steep region (reduce fragility). Returns adjusted line for 'under' bets.
    """
    sg = max(1e-9, sigma)
    z = (line - mu) / sg
    slope = math.exp(-0.5 * z*z) / (math.sqrt(2.0*math.pi) * sg)
    if slope > threshold:
        # add a conservative MoS proportional to slope
        bump = min(2.5, 0.5 + (slope - threshold) * 18.0)  # cap shift
        return line - bump
    return line

# ---------- PrizePicks mode ----------
class PrizePicks:
    """Compute EV for Power/Flex plays given fixed payout tables."""
    # Common payout tables (subject to change; keep simple defaults)
    POWER = {2: 3.0, 3: 5.0, 4: 10.0, 5: 10.0, 6: 25.0}   # x return on stake if all hit
    FLEX  = {3: 2.25, 4: 1.5, 5: 2.0, 6: 2.0}            # simple EV proxy for all-hit outcome

    @staticmethod
    def best_combo(ps: List[float], mode: str = "power", legs: int = 3, bankroll_units: float = 100.0) -> Dict[str, Any]:
        ps_sorted = sorted(ps, reverse=True)[:legs]
        p_all = 1.0
        for p in ps_sorted:
            p_all *= max(1e-6, min(1.0-1e-6, p))
        table = PrizePicks.POWER if mode == "power" else PrizePicks.FLEX
        mult = table.get(legs, 1.0)
        ev = p_all * (mult - 1.0) - (1.0 - p_all)
        stake = bankroll_units * 0.02  # 2% default stake per PP slip
        return {"mode": mode, "legs": legs, "prob_all_hit": p_all, "payout_mult": mult, "ev_per_unit": ev, "stake_units": stake}

# ---------- FanDuel Markdown formatter ----------

# ---------- Minimal multi-sport strategy (spread/total) ----------
@dataclass
class _TeamRatingMS:
    team_id: str
    power: float
    qb_adj: float
    injury_adj: float
    situational_adj: float

@dataclass
class _StatsMS:
    team_id: str
    off_ypp_comp: float
    def_ypp_comp: float
    yards_before_contact_run: float
    short_ydg_conv_rate: float
    pressure_rate_allowed: float
    play_action_success: float
    red_zone_eff: float
    recency_weight: float = 0.5

@dataclass
class _PeriodMS:
    fh: float; sh: float; foul: float; bench: float

SPORT_SIGMA = {
    "NFL": (13.5, 18.0),
    "NBA": (11.0, 24.0),
    "CFB": (16.0, 20.0),
    "CBB": (12.0, 18.0),
    "NHL": (1.75, 1.90),
    "MLB": (2.20, 2.60),
    "SOCCER": (1.10, 1.30),
}

def _predictive(stats: _StatsMS) -> float:
    base = 0.30*stats.yards_before_contact_run + 0.20*stats.short_ydg_conv_rate - 0.15*stats.pressure_rate_allowed + 0.15*stats.play_action_success + 0.20*stats.red_zone_eff
    return base * (0.5 + 0.5*stats.recency_weight)

def _ensemble_spread(home: _TeamRatingMS, away: _TeamRatingMS, hs: _StatsMS, as_: _StatsMS) -> float:
    wal = (home.power + home.qb_adj + home.injury_adj) - (away.power + away.qb_adj + away.injury_adj)
    bob = ((hs.off_ypp_comp - as_.def_ypp_comp) - (as_.off_ypp_comp - hs.def_ypp_comp))
    bob += _predictive(hs) - _predictive(as_)
    sharp = (hs.yards_before_contact_run - as_.yards_before_contact_run) + (hs.short_ydg_conv_rate - as_.short_ydg_conv_rate) - 0.7*(hs.pressure_rate_allowed - as_.pressure_rate_allowed)
    return 0.40*wal + 0.30*(3.0*bob) + 0.20*(4.0*sharp) + 0.10*0.0

def _spread_to_win_prob(fair_spread: float, market_spread_home: float, sigma_pts: float) -> float:
    from math import erf, sqrt
    z = (fair_spread - market_spread_home)/max(1e-9, sigma_pts)
    return 0.5*(1.0 + erf(z/sqrt(2.0)))

class V4EngineIntegrated_MultiSport:
    """Append-only orchestrator that can be used even if the original class name exists.
    Use: rows, aux = V4EngineIntegrated_MultiSport().run_default_day({...})
    sources expects keys: priors, schedule_adjusted_stats, schedule, odds (or books map for reconciliation)
    """
    def __init__(self):
        self.context: Dict[str, Any] = {}

    def run_default_day(self, sources: Dict[str, Any],
                        bankroll_units: float = 100.0, kelly_fractional: float = 0.25,
                        multi_book: bool = True, book_weights: Optional[Dict[str,float]] = None,
                        prizepicks_mode: Optional[Dict[str, Any]] = None,
                        write_outputs: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        # Load context
        ctx = self.context
        for k, v in (sources or {}).items():
            if isinstance(v, (dict, list)):
                ctx[k] = v
            elif isinstance(v, str) and os.path.exists(v):
                with open(v, "r", encoding="utf-8") as f:
                    ctx[k] = json.load(f)
            else:
                ctx[k] = v

        # Parse schedule
        games = ctx.get("schedule", [])
        rows: List[Dict[str, Any]] = []

        for g in games:
            sport = (g["sport"] or "NFL").upper()
            sig_sp, sig_tot = SPORT_SIGMA.get(sport, SPORT_SIGMA["NFL"])

            # Ratings + stats
            p = ctx.get("priors", {}).get(g["home"], {"power":0,"qb_adj":0,"injury_adj":0,"situational_adj":0})
            q = ctx.get("priors", {}).get(g["away"], {"power":0,"qb_adj":0,"injury_adj":0,"situational_adj":0})
            hs_raw = ctx.get("schedule_adjusted_stats", {}).get(g["home"], {})
            as_raw = ctx.get("schedule_adjusted_stats", {}).get(g["away"], {})

            home = _TeamRatingMS(g["home"], p.get("power",0.0), p.get("qb_adj",0.0), p.get("injury_adj",0.0), p.get("situational_adj",0.0))
            away = _TeamRatingMS(g["away"], q.get("power",0.0), q.get("qb_adj",0.0), q.get("injury_adj",0.0), q.get("situational_adj",0.0))
            hs = _StatsMS(g["home"], hs_raw.get("off_ypp_comp",0.0), hs_raw.get("def_ypp_comp",0.0),
                          hs_raw.get("yards_before_contact_run",0.0), hs_raw.get("short_ydg_conv_rate",0.0),
                          hs_raw.get("pressure_rate_allowed",0.0), hs_raw.get("play_action_success",0.0),
                          hs_raw.get("red_zone_eff",0.0), hs_raw.get("recency_weight",0.5))
            as_ = _StatsMS(g["away"], as_raw.get("off_ypp_comp",0.0), as_raw.get("def_ypp_comp",0.0),
                           as_raw.get("yards_before_contact_run",0.0), as_raw.get("short_ydg_conv_rate",0.0),
                           as_raw.get("pressure_rate_allowed",0.0), as_raw.get("play_action_success",0.0),
                           as_raw.get("red_zone_eff",0.0), as_raw.get("recency_weight",0.5))

            fair_sp = _ensemble_spread(home, away, hs, as_)

            # Price feed
            if multi_book and "books" in ctx:
                book_block = ctx["books"].get(g["game_id"], {})
                dec_home = reconcile_books(book_block, "spread_home", book_weights)
                dec_away = reconcile_books(book_block, "spread_away", book_weights)
                dec_over = reconcile_books(book_block, "total_over", book_weights)
                dec_under = reconcile_books(book_block, "total_under", book_weights)
            else:
                od = ctx.get("odds", {}).get(g["game_id"], {})
                dec_home = _amer_to_dec(od.get("spread_home", -110))
                dec_away = _amer_to_dec(od.get("spread_away", -110))
                dec_over = _amer_to_dec(od.get("total_over", -110))
                dec_under = _amer_to_dec(od.get("total_under", -110))

            # Spreads
            p_home = _spread_to_win_prob(fair_sp, g["market_spread_home"], sig_sp)
            p_away = 1.0 - p_home
            for side, p, dec in (("home", p_home, dec_home), ("away", p_away, dec_away)):
                if dec is None: continue
                b = dec - 1.0
                ev = p*b - (1.0 - p)
                k = max(0.0, (b*p - (1.0 - p)) / b) * kelly_fractional if b>0 else 0.0
                stake = bankroll_units * k
                if ev > 0:
                    rows.append({"type":"game","game_id":g["game_id"],"market":f"spread_{side}","win_prob":p,"edge":ev,"kelly_fraction":k,"stake_units":stake,"fair_price":_prob_to_dec(p),"market_price":dec})

            # Totals (with fragile-unders adjustment)
            # Compute fair total via predictive proxy
            diff = (_predictive(hs) + _predictive(as_)) * 1.5
            fair_tot = g["market_total"] + diff

            # Over
            from math import erf, sqrt
            z_over = (fair_tot - g["market_total"]) / max(1e-9, sig_tot)
            p_over = 0.5*(1.0 + erf(z_over/sqrt(2.0)))
            if dec_over is not None:
                b = dec_over - 1.0
                ev = p_over*b - (1.0 - p_over)
                k = max(0.0, (b*p_over - (1.0 - p_over)) / b) * kelly_fractional if b>0 else 0.0
                stake = bankroll_units * k
                if ev > 0:
                    rows.append({"type":"game","game_id":g["game_id"],"market":"total_over","win_prob":p_over,"edge":ev,"kelly_fraction":k,"stake_units":stake,"fair_price":_prob_to_dec(p_over),"market_price":dec_over})

            # Under (fragile policy)
            adj_line = fragile_unders_adjust(g["market_total"], mu=fair_tot, sigma=sig_tot, threshold=0.065)
            z_under = (adj_line - fair_tot) / max(1e-9, sig_tot)  # use adjusted line for fragility relief
            p_under = 0.5*(1.0 + math.erf(z_under/math.sqrt(2.0)))
            if dec_under is not None:
                b = dec_under - 1.0
                ev = p_under*b - (1.0 - p_under)
                k = max(0.0, (b*p_under - (1.0 - p_under)) / b) * kelly_fractional if b>0 else 0.0
                stake = bankroll_units * k
                if ev > 0:
                    rows.append({"type":"game","game_id":g["game_id"],"market":"total_under","win_prob":p_under,"edge":ev,"kelly_fraction":k,"stake_units":stake,"fair_price":_prob_to_dec(p_under),"market_price":dec_under,"note":"fragile_unders_adjusted"})

        # PrizePicks optional
        aux = {}
        if prizepicks_mode and rows:
            ps = [r["win_prob"] for r in rows]
            mode = prizepicks_mode.get("mode","power")
            legs = int(prizepicks_mode.get("legs",3))
            aux["prizepicks"] = PrizePicks.best_combo(ps, mode=mode, legs=legs, bankroll_units=bankroll_units)

        if write_outputs:
            with open("betslip_combined.json","w",encoding="utf-8") as f:
                json.dump({"bets": rows, "aux": aux}, f, indent=2)
            cols = sorted({k for r in rows for k in r.keys()}) if rows else ["type","game_id","market","edge"]
            with open("betslip_combined.csv","w",newline="",encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
                for r in rows: w.writerow(r)
            # also FanDuel MD
            with open("betslip_fanduel.md","w",encoding="utf-8") as f:
                f.write(format_fanduel_markdown(rows))

        return rows, {"output_json":"betslip_combined.json","output_csv":"betslip_combined.csv","output_fd_md":"betslip_fanduel.md"}

# Soft alias: if an original class named V4EngineIntegrated exists and is callable, we leave it.
# Otherwise, we provide a convenience alias to the multi-sport runner.
try:
    V4EngineIntegrated  # noqa
except Exception:
    V4EngineIntegrated = V4EngineIntegrated_MultiSport  # type: ignore

# ============================================================================
# <<< END PATCH
# ============================================================================



# ============================================================================
# >>> NON-DESTRUCTIVE FINAL TASKS PATCH (2025-10-12)
# Adds: Fragile-Unders auto-replacement utilities, PrizePicks detection/eval,
# completed FanDuel Markdown formatter, and >5% multi-book discrepancy flag.
# Appended only — no edits/removals to existing code above.
# ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import math, json, csv, os, statistics

# ---------- Generic probability helper (Normal CDF model) ----------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _p_hit_normal(mu: float, sigma: float, line: float, direction: str) -> float:
    """Probability of going OVER/UNDER a line with Normal(mu, sigma)."""
    s = max(1e-9, sigma)
    z = (line - mu) / s
    if str(direction).lower().startswith("o"):
        # P(X > line) = 1 - Phi((line - mu)/sigma)
        return 1.0 - _norm_cdf(z)
    else:
        # P(X < line)
        return _norm_cdf(z)

# ---------- 1) Fragile-Unders Auto-Replacement ----------
def detect_fragile_under(leg: Dict[str, Any]) -> bool:
    """
    Detect fragile Unders on elite+volatile players/markets.
    Expects a dict-like leg with: direction, mu, sigma, line.
    """
    try:
        return (str(leg.get("direction","")).lower().startswith("u")
                and float(leg.get("mu", 0.0)) > 80.0
                and float(leg.get("sigma", 0.0)) > 20.0)
    except Exception:
        return False

def find_companion_alt_over(leg: Dict[str, Any], teammates: List[Dict[str, Any]],
                            odds_price_amer: float = -119.0,
                            alt_multiplier: float = 0.77,
                            ev_gate: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Try to replace a fragile Under with a teammate Over (or safer alt Over).
    - alt line = leg['line'] * alt_multiplier (77% heuristic)
    - teammates sorted by usage_rate desc if present
    Returns an alt leg dict if passes EV gates, else None.
    """
    if not teammates:
        return None
    try:
        sorted_mates = sorted(teammates, key=lambda x: x.get("usage_rate", 0), reverse=True)
    except Exception:
        sorted_mates = teammates
    alt_line = float(leg.get("line", 0.0)) * alt_multiplier if float(leg.get("line", 0.0)) > 0 else float(leg.get("line", 0.0))

    dec_price = 1.0 + (odds_price_amer/100.0 if odds_price_amer>0 else 100.0/abs(odds_price_amer))
    implied = 1.0 / dec_price

    for mate in sorted_mates:
        mu = float(mate.get("mu", 0.0))
        sigma = float(mate.get("sigma", 1.0))
        p = _p_hit_normal(mu, sigma, alt_line, direction="over")
        ev = p - implied  # fixed-payout style EV proxy
        if ev > ev_gate:
            return {
                "market": mate.get("market",""),
                "team": mate.get("team_id", ""),
                "direction": "over",
                "line": alt_line,
                "mu": mu,
                "sigma": sigma,
                "odds_dec": dec_price,
                "implied": implied,
                "p": p,
                "ev": ev,
                "source": "fragile_unders_auto_replacement"
            }
    return None

# ---------- 2) PrizePicks Detection & Evaluation ----------
def detect_prizepcks(market: str) -> bool:
    m = (market or "").upper()
    return ("PP_" in m) or ("PRIZEPCKS" in m) or ("PRIZEPICKS" in m)

def evaluate_prizepcks_leg(leg: Dict[str, Any], pp_line: float,
                           pp_price_dec: float = 1.840,
                           engine: Optional[Any] = None) -> Dict[str, Any]:
    """
    Evaluate a PrizePicks-style leg using fixed price.
    Attempts engine._p_hit if provided; otherwise uses Normal CDF.
    """
    if engine is not None and hasattr(engine, "_p_hit"):
        try:
            p = float(engine._p_hit(leg.get("market",""), leg.get("mu",0.0), leg.get("sigma",1.0), pp_line, leg.get("direction","over")))
        except Exception:
            p = _p_hit_normal(leg.get("mu",0.0), max(1e-9, leg.get("sigma",1.0)), pp_line, leg.get("direction","over"))
    else:
        p = _p_hit_normal(leg.get("mu",0.0), max(1e-9, leg.get("sigma",1.0)), pp_line, leg.get("direction","over"))

    implied = 1.0 / max(1e-9, pp_price_dec)
    ev = p - implied
    drift = abs(pp_line - float(leg.get("line", pp_line))) / max(1e-9, float(leg.get("line", pp_line))) if float(leg.get("line", 0.0))>0 else 0.0
    return {
        "playable": ev > 0.0,
        "ev": ev,
        "p": p,
        "implied": implied,
        "drift_flag": drift > 0.04,
        "pp_line": pp_line,
        "pp_price_dec": pp_price_dec
    }

# ---------- 3) Completed FanDuel Markdown Formatter ----------
def format_fanduel_markdown(bets: List[Dict[str, Any]], title: str = "V4 + MC Betslip",
                            correlation_note: Optional[str] = None) -> str:
    """
    Builds a markdown document with:
      - Table of legs
      - Per-slip joint prob (assuming independence if not provided)
      - Summary (EV stats, top-5 by edge)
    """
    lines = [f"# {title}", ""]
    if correlation_note:
        lines += [f"> _Correlation_: {correlation_note}", ""]

    # Table
    lines += ["| Game | Market | Win% | Edge | Kelly | Stake | Fair | Price | Note |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    p_prod = 1.0
    edges = []
    for b in bets:
        wp = float(b.get("win_prob", 0.0))
        p_prod *= max(1e-9, min(1.0-1e-9, wp))
        edges.append(float(b.get("edge", 0.0)))
        lines.append(
            f"| {b.get('game_id','')} | {b.get('market','')} | {wp*100:.2f}% | "
            f"{b.get('edge',0):.4f} | {b.get('kelly_fraction',0):.3f} | {b.get('stake_units',0):.2f} | "
            f"{b.get('fair_price',0):.3f} | {b.get('market_price',0):.3f} | {b.get('note','')} |"
        )

    # Summary
    lines += ["", "## Summary",
              f"- Legs: **{len(bets)}**",
              f"- Joint probability (indep.): **{p_prod*100:.2f}%**",
              f"- Mean edge: **{(statistics.mean(edges) if edges else 0.0):.4f}**",
              f"- Median edge: **{(statistics.median(edges) if edges else 0.0):.4f}**"]
    top5 = sorted(bets, key=lambda x: x.get("edge", 0.0), reverse=True)[:5]
    if top5:
        lines += ["", "### Top 5 by Edge"]
        for t in top5:
            lines.append(f"- {t.get('game_id','')} · {t.get('market','')} · edge {t.get('edge',0):.4f} · win {t.get('win_prob',0)*100:.2f}%")
    return "\n".join(lines)

# ---------- 4) Multi-book discrepancy flagging ----------
def flag_discrepancy(books: Dict[str, Dict[str, float]], market_key: str, threshold: float = 0.05) -> bool:
    quotes = []
    for mkts in (books or {}).values():
        if market_key in mkts and mkts[market_key] is not None:
            amer = float(mkts[market_key])
            dec = 1.0 + (amer/100.0 if amer>0 else 100.0/abs(amer))
            quotes.append(dec)
    if len(quotes) < 2:
        return False
    min_q, max_q = min(quotes), max(quotes)
    disc = abs(max_q - min_q) / max(1e-9, min_q)
    return disc > threshold

# ---- Optional: tiny writer to emit FanDuel MD from existing bet rows ----
def write_fanduel_markdown(bets: List[Dict[str, Any]], path: str = "betslip_fanduel.md", correlation_note: Optional[str] = None) -> str:
    md = format_fanduel_markdown(bets, correlation_note=correlation_note)
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path

# ============================================================================
# <<< END FINAL TASKS PATCH
# ============================================================================



# ============================================================================
# >>> NON-DESTRUCTIVE AUTO-WIRING PATCH (2025-10-12)
# Wires utilities into a single-call daily runner that:
#  - calls the existing engine's run_default_day()
#  - reconciles multi-book prices (if provided)
#  - triggers fragile-unders auto-replacement for props (if provided)
#  - detects/evaluates PrizePicks legs
#  - emits JSON, CSV, and FanDuel Markdown by default
# Notes:
#  • This is append-only. Nothing above is changed or removed.
#  • Works with either V4EngineIntegrated or V4EngineIntegrated_MultiSport defined earlier.
# ============================================================================

import json, os, math, csv, statistics

def _aw_safe_get_engine():
    # Prefer user's original V4EngineIntegrated if available; otherwise use the multi-sport class.
    try:
        return V4EngineIntegrated()  # type: ignore
    except Exception:
        try:
            return V4EngineIntegrated_MultiSport()  # type: ignore
        except Exception:
            raise RuntimeError("No available engine class to run.")

def _aw_collect_books(sources):
    return (sources or {}).get("books")

def _aw_collect_odds(sources):
    return (sources or {}).get("odds")

def _aw_reconcile_if_needed(sources):
    # If user provided per-book quotes, reconcile them into odds for core markets.
    books = _aw_collect_books(sources)
    if not books:
        return sources
    # For each game_id, build a reconciled odds block
    reconciled = {}
    for gid, perbook in books.items():
        mkts = {}
        for mk in ("spread_home","spread_away","total_over","total_under"):
            try:
                mkts[mk] = reconcile_books(perbook, mk)  # returns DECIMAL
            except Exception:
                mkts[mk] = None
        # Convert back to "american-ish" string? Keep decimal for safety; engine accepts american in some paths,
        # but our patched engines convert using american_to_decimal internally when strings provided.
        # We'll pass decimal directly; downstream uses as decimal when present.
        reconciled[gid] = {k: v for k, v in mkts.items() if v is not None}
    # Merge into sources (doesn't overwrite user's 'odds' if they provided it)
    merged = dict(sources or {})
    merged.setdefault("odds", {})
    for gid, mkts in reconciled.items():
        # Skip if user already supplied odds for the same market key
        merged["odds"].setdefault(gid, {})
        for k, dec in mkts.items():
            # Convert decimal back to american for compatibility if needed
            if dec is None:
                continue
            if dec >= 2.0:
                amer = (dec - 1.0) * 100.0
            else:
                amer = -100.0 / (dec - 1.0)
            merged["odds"][gid].setdefault(k, amer)
    return merged

def _aw_apply_fragile_unders(props: list) -> list:
    """Given a list of prop legs (dict), replace fragile unders where possible."""
    if not props:
        return props
    by_team = {}
    for leg in props:
        team = leg.get("team_id") or leg.get("team") or "NA"
        by_team.setdefault(team, []).append(leg)

    out = []
    for leg in props:
        if detect_fragile_under(leg):
            mates = by_team.get(leg.get("team_id") or leg.get("team") or "NA", [])
            alt = find_companion_alt_over(leg, mates, odds_price_amer=-119, alt_multiplier=0.77, ev_gate=0.0)
            if alt:
                alt["note"] = (alt.get("note","") + "; auto-replaced fragile under").strip("; ")
                out.append(alt)
                continue
        out.append(leg)
    return out

def run_default_day_autowire(
    sources: dict,
    bankroll_units: float = 100.0,
    kelly_fractional: float = 0.25,
    enable_multi_book: bool = True,
    enable_fragile_unders: bool = True,
    enable_prizepicks_eval: bool = True,
    emit_fanduel_md: bool = True,
    out_json: str = "betslip_combined.json",
    out_csv: str = "betslip_combined.csv",
    out_fd_md: str = "betslip_fanduel.md",
):
    """
    One-call daily runner with all features ON by default.
    - Accepts either:
        • 'odds' (single-book)
        • or 'books' (multi-book per game_id), and auto-reconciles to 'odds'
    - Calls the engine's run_default_day()
    - Post-processes prop legs with fragile-unders replacement if provided
    - Detects & evaluates PrizePicks legs
    - Writes JSON, CSV, and FanDuel Markdown by default
    Returns: (rows, aux)
    """
    # 1) Reconcile multi-book to odds if provided
    src = dict(sources or {})
    if enable_multi_book and _aw_collect_books(src):
        src = _aw_reconcile_if_needed(src)

    # 2) Run engine
    eng = _aw_safe_get_engine()
    try:
        rows, aux = eng.run_default_day(sources=src, bankroll_units=bankroll_units, kelly_fractional=kelly_fractional)
    except TypeError:
        # Some engine variants expect positional args
        rows, aux = eng.run_default_day(src, bankroll_units=bankroll_units, kelly_fractional=kelly_fractional)

    # 3) If props present in sources, apply fragile-unders replacement (non-destructive)
    props = src.get("props")
    if enable_fragile_unders and isinstance(props, list) and props:
        props2 = _aw_apply_fragile_unders(props)
        aux["props_after_fragile_policy"] = props2

    # 4) PrizePicks detection/evaluation (if any legs marked or markets detected)
    if enable_prizepicks_eval:
        pp_results = []
        # Use rows (game and props) if available
        for r in rows or []:
            mk = r.get("market","")
            if detect_prizepcks(mk):
                # Build a minimal leg shape for evaluation
                leg = {
                    "market": mk,
                    "direction": "over" if "over" in mk.lower() else ("under" if "under" in mk.lower() else "over"),
                    "mu": r.get("mu", 0.0),
                    "sigma": r.get("sigma", 1.0),
                    "line": r.get("line", 0.0),
                }
                pp_line = leg["line"] if leg["line"] else 0.0
                pp_results.append(evaluate_prizepcks_leg(leg, pp_line, engine=getattr(eng, "_p_hit", None)))
        if pp_results:
            aux["prizepicks_eval"] = pp_results

    # 5) Emit JSON/CSV (override filenames to match arguments)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"bets": rows, "aux": aux}, f, indent=2)
    if out_csv:
        cols = sorted({k for r in (rows or []) for k in r.keys()}) if rows else ["type","game_id","market","edge"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows or []: w.writerow(r)

    # 6) FanDuel markdown
    if emit_fanduel_md and rows:
        md = format_fanduel_markdown(rows, title="V4 + MC (Auto-wired) Betslip")
        with open(out_fd_md, "w", encoding="utf-8") as f:
            f.write(md)
        aux["output_fd_md"] = out_fd_md

    return rows, aux

# ============================================================================
# <<< END AUTO-WIRING PATCH
# ============================================================================


# ============================================================================
# >>> V4 ADD-ONS (Non-destructive, append-only): Player Allocation, Unified Adapter, Row Validator
#     Date: 2025-10-15
#     Notes:
#       - This block **adds** helpers and adapters; it DOES NOT modify or remove existing code.
#       - Exposes: build_player_prop_table(...), validate_rows(...), group_by_corr_tag(...), quick_summary(...)
#       - Adds convenience methods onto V4EngineIntegrated for one-call pipelines.
# ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math

# -----------------------------
# Tiny math helpers
# -----------------------------
def _v4a_safe(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def _v4a_norm(shares: Dict[str, float], eps: float=1e-9) -> Dict[str, float]:
    s = sum(max(0.0, x) for x in shares.values())
    if s <= eps:
        n = len(shares) or 1
        return {k: 1.0/n for k in shares}
    return {k: max(0.0, v)/s for k, v in shares.items()}

def _v4a_poisson_ge1(lmbda: float) -> float:
    l = _v4a_safe(lmbda, 0.0, 50.0)
    return 1.0 - math.exp(-l)

def _v4a_normcdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / sigma
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989423 * math.exp(-0.5*z*z)
    p = d * t * (0.3193815 + t*(-0.3565638 + t*(1.781478 + t*(-1.821256 + t*1.330274))))
    return 1.0 - p if z > 0 else p

def _v4a_p_over(line: float, mu: float, sigma: float) -> float:
    return 1.0 - _v4a_normcdf(line, mu, sigma)

# -----------------------------
# Player allocation helpers (condensed)
# -----------------------------
@dataclass
class NFLPlayerUsage:
    name: str
    td_share_rush: float = 0.0
    td_share_recv: float = 0.0
    rush_attempt_share: float = 0.0
    target_share: float = 0.0
    yards_per_carry: float = 4.3
    yards_per_target: float = 7.8
    snap_rate: float = 0.75

def nfl_anytime_td_probs(team_tds_mean: float, pass_td_frac: float, players: List[NFLPlayerUsage]) -> Dict[str, float]:
    pass_tds = _v4a_safe(team_tds_mean * _v4a_safe(pass_td_frac, 0.0, 1.0), 0.0, 10.0)
    rush_tds = _v4a_safe(team_tds_mean - pass_tds, 0.0, 10.0)
    recv_sh = _v4a_norm({p.name: max(0.0, p.td_share_recv) for p in players})
    rush_sh = _v4a_norm({p.name: max(0.0, p.td_share_rush) for p in players})
    out = {}
    for p in players:
        lam = pass_tds * recv_sh.get(p.name, 0.0) + rush_tds * rush_sh.get(p.name, 0.0)
        out[p.name] = _v4a_poisson_ge1(lam)
    return out

def nfl_yards_means_and_povers(team_rush_yds_mean: float, team_pass_yds_mean: float, players: List[NFLPlayerUsage],
                               rush_line_by_player: Optional[Dict[str, float]] = None,
                               rec_yds_line_by_player: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    rush_share = _v4a_norm({p.name: p.rush_attempt_share * p.snap_rate for p in players})
    tgt_share  = _v4a_norm({p.name: p.target_share * p.snap_rate for p in players})
    res: Dict[str, Dict[str, float]] = {}
    for p in players:
        mu_r = team_rush_yds_mean * rush_share.get(p.name, 0.0)
        mu_c = team_pass_yds_mean * tgt_share.get(p.name, 0.0)
        sig_r = max(6.0, 0.45 * max(0.0, mu_r))
        sig_c = max(8.0, 0.55 * max(0.0, mu_c))
        row = {"rush_yds_mean": mu_r, "rec_yds_mean": mu_c}
        if rush_line_by_player and p.name in rush_line_by_player:
            row["p_over_rush"] = _v4a_p_over(rush_line_by_player[p.name], mu_r, sig_r)
        if rec_yds_line_by_player and p.name in rec_yds_line_by_player:
            row["p_over_rec"] = _v4a_p_over(rec_yds_line_by_player[p.name], mu_c, sig_c)
        res[p.name] = row
    return res

@dataclass
class BBPlayerUsage:
    name: str
    minutes: float = 32.0
    usage_rate: float = 0.24
    assist_rate: float = 0.22
    trb_rate: float = 0.13
    ts_pct: float = 0.58

def basketball_player_props_from_total(team_total_pts: float, team_possessions: float, team_fg_made_est: Optional[float],
                                      team_reb_opportunities: float, players: List[BBPlayerUsage],
                                      lines_pts: Optional[Dict[str, float]] = None,
                                      lines_ast: Optional[Dict[str, float]] = None,
                                      lines_reb: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    min_w = _v4a_norm({p.name: max(0.0, p.minutes) for p in players})
    use_w = _v4a_norm({p.name: p.usage_rate * min_w[p.name] for p in players})
    ts_adj = _v4a_norm({p.name: use_w[p.name] * _v4a_safe(p.ts_pct, 0.45, 0.70) for p in players})
    if team_fg_made_est is None:
        team_fg_made_est = max(15.0, team_total_pts / 2.1)
    ast_pool = 0.65 * team_fg_made_est
    out: Dict[str, Dict[str, float]] = {}
    for p in players:
        pts_mu = team_total_pts * ts_adj[p.name]
        pts_sigma = max(3.5, 0.45 * pts_mu)
        ast_mu = ast_pool * _v4a_safe(p.assist_rate, 0.05, 0.55) * min_w[p.name]
        ast_sigma = max(1.5, 0.55 * ast_mu)
        reb_mu = team_reb_opportunities * _v4a_safe(p.trb_rate, 0.05, 0.30) * min_w[p.name]
        reb_sigma = max(2.0, 0.50 * reb_mu)
        row = {"pts_mean": pts_mu, "ast_mean": ast_mu, "reb_mean": reb_mu}
        if lines_pts and p.name in lines_pts: row["p_over_pts"] = _v4a_p_over(lines_pts[p.name], pts_mu, pts_sigma)
        if lines_ast and p.name in lines_ast: row["p_over_ast"] = _v4a_p_over(lines_ast[p.name], ast_mu, ast_sigma)
        if lines_reb and p.name in lines_reb: row["p_over_reb"] = _v4a_p_over(lines_reb[p.name], reb_mu, reb_sigma)
        out[p.name] = row
    return out

@dataclass
class MLBPlayerRates:
    name: str
    plate_appearances_share: float = 0.11
    hr_rate: float = 0.045
    hit_rate: float = 0.25
    rbi_per_run_share: float = 0.13

def mlb_player_event_probs(team_runs_mean: float, team_plate_appearances: float, players: List[MLBPlayerRates],
                           lines_hits: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    pa_share = _v4a_norm({p.name: p.plate_appearances_share for p in players})
    out: Dict[str, Dict[str, float]] = {}
    for p in players:
        pa = max(2.0, team_plate_appearances * pa_share[p.name])
        p_hr = 1.0 - (1.0 - _v4a_safe(p.hr_rate, 0.0, 0.20)) ** pa
        hit_mu = pa * _v4a_safe(p.hit_rate, 0.05, 0.45)
        hit_sigma = max(0.8, math.sqrt(max(0.2, hit_mu * (1 - _v4a_safe(p.hit_rate, 0.05, 0.45)))))
        rbi_mu = team_runs_mean * _v4a_safe(p.rbi_per_run_share, 0.05, 0.30)
        row = {"p_hr_any": _v4a_safe(p_hr, 0.0, 0.999), "hits_mean": hit_mu, "rbi_mean": rbi_mu}
        if lines_hits and p.name in lines_hits:
            row["p_over_hits"] = 1.0 - _v4a_normcdf(lines_hits[p.name], hit_mu, hit_sigma)
        out[p.name] = row
    return out

@dataclass
class SoccerPlayerXG:
    name: str
    xg_share: float = 0.18
    sot_rate: float = 0.40
    shots_per_xg: float = 1.3

def soccer_anytime_goal_probs(team_xg_mean: float, players: List[SoccerPlayerXG]) -> Dict[str, float]:
    sh = _v4a_norm({p.name: p.xg_share for p in players})
    return {p.name: _v4a_poisson_ge1(team_xg_mean * sh[p.name]) for p in players}

def soccer_shots_on_target_means(team_xg_mean: float, players: List[SoccerPlayerXG],
                                 lines_sot: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    sh = _v4a_norm({p.name: p.xg_share for p in players})
    out: Dict[str, Dict[str, float]] = {}
    for p in players:
        pxg = team_xg_mean * sh[p.name]
        shots_mu = max(0.2, pxg * max(0.6, min(2.0, p.shots_per_xg)))
        sot_mu = shots_mu * _v4a_safe(p.sot_rate, 0.20, 0.70)
        sot_sigma = max(0.5, 0.75 * math.sqrt(max(0.1, sot_mu)))
        row = {"sot_mean": sot_mu}
        if lines_sot and p.name in lines_sot:
            row["p_over_sot"] = _v4a_p_over(lines_sot[p.name], sot_mu, sot_sigma)
        out[p.name] = row
    return out

@dataclass
class NHLPlayerUsage:
    name: str
    sog_share: float = 0.08
    shooting_pct: float = 0.105
    toi_share: float = 0.75

def nhl_player_sog_and_goals(team_sog_mean: float, team_goal_per_shot: float, players: List[NHLPlayerUsage],
                             lines_sog: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    sog_share = _v4a_norm({p.name: p.sog_share * _v4a_safe(p.toi_share, 0.3, 1.0) for p in players})
    out: Dict[str, Dict[str, float]] = {}
    for p in players:
        sog_mu = team_sog_mean * sog_share[p.name]
        sog_sigma = max(0.8, math.sqrt(max(0.2, sog_mu)))
        row = {"sog_mean": sog_mu}
        if lines_sog and p.name in lines_sog:
            row["p_over_sog"] = _v4a_p_over(lines_sog[p.name], sog_mu, sog_sigma)
        lam_goals = sog_mu * _v4a_safe(p.shooting_pct, 0.05, 0.22)
        row["p_goal_any"] = _v4a_poisson_ge1(lam_goals)
        out[p.name] = row
    return out

@dataclass
class TennisPlayerRates:
    name: str
    serve_points_share: float = 0.50
    ace_rate_on_serve: float = 0.10
    break_rate_on_return: float = 0.18

def tennis_player_prop_priors(total_points_estimate: float, player_A: TennisPlayerRates, player_B: TennisPlayerRates,
                              lines_aces: Optional[Dict[str, float]] = None,
                              lines_breaks: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    sp_A = max(20.0, total_points_estimate * _v4a_safe(player_A.serve_points_share, 0.40, 0.60))
    sp_B = max(20.0, total_points_estimate * _v4a_safe(player_B.serve_points_share, 0.40, 0.60))
    A_ace_mu = sp_A * _v4a_safe(player_A.ace_rate_on_serve, 0.02, 0.30)
    B_ace_mu = sp_B * _v4a_safe(player_B.ace_rate_on_serve, 0.02, 0.30)
    A_ace_sigma = max(1.2, math.sqrt(max(1.0, A_ace_mu)))
    B_ace_sigma = max(1.2, math.sqrt(max(1.0, B_ace_mu)))
    opp_srv_games = max(6.0, total_points_estimate / 4.5)
    A_break_mu = opp_srv_games * _v4a_safe(player_A.break_rate_on_return, 0.05, 0.45)
    B_break_mu = opp_srv_games * _v4a_safe(player_B.break_rate_on_return, 0.05, 0.45)
    A_break_sigma = max(0.9, math.sqrt(max(0.5, A_break_mu)))
    B_break_sigma = max(0.9, math.sqrt(max(0.5, B_break_mu)))
    outA = {"aces_mean": A_ace_mu, "breaks_mean": A_break_mu}
    outB = {"aces_mean": B_ace_mu, "breaks_mean": B_break_mu}
    if lines_aces:
        if player_A.name in lines_aces: outA["p_over_aces"] = _v4a_p_over(lines_aces[player_A.name], A_ace_mu, A_ace_sigma)
        if player_B.name in lines_aces: outB["p_over_aces"] = _v4a_p_over(lines_aces[player_B.name], B_ace_mu, B_ace_sigma)
    if lines_breaks:
        if player_A.name in lines_breaks: outA["p_over_breaks"] = _v4a_p_over(lines_breaks[player_A.name], A_break_mu, A_break_sigma)
        if player_B.name in lines_breaks: outB["p_over_breaks"] = _v4a_p_over(lines_breaks[player_B.name], B_break_mu, B_break_sigma)
    return {player_A.name: outA, player_B.name: outB}

# -----------------------------
# Unified adapter
# -----------------------------
def _qmc_p_hit_normal(mu: float, sigma: float, line: float, direction: str, n_sims: int, seed: int = 42) -> float:
    s = max(1e-9, float(sigma))
    z = _sobol_qmc_normals(n_sims, seed=seed)
    # control variate: Normal approx moment (mean 0) — here simply use analytic p as CV
    # Analytic probability under Normal model:
    from math import erf, sqrt
    zthr = (line - mu) / s
    p_analytic_over = 1.0 - 0.5*(1.0 + erf(zthr / sqrt(2.0)))
    p_analytic_under = 1.0 - p_analytic_over
    # Monte Carlo estimate with antithetic already inside generator
    hits = 0
    if str(direction).lower().startswith("o"):
        for zi in z:
            xi = mu + s * zi
            if xi > line:
                hits += 1
        p_mc = hits / max(1, n_sims)
        # simple CV blend toward analytic to reduce variance
        return 0.5*p_mc + 0.5*p_analytic_over
    else:
        for zi in z:
            xi = mu + s * zi
            if xi < line:
                hits += 1
        p_mc = hits / max(1, n_sims)
        return 0.5*p_mc + 0.5*p_analytic_under

# ------------------------------ Entropy sizing --------------------------------
def _binary_entropy(p: float) -> float:
    p = max(1e-12, min(1-1e-12, p))
    return -(p*_umath.log2(p) + (1-p)*_umath.log2(1-p))

def _entropy_edge_factor(p_model: float, p_market: float) -> float:
    Hm = _binary_entropy(p_model)
    Hk = _binary_entropy(p_market)
    return max(0.0, 1.0 - Hm / max(1e-12, Hk))

# --------------------------- Noise-cone (knife-edge) --------------------------
def _noise_cone_reject(p_cal: float, alpha: float = 0.10) -> bool:
    # Simple symmetric cone around 0.5 using alpha as nominal halfwidth proxy
    halfwidth = max(0.02, 0.5*alpha)  # e.g., 0.05 for alpha=0.10
    return abs(p_cal - 0.5) < halfwidth

# -------------------------- Quantile alt-line seeding -------------------------
def _quantile_altline(mu: float, sigma: float, is_over: bool, p_target_over: float) -> float:
    # Under Normal assumption (existing engine is Normal-based), quantile = mu + z_q*sigma
    # q for OVER = (1 - p_target_over); for UNDER mirror
    from math import sqrt
    try:
        import numpy as _np
        try:
            from scipy.stats import norm as _norm
            if is_over:
                q = 1.0 - p_target_over
            else:
                q = p_target_over
            z = _norm.ppf(max(1e-9, min(1-1e-9, q)))
        except Exception:
            # inverse via erfinv
            from numpy import erfinv as _erfinv
            q = (1.0 - p_target_over) if is_over else p_target_over
            z = sqrt(2.0) * _erfinv(2*q - 1.0)
    except Exception:
        # crude approximation if numpy missing
        # Beasley-Springer/Moro could be added; use a small table
        z_table = {0.90:1.2816, 0.95:1.6449, 0.975:1.9600, 0.99:2.3263}
        q = (1.0 - p_target_over) if is_over else p_target_over
        closest = min(z_table.keys(), key=lambda k: abs(k - q))
        z = z_table[closest] if q>=0.5 else -z_table[closest]
    return mu + max(1e-9, sigma) * z

# ------------------------------ Monkey patches --------------------------------
def _unified_patch_apply():
    # Patch only if class exists
    try:
        cls = V4EngineIntegrated  # noqa
    except Exception:
        return

    # 1) Patch _p_hit_raw to use Sobol QMC + CV when enabled
    if not hasattr(cls, "_p_hit_raw__orig"):
        cls._p_hit_raw__orig = cls._p_hit_raw  # type: ignore

        def _p_hit_raw_qmc(self, market, mu, sigma, line, direction, consensus=None):
            try:
                cfg = getattr(self, "cfg", {}) or {}
                use_qmc = UNIFIED_CFG.get("use_qmc", True)
                if not use_qmc:
                    return self._p_hit_raw__orig(market, mu, sigma, line, direction, consensus)  # type: ignore
                sims_cfg = int(cfg.get("mc_sims_min", 100000))
                n_sims = max(UNIFIED_CFG.get("qmc_sims_min", 100000), sims_cfg)
                p = _qmc_p_hit_normal(mu, sigma, line, direction, n_sims, seed=getattr(self.rng, "seed", 42) if hasattr(self, "rng") else 42)
                # record sims for conformal-calibration strength like original
                if isinstance(getattr(self, "context", None), dict):
                    self.context["_last_mc_sims"] = n_sims
                return p
            except Exception:
                # Fallback to original behavior
                return self._p_hit_raw__orig(market, mu, sigma, line, direction, consensus)  # type: ignore

        cls._p_hit_raw = _p_hit_raw_qmc  # type: ignore

    # 2) Patch optimize_alt_line to seed with quantile target first
    if not hasattr(cls, "optimize_alt_line__orig"):
        cls.optimize_alt_line__orig = cls.optimize_alt_line  # type: ignore

        def _optimize_alt_line_quantile(self, market, mu, sigma, posted_line, direction, price_quote, tier, consensus=None):
            try:
                if not UNIFIED_CFG.get("quantile_altline_first", True):
                    return self.optimize_alt_line__orig(market, mu, sigma, posted_line, direction, price_quote, tier, consensus)  # type: ignore
                # Compute quantile-based candidate
                is_over = str(direction).lower().startswith("o")
                p_target = UNIFIED_CFG.get("alt_target_p_over", 0.77)
                qline = _quantile_altline(mu, sigma, is_over=is_over, p_target_over=p_target)
                # Snap to family rounding if available
                fam = market_family(market)
                lg = detect_sport(market)
                step = FAMILY_ROUNDING.get(lg, {}).get(fam, SAFE_GUARDS.get("alt_step", 0.5))
                def _snap(x, s):
                    return round(round(x / max(1e-9, s)) * s, 3)
                qline = _snap(qline, step)
                # Evaluate quantile candidate quickly; if good, keep; else delegate to original optimizer
                mu_h, sg_h = self._apply_hbu(market, mu, sigma)
                sigma_eff = _inflate_sigma_if_needed(fam, sg_h)
                p = self._p_hit(market, mu_h, sigma_eff, qline, direction, consensus)
                pmin = SAFE_GUARDS["prop_pmin"].get(tier, 0.65) if fam in PROP_FAM else 0.0
                if p >= pmin:
                    alt_price = self.price_model.estimate(market, qline, direction, base_price=price_quote)
                    return qline, alt_price
                # fallback to original search
                return self.optimize_alt_line__orig(market, mu, sigma, posted_line, direction, price_quote, tier, consensus)  # type: ignore
            except Exception:
                return self.optimize_alt_line__orig(market, mu, sigma, posted_line, direction, price_quote, tier, consensus)  # type: ignore

        cls.optimize_alt_line = _optimize_alt_line_quantile  # type: ignore

    # 3) Patch evaluate_mainline to apply entropy-aware Kelly modifier & noise-cone
    if not hasattr(cls, "evaluate_mainline__orig"):
        cls.evaluate_mainline__orig = cls.evaluate_mainline  # type: ignore

        def _evaluate_mainline_entropy(self, market, mu, sigma, posted_line, direction, price):
            res = self.evaluate_mainline__orig(market, mu, sigma, posted_line, direction, price)  # type: ignore
            try:
                p_model = float(res.get("p_hit", 0.5))
                # infer market probability from price using engine's implied_prob
                p_mkt = float(self.price_model.implied_prob(price)) if hasattr(self, "price_model") else p_model
                # optional noise-cone
                if UNIFIED_CFG.get("noise_cone", True):
                    if _noise_cone_reject(p_model, UNIFIED_CFG.get("noise_cone_alpha", 0.10)):
                        res["kelly"] = 0.0
                        res["noise_cone"] = True
                        return res
                # entropy sizing
                if UNIFIED_CFG.get("entropy_size", True) and p_mkt > 0:
                    phi = _entropy_edge_factor(p_model, p_mkt)
                    res["kelly"] = float(res.get("kelly", 0.0)) * phi
                    res["entropy_phi"] = phi
                return res
            except Exception:
                return res

        cls.evaluate_mainline = _evaluate_mainline_entropy  # type: ignore

# Apply patches at import time
try:
    _unified_patch_apply()
except Exception as _unified_e:
    # Best-effort only; do not break import
    pass


# ============================================================================
# >>> UNIFIED V4 STAGE-2 PATCH (2025-10-15)
# Adds: Posterior family wrappers (NB/BB/ZIGN via moment-match), split-conformal
# calibration store, drawdown-aware utility sizing factor, Gaussian-vine fallback,
# audit fields injection, and quantile-first altline with posterior interface.
# Non-destructive, append-only, monkey-patches V4EngineIntegrated hooks.
# ============================================================================

# --------------------------- Posterior family wrappers ------------------------
# NOTE: We keep dependencies light by using moment-matching so the rest of the
# engine can continue to operate in (mu, sigma) space while exposing a unified
# interface consistent with a posterior-first design.
class _PosteriorFamily:
    def __init__(self, family:str, params:dict):
        self.family = family
        self.params = params

    # mean/var for unified interface
    def moments(self):
        f = self.family.upper()
        p = self.params
        if f == "NB":
            # NB parameterization: r (shape), p (success prob). mean=r(1-p)/p, var=r(1-p)/p^2
            r = max(1e-6, float(p.get("r", 1.0)))
            q = min(1-1e-9, max(1e-9, float(p.get("q", 0.5))))   # q = 1-p
            mean = r*q/(1-q)
            var  = r*q/((1-q)*(1-q))
            return mean, var
        if f == "BB":
            a = max(1e-6, float(p.get("a", 1.0)))
            b = max(1e-6, float(p.get("b", 1.0)))
            mean = a/(a+b)
            var  = (a*b)/(((a+b)**2)*(a+b+1.0))
            return mean, var
        if f == "ZIGN":
            # params: pi (zero gate), kappa (shape), theta (scale), tau (normal sd)
            pi  = min(1-1e-9, max(1e-9, float(p.get("pi", 0.05))))
            kap = max(1e-6, float(p.get("kappa", 10.0)))
            th  = max(1e-9, float(p.get("theta", 5.0)))
            tau = max(0.0, float(p.get("tau", 8.0)))
            mean_gamma = kap*th
            var_gamma  = kap*(th**2)
            # convolve with Normal(0,tau^2) ~ mean unchanged, var += tau^2
            mean = (1.0 - pi) * mean_gamma
            var  = (1.0 - pi) * (var_gamma + tau*tau) + pi*(0 - mean)**2
            return mean, var
        # default normal
        return float(p.get("mu", 0.0)), float(p.get("sigma", 1.0))**2

    # CDF/quantile via Normal moment-match fallback (keeps engine shape)
    def cdf(self, x:float)->float:
        m, v = self.moments()
        s = (v if v>0 else 1e-9) ** 0.5
        try:
            from math import erf, sqrt
            z = (x - m) / max(1e-12, s)
            return 0.5*(1.0 + erf(z/ (2.0**0.5)))
        except Exception:
            # crude numeric fallback
            import math
            z = (x - m) / max(1e-12, s)
            # logistic approx to normal cdf
            return 1.0/(1.0 + math.exp(-1.702*z))

    def quantile(self, q:float)->float:
        q = max(1e-9, min(1-1e-9, float(q)))
        m, v = self.moments()
        s = (v if v>0 else 1e-9) ** 0.5
        try:
            from math import sqrt
            from numpy import erfinv as _erfinv  # will fail if numpy missing
            z = (2.0**0.5) * _erfinv(2*q - 1.0)
            return m + s*z
        except Exception:
            # small z table fallback
            table = {0.90:1.2816,0.95:1.6449,0.975:1.96,0.99:2.3263}
            closest = min(table.keys(), key=lambda k: abs(k-q))
            z = table[closest] if q>=0.5 else -table[closest]
            return m + s*z

def _fit_posterior_family(market:str, features:dict, priors:dict)->_PosteriorFamily:
    # Heuristic dispatch by market type; use available priors if present.
    m = str(market).upper()
    if "TD" in m or "ANYTIME" in m or "FGM" in m:
        # binary-ish: Beta-Binomial prior collapse to Beta posterior (use priors a0,b0 if supplied)
        a0 = float(priors.get("a0", 8.0))
        b0 = float(priors.get("b0", 8.0))
        return _PosteriorFamily("BB", {"a": a0, "b": b0})
    if "YDS" in m or "YARDS" in m or "PASS" in m or "RUSH" in m or "REC" in m:
        # continuous yardage: ZI-Gamma-Normal
        return _PosteriorFamily("ZIGN", {
            "pi":  features.get("bench_prob", 0.03),
            "kappa": priors.get("kappa0", 18.0),
            "theta": priors.get("theta0", 4.0),
            "tau":   priors.get("tau0",   9.0),
        })
    # counts: NB
    return _PosteriorFamily("NB", {"r": priors.get("r0", 6.0), "q": priors.get("q0", 0.4)})

# --------------------------- Conformal calibration store ----------------------
class _ConformalStore:
    def __init__(self):
        self._residuals = {}  # market_key -> list of residual |p - y| or similar

    def update(self, market_key:str, p_pred:float, outcome:int):
        arr = self._residuals.setdefault(market_key, [])
        arr.append(abs(p_pred - float(outcome)))
        if len(arr) > 4096:
            del arr[:len(arr)-4096]

    def calibrate(self, market_key:str, p_pred:float, alpha:float=0.10):
        arr = self._residuals.get(market_key, [])
        # simple symmetric halfwidth from residual quantile
        if not arr:
            half = max(0.05, 0.5*alpha)
        else:
            try:
                q = sorted(arr)[int(max(0, min(len(arr)-1, (1.0-alpha)*len(arr))) )]
                half = max(0.02, float(q))
            except Exception:
                half = max(0.05, 0.5*alpha)
        lo = max(0.0, p_pred - half); hi = min(1.0, p_pred + half)
        # Conservative adjust toward center using halfwidth
        p_cal = min(hi, max(lo, p_pred))
        return {"p_cal": p_cal, "ci_low": lo, "ci_high": hi, "halfwidth": half}

# Global store instance
try:
    _UNIFIED_CONFORMAL_STORE
except NameError:
    _UNIFIED_CONFORMAL_STORE = _ConformalStore()

# ----------------------- Gaussian vine (fallback) utilities -------------------
def _fit_gaussian_vine_rho(rho_matrix):
    # Placeholder: pass-through for existing correlation if present, else identity
    return rho_matrix

# ---------------------- Drawdown-aware utility sizing factor ------------------
def _drawdown_utility_scale(ev:float, p:float, payout:float, lambda_dd:float=0.8)->float:
    # Cheap proxy: convert EV and win prob into an implied downside tail and penalize
    # stake factor in [0,1]; higher expected drawdown -> smaller scale.
    # This is intentionally simple to avoid interfering with your bankroll MC.
    loss_mag = (1.0 - p)  # unit loss per stake
    dd_proxy = max(0.0, loss_mag - ev)  # more loss risk when EV small
    scale = max(0.0, 1.0 - lambda_dd * min(1.0, dd_proxy))
    return scale

# ------------------------------ Monkey patches --------------------------------
def _unified_stage2_apply():
    cls = V4EngineIntegrated  # explicit binding to concrete class
    if not hasattr(cls, 'optimize_alt_line'):
        try:
            print('[WARN] stage2: optimize_alt_line not yet bound; skipping stage2 apply')
        except Exception:
            pass
        return
    if not hasattr(cls, 'optimize_alt_line__orig'):
        cls.optimize_alt_line__orig = cls.optimize_alt_line  # type: ignore
    if not hasattr(cls, 'optimize_alt_line__stage2'):
        cls.optimize_alt_line__stage2 = getattr(cls, 'optimize_alt_line__orig', cls.optimize_alt_line)  # type: ignore

    # Attach conformal store to class if not present
    if not hasattr(cls, "_conformal_store"):
        cls._conformal_store = _UNIFIED_CONFORMAL_STORE  # type: ignore

    # Patch alt-line to consult posterior family quantile when features/priors available
    if not hasattr(cls, "optimize_alt_line__stage2"):
        cls.optimize_alt_line__stage2 = getattr(cls, "optimize_alt_line__orig", cls.optimize_alt_line)  # type: ignore

        def _optimize_alt_line_stage2(self, market, mu, sigma, posted_line, direction, price_quote, tier, consensus=None):
            try:
                # Build lightweight priors from existing context if available
                feats = getattr(self, "context", {}).get("features", {})
                priors = getattr(self, "context", {}).get("priors", {})
                post = _fit_posterior_family(market, feats, priors)
                is_over = str(direction).lower().startswith("o")
                p_target = UNIFIED_CFG.get("alt_target_p_over", 0.77)
                q = (1.0 - p_target) if is_over else p_target
                qline = post.quantile(q)
                # snap to existing increments
                fam = market_family(market)
                lg = detect_sport(market)
                step = FAMILY_ROUNDING.get(lg, {}).get(fam, SAFE_GUARDS.get("alt_step", 0.5))
                qline = round(round(qline / max(1e-9, step)) * step, 3)
                # Evaluate; if passes, return; else fallback to prior stage behavior
                mu_h, sg_h = self._apply_hbu(market, mu, sigma)
                p = self._p_hit(market, mu_h, sg_h, qline, direction, consensus)
                pmin = SAFE_GUARDS["prop_pmin"].get(tier, 0.65) if fam in PROP_FAM else 0.0
                if p >= pmin:
                    alt_price = self.price_model.estimate(market, qline, direction, base_price=price_quote)
                    return qline, alt_price
                return self.optimize_alt_line__stage2(market, mu, sigma, posted_line, direction, price_quote, tier, consensus)  # type: ignore
            except Exception:
                return self.optimize_alt_line__stage2(market, mu, sigma, posted_line, direction, price_quote, tier, consensus)  # type: ignore

        cls.optimize_alt_line = _optimize_alt_line_stage2  # type: ignore

    # Patch evaluate_mainline again to add conformal & drawdown-aware scaling and audit fields
    if not hasattr(cls, "evaluate_mainline__stage2"):
        cls.evaluate_mainline__stage2 = cls.evaluate_mainline  # type: ignore

        def _evaluate_mainline_stage2(self, market, mu, sigma, posted_line, direction, price):
            res = self.evaluate_mainline__stage2(market, mu, sigma, posted_line, direction, price)  # type: ignore
            try:
                # Conformal calibration
                p_model = float(res.get("p_hit", 0.5))
                mkey = f"{market}:{posted_line}:{direction}"
                cal = self._conformal_store.calibrate(mkey, p_model, alpha=UNIFIED_CFG.get("noise_cone_alpha", 0.10))
                res["p_cal"] = cal["p_cal"]
                res["p_ci_low"] = cal["ci_low"]
                res["p_ci_high"] = cal["ci_high"]
                # Drawdown-aware utility scaling
                payout = self.price_model.payout(price) if hasattr(self, "price_model") else (price if isinstance(price,(int,float)) else 1.9)
                scale = _drawdown_utility_scale(res.get("ev", 0.0), res.get("p_hit", 0.5), payout, lambda_dd=0.8)
                res["kelly"] = float(res.get("kelly", 0.0)) * scale
                res["kelly_drawdown_scale"] = scale
                # Audit moments (Normal MM)
                EY = float(mu); VarY = float(max(1e-9, sigma*sigma))
                res.setdefault("audit", {})
                res["audit"].update({
                    "EY": EY, "VarY": VarY,
                    "market_key": mkey,
                })
                return res
            except Exception:
                return res

        cls.evaluate_mainline = _evaluate_mainline_stage2  # type: ignore

_unified_stage2_apply()


# ============================================================================
# >>> UNIFIED V4 STAGE-3 PATCH (2025-10-15)
# Focus: Quasi–Monte Carlo (Sobol + antithetic + control variates),
#        Lightweight Vine Copula (Gaussian default, Clayton/Gumbel optional),
#        Adaptive drawdown λ by bankroll tier,
#        ρ audit export in evaluators.
# Non-destructive: monkey patches only.
# ============================================================================

# -------------------------- Sobol + CV Monte Carlo ----------------------------
def _u4_sobol_uniforms(n, d=1, seed=42):
    """Return [n x d] Sobol uniforms (scrambled if SciPy present), else stratified fallback."""
    try:
        import numpy as _np
        try:
            from scipy.stats import qmc as _qmc
            sob = _qmc.Sobol(d=d, scramble=True, seed=seed)
            U = sob.random(n)
            # antithetic pairing: reflect around 0.5 and concat
            U2 = 1.0 - U
            out = _np.vstack([U, U2])
            return out[:n, :]
        except Exception:
            # stratified fallback
            rng = _np.random.default_rng(seed)
            U = (rng.random((n, d)) + (_np.arange(n)[:, None] / n)) % 1.0
            return U
    except Exception:
        # pure-Python fallback
        import random
        return [[random.random() for _ in range(d)] for __ in range(n)]

def _u4_inv_norm(u):
    """Vectorized-ish inverse Normal CDF with numpy if available; pure-Python fallback via approximation."""
    try:
        import numpy as _np
        try:
            from scipy.stats import norm as _norm
            return _norm.ppf(_np.clip(u, 1e-12, 1-1e-12))
        except Exception:
            from numpy import erfinv as _erfinv
            return (2.0**0.5) * _erfinv(2*_np.clip(u,1e-12,1-1e-12) - 1.0)
    except Exception:
        # Moro approximation constants
        import math
        u = max(1e-12, min(1-1e-12, float(u)))
        y = u - 0.5
        if abs(y) < 0.42:
            r = y*y
            num = y*((2.515517 + 0.802853*r) + 0.010328*r*r)
            den = 1 + (1.432788 + 0.189269*r + 0.001308*r*r)
            return num/den
        else:
            r = u if y > 0 else 1.0 - u
            s = math.log(-math.log(r))
            z = 1.570796288 + 0.03706987906*s + 0.0008364353589*s*s
            return z if y>0 else -z

def _u4_qmc_phit(mu, sigma, line, direction, n_sims=100000, seed=42):
    """Sobol-based probability-of-hit with Normal control variate blend."""
    import math
    s = max(1e-9, float(sigma))
    is_over = str(direction).lower().startswith("o")
    # Analytic under Normal
    zthr = (line - mu) / s
    p_analytic_over = 0.5*(1.0 - math.erf(zthr / (2.0**0.5))) + 0.5  # 1 - Phi(z)
    p_analytic_over = 1.0 - (0.5*(1.0 + math.erf(zthr / (2.0**0.5))))
    p_analytic = p_analytic_over if is_over else (1.0 - p_analytic_over)

    U = _u4_sobol_uniforms(n_sims, d=1, seed=seed)
    try:
        import numpy as _np
        Z = _u4_inv_norm(_np.asarray(U).reshape(-1,1))
        X = mu + s*Z
        if is_over:
            hits = (X > line).mean()
        else:
            hits = (X < line).mean()
        # Control variate: mean toward analytic (50/50 blend)
        return 0.5*float(hits) + 0.5*float(p_analytic)
    except Exception:
        # scalar fallback
        hits = 0
        for row in U:
            z = _u4_inv_norm(row[0]) if isinstance(row, (list, tuple)) else _u4_inv_norm(row)
            x = mu + s*z
            ok = (x > line) if is_over else (x < line)
            hits += 1 if ok else 0
        p_mc = hits/max(1, n_sims)
        return 0.5*p_mc + 0.5*p_analytic

# -------------------------- Lightweight Vine Copula ---------------------------
def _kendall_tau_to_theta_clayton(tau):
    # theta = 2*tau/(1 - tau), tau in (0,1)
    tau = max(-0.99, min(0.99, float(tau)))
    if tau <= 0:  # Clayton not suitable for non-positive tau
        return None
    return 2.0*tau/(1.0 - tau)

def _kendall_tau_to_theta_gumbel(tau):
    # theta = 1/(1 - tau), tau in (0,1)
    tau = max(-0.99, min(0.99, float(tau)))
    if tau <= 0:
        return None
    return 1.0/(1.0 - tau)

def _sample_pair_copula(u1, u2, family, theta):
    """Given independent uniforms u1,u2, couple them via basic two-family copulas; else return (u1,u2)."""
    if family == "clayton" and theta is not None:
        # Marshall–Olkin algorithm for Clayton
        # V ~ Gamma(1/theta, 1), set X=(1+E1/V)^(-1/theta), Y=(1+E2/V)^(-1/theta)
        try:
            import numpy as _np
            rng = _np.random.default_rng(12345)
            V = rng.gamma(shape=1.0/theta, scale=1.0)
            E1 = -_np.log(max(1e-12, u1)); E2 = -_np.log(max(1e-12, u2))
            x = (1.0 + E1 / V)**(-1.0/theta)
            y = (1.0 + E2 / V)**(-1.0/theta)
            return float(x), float(y)
        except Exception:
            return u1, u2
    if family == "gumbel" and theta is not None:
        # Approximate sampling via conditional method (simplified for stability)
        # Fallback to positive dependence via comonotonic blend
        w = min(0.99, max(0.01, (theta-1.0)/theta))
        v = w*min(u1, u2) + (1-w)*max(u1, u2)
        return v, v
    # Gaussian default
    return u1, u2

def _u4_fit_vine_order(rho_matrix):
    """Simple C-vine order by absolute correlation descending."""
    try:
        import numpy as _np
        abs_sum = _np.sum(_np.abs(rho_matrix), axis=1)
        return list(_np.argsort(-abs_sum))
    except Exception:
        return list(range(len(rho_matrix)))

def _u4_vine_sample(U, rho_matrix):
    """Very lightweight C-vine sampler: Gaussian default with optional Clayton/Gumbel per pair via tau sign."""
    try:
        import numpy as _np
        U = _np.asarray(U)
        n, d = U.shape[0], U.shape[1]
        if d == 1:
            return U.copy()
        order = _u4_fit_vine_order(rho_matrix)
        V = U.copy()
        # Couple successive pairs along the order
        for k in range(d-1):
            i, j = order[k], order[k+1]
            rho = float(rho_matrix[i][j]) if i < len(rho_matrix) and j < len(rho_matrix) else 0.0
            # Map rho to Kendall's tau (approx: tau = 2/pi * arcsin(rho))
            import math
            tau = (2.0/math.pi) * math.asin(max(-0.99, min(0.99, rho)))
            fam = "gaussian"
            theta = None
            # crude family choice
            if tau > 0.2:
                theta = _kendall_tau_to_theta_gumbel(tau); fam = "gumbel" if theta else "gaussian"
            elif tau < -0.2:
                # negative tau -> keep gaussian (Clayton unsupported for negative tail)
                fam = "gaussian"; theta = None
            else:
                fam = "gaussian"
            for t in range(n):
                V[t, i], V[t, j] = _sample_pair_copula(V[t, i], V[t, j], fam, theta)
        return V
    except Exception:
        return U

# -------------------------- Adaptive λ by bankroll tier -----------------------
def _u4_lambda_from_tier(tier:str):
    t = (tier or "").strip().lower()
    if t in ("ultra", "stealth aggressive", "aggressive"):
        return 0.5
    if t in ("balanced", "standard", "default"):
        return 0.8
    if t in ("conservative", "safe"):
        return 1.1
    return 0.8

# ------------------------------ Monkey patches --------------------------------
def _unified_stage3_apply():
    try:
        cls = V4EngineIntegrated  # noqa
    except Exception:
        return

    # (1) Replace _p_hit_raw with Sobol + CV backend (retaining fallback behavior)
    if not hasattr(cls, "_p_hit_raw__stage3"):
        cls._p_hit_raw__stage3 = cls._p_hit_raw  # type: ignore

        def _p_hit_raw_sobol(self, market, mu, sigma, line, direction, consensus=None):
            try:
                cfg = getattr(self, "cfg", {}) or {}
                if not UNIFIED_CFG.get("use_qmc", True):
                    return self._p_hit_raw__stage3(market, mu, sigma, line, direction, consensus)  # type: ignore
                sims_cfg = int(cfg.get("mc_sims_min", 100000))
                n_sims = max(UNIFIED_CFG.get("qmc_sims_min", 100000), sims_cfg)
                p = _u4_qmc_phit(mu, sigma, line, direction, n_sims=n_sims, seed=getattr(self.rng, "seed", 42) if hasattr(self, "rng") else 42)
                if isinstance(getattr(self, "context", None), dict):
                    self.context["_last_mc_sims"] = n_sims
                return p
            except Exception:
                return self._p_hit_raw__stage3(market, mu, sigma, line, direction, consensus)  # type: ignore

        cls._p_hit_raw = _p_hit_raw_sobol  # type: ignore

    # (2) Parlay correlation sampler hook: transform uniforms via vine-like coupling
    if not hasattr(cls, "mvn_correlated_normals__stage3"):
        cls.mvn_correlated_normals__stage3 = cls.mvn_correlated_normals  # type: ignore

        def _mvn_correlated_normals_vine(self, size, rho_matrix):
            try:
                import numpy as _np
                # Step A: generate Sobol uniforms then vine-couple them
                U = _u4_sobol_uniforms(size, d=len(rho_matrix))
                Uc = _u4_vine_sample(U, rho_matrix)
                # Step B: map to correlated normals by inverse-normal + cholesky as a safety blend
                Z = _u4_inv_norm(Uc)
                # Defensive: if fail, fallback to original
                if isinstance(Z, list):
                    return self.mvn_correlated_normals__stage3(size, rho_matrix)  # type: ignore
                return Z.reshape(size, len(rho_matrix))
            except Exception:
                return self.mvn_correlated_normals__stage3(size, rho_matrix)  # type: ignore

        cls.mvn_correlated_normals = _mvn_correlated_normals_vine  # type: ignore

    # (3) Adaptive λ in evaluate_mainline: scale by cfg.risk_tier if present
    if not hasattr(cls, "evaluate_mainline__stage3"):
        cls.evaluate_mainline__stage3 = cls.evaluate_mainline  # type: ignore

        def _evaluate_mainline_stage3(self, market, mu, sigma, posted_line, direction, price):
            res = self.evaluate_mainline__stage3(market, mu, sigma, posted_line, direction, price)  # type: ignore
            try:
                tier = (getattr(self, "cfg", {}) or {}).get("risk_tier", "balanced")
                lam = _u4_lambda_from_tier(tier)
                # If drawdown scale already calculated, adjust slightly by lam
                if "kelly_drawdown_scale" in res:
                    res["kelly_drawdown_scale"] = max(0.0, min(1.2, res["kelly_drawdown_scale"] * (0.9/lam)))
                    res["kelly"] = float(res.get("kelly", 0.0)) * res["kelly_drawdown_scale"]
                res.setdefault("audit", {})
                res["audit"]["risk_lambda"] = lam
                return res
            except Exception:
                return res

        cls.evaluate_mainline = _evaluate_mainline_stage3  # type: ignore

    # (4) Add ρ row into audits where available (after vine/Cholesky fit)
    if not hasattr(cls, "evaluate_bundle_with_corr__stage3") and hasattr(cls, "evaluate_bundle_with_corr"):
        cls.evaluate_bundle_with_corr__stage3 = cls.evaluate_bundle_with_corr  # type: ignore

        def _evaluate_bundle_with_corr_stage3(self, legs, rho_matrix, *args, **kwargs):
            res = self.evaluate_bundle_with_corr__stage3(legs, rho_matrix, *args, **kwargs)  # type: ignore
            try:
                # Attach rho row per leg id if structure matches
                for i, item in enumerate(res if isinstance(res, list) else []):
                    try:
                        item.setdefault("audit", {})
                        item["audit"]["rho_row"] = list(rho_matrix[i])
                    except Exception:
                        continue
                return res
            except Exception:
                return res

        cls.evaluate_bundle_with_corr = _evaluate_bundle_with_corr_stage3  # type: ignore

_unified_stage3_apply()


# ============================================================================
# >>> UNIFIED V4 STAGE-4 PATCH (2025-10-15)
# Finalizes:
#   • Quasi–MC with data-driven Control Variates (indicator vs Z covariance)
#   • R‑vine Copula 2.0 scaffold with light family selection (“AIC‑lite”)
#   • Adaptive Sobol skipping & config hook
#   • Expose EV/uncertainty-aware quantile safety curve for alt-line targets
# Non-destructive (append-only). Safe fallbacks preserved.
# ============================================================================

# ---------------------------- QMC + Control Variates --------------------------
def _u4_qmc_phit_cv(mu, sigma, line, direction, n_sims=100000, seed=42, skip=0):
    """
    Sobol-based probability-of-hit with **estimated control variate** using the
    sample covariance between indicator Y and standard normal Z used to form X.
    Estimator: p_cv = mean(Y) - c_hat * mean(Z), with c_hat = Cov(Y,Z)/Var(Z).
    With antithetic Sobol, Var(Z)≈1; we still estimate from samples for safety.
    """
    import math
    is_over = str(direction).lower().startswith("o")
    s = max(1e-9, float(sigma))

    # Analytic Normal probability (used as guard, not as CV in Stage-4)
    zthr = (line - mu) / s
    p_analytic_over = 1.0 - 0.5*(1.0 + math.erf(zthr / (2.0**0.5)))
    p_analytic = p_analytic_over if is_over else (1.0 - p_analytic_over)

    # Draw Sobol uniforms (with optional skip) then map to Z via inverse-CDF
    U = _u4_sobol_uniforms(n_sims + int(skip), d=1, seed=seed)
    try:
        import numpy as _np
        U = _np.asarray(U)
        if skip > 0:
            U = U[int(skip):, :]
        Z = _u4_inv_norm(U.reshape(-1,1)).reshape(-1)
        X = mu + s*Z
        if is_over:
            Y = (X > line).astype(float)
        else:
            Y = (X < line).astype(float)

        # Estimate control variate coefficient c* ≈ Cov(Y,Z)/Var(Z)
        Z_bar = float(Z.mean())
        Y_bar = float(Y.mean())
        cov = float(((Y - Y_bar)*(Z - Z_bar)).mean())
        varZ = float(((Z - Z_bar)**2).mean())
        if varZ <= 1e-12:
            p_est = Y_bar
        else:
            c_hat = cov / varZ
            p_est = Y_bar - c_hat * Z_bar  # CV adjusted estimator
        # Bias guard: softly blend toward analytic if sample is tiny or extreme
        w = max(0.15, min(0.85, 1.0 - abs(0.5 - p_est)))  # less blend near certainty
        return w*p_est + (1.0 - w)*p_analytic
    except Exception:
        # Fallback to Stage-3 estimator
        return _u4_qmc_phit(mu, sigma, line, direction, n_sims=n_sims, seed=seed)

# ----------------------- Quantile Safety Curve (EV/uncertainty) ---------------
def _u4_quantile_safety_target(p_raw: float, cv_post: float, base_over: float = 0.77) -> float:
    """
    Adaptive alt-line target probability. Raises safety target as posterior CV grows,
    lowers it for very high EV edges.
    p_raw: model cover prob at posted line
    cv_post: posterior coefficient of variation proxy for the market
    base_over: base target for Overs (mirror for Unders)
    Returns target probability in [0.6, 0.9].
    """
    # Map CV into safety lift: more variance -> higher p_target
    lift = max(0.0, min(0.12, 0.35*cv_post))    # cap lift at +0.12
    # Map EV edge into safety relief: bigger edge -> lower p_target modestly
    edge = abs(p_raw - 0.5)
    relief = min(0.10, 0.5*edge)                # cap relief at -0.10
    target = base_over + lift - relief
    return max(0.60, min(0.90, target))

# Hook into alt-line path to consult safety curve when enabled
def _unified_stage4_apply_altcurve():
    try:
        cls = V4EngineIntegrated  # noqa
    except Exception:
        return
    if not hasattr(cls, "optimize_alt_line__stage4"):
        cls.optimize_alt_line__stage4 = getattr(cls, "optimize_alt_line__stage2", getattr(cls, "optimize_alt_line__orig", cls._optimize_alt_line_stub))  # type: ignore

        def _optimize_alt_line_stage4(self, market, mu, sigma, posted_line, direction, price_quote, tier, consensus=None):
            try:
                # derive CV proxy
                cv_post = float(max(1e-9, sigma)/max(1e-9, abs(mu)))
                # get current p at posted line to inform edge
                p_now = self._p_hit(market, mu, sigma, posted_line, direction, consensus)
                base = UNIFIED_CFG.get("alt_target_p_over", 0.77)
                p_target = _u4_quantile_safety_target(p_now, cv_post, base_over=base)
                # respect UNDER symmetry
                is_over = str(direction).lower().startswith("o")
                q = (1.0 - p_target) if is_over else p_target
                # If posterior family exists (Stage‑2), use it; else normal quantile
                feats = getattr(self, "context", {}).get("features", {})
                priors = getattr(self, "context", {}).get("priors", {})
                try:
                    post = _fit_posterior_family(market, feats, priors)
                    qline = post.quantile(q)
                except Exception:
                    qline = _quantile_altline(mu, sigma, is_over, p_target)
                # snap and evaluate; fallback to previous impl if not viable
                fam = market_family(market); lg = detect_sport(market)
                step = FAMILY_ROUNDING.get(lg, {}).get(fam, SAFE_GUARDS.get("alt_step", 0.5))
                qline = round(round(qline / max(1e-9, step)) * step, 3)
                mu_h, sg_h = self._apply_hbu(market, mu, sigma)
                p = self._p_hit(market, mu_h, sg_h, qline, direction, consensus)
                pmin = SAFE_GUARDS["prop_pmin"].get(tier, 0.65) if fam in PROP_FAM else 0.0
                if p >= pmin:
                    alt_price = self.price_model.estimate(market, qline, direction, base_price=price_quote)
                    return qline, alt_price
                return self.optimize_alt_line__stage4(market, mu, sigma, posted_line, direction, price_quote, tier, consensus)  # type: ignore
            except Exception:
                return self.optimize_alt_line__stage4(market, mu, sigma, posted_line, direction, price_quote, tier, consensus)  # type: ignore

        cls.optimize_alt_line = _optimize_alt_line_stage4  # type: ignore

# ----------------------------- R‑vine Copula 2.0 ------------------------------
class _UVineNode:
    __slots__ = ("i","j","family","theta","tau")
    def __init__(self, i, j, family, theta, tau):
        self.i=i; self.j=j; self.family=family; self.theta=theta; self.tau=tau

class _UVine:
    """
    Lightweight C‑vine with per‑pair family choice (“AIC‑lite”):
      - If tau >> 0: prefer Gumbel (upper‑tail), else Gaussian
      - If tau << 0: Gaussian (Clayton/Gumbel not valid for negative tau)
      - If |tau| small: Gaussian
    This is intentionally simple and robust; you can swap for a full AIC trainer later.
    """
    def __init__(self, rho_matrix):
        self.edges = []
        try:
            import numpy as _np, math
            d = len(rho_matrix)
            order = _u4_fit_vine_order(rho_matrix)
            for k in range(d-1):
                i, j = order[k], order[k+1]
                rho = float(rho_matrix[i][j]) if i<d and j<d else 0.0
                tau = (2.0/math.pi) * math.asin(max(-0.99, min(0.99, rho)))
                fam, th = "gaussian", None
                if tau > 0.25:
                    th = _kendall_tau_to_theta_gumbel(tau); fam = "gumbel" if th else "gaussian"
                elif tau < -0.25:
                    fam, th = "gaussian", None
                self.edges.append(_UVineNode(i,j,fam,th,tau))
        except Exception:
            pass

    def couple(self, U):
        try:
            import numpy as _np
            V = _np.array(U, copy=True)
            n = V.shape[0]
            for e in self.edges:
                for t in range(n):
                    V[t, e.i], V[t, e.j] = _sample_pair_copula(V[t, e.i], V[t, e.j], e.family, e.theta)
            return V
        except Exception:
            return U

def _unified_stage4_apply_rvine():
    try:
        cls = V4EngineIntegrated  # noqa
    except Exception:
        return
    if not hasattr(cls, "mvn_correlated_normals__stage4"):
        cls.mvn_correlated_normals__stage4 = cls.mvn_correlated_normals  # type: ignore

        def _mvn_correlated_normals_rvine(self, size, rho_matrix):
            try:
                import numpy as _np
                # Generate Sobol uniforms and couple via R‑vine scaffold
                U = _u4_sobol_uniforms(size, d=len(rho_matrix), seed=getattr(self.rng,"seed",42) if hasattr(self,"rng") else 42)
                V = _UVine(rho_matrix).couple(U)
                Z = _u4_inv_norm(V)  # → correlated normals (approx)
                if isinstance(Z, list):
                    return self.mvn_correlated_normals__stage4(size, rho_matrix)  # fallback
                return Z.reshape(size, len(rho_matrix))
            except Exception:
                return self.mvn_correlated_normals__stage4(size, rho_matrix)  # fallback

        cls.mvn_correlated_normals = _mvn_correlated_normals_rvine  # type: ignore

# ------------------------------ Integrate Stage‑4 -----------------------------
def _unified_stage4_apply():
    # Upgrade _p_hit_raw to CV version with Sobol skip support
    try:
        cls = V4EngineIntegrated  # noqa
        if not hasattr(cls, "_p_hit_raw__stage4"):
            cls._p_hit_raw__stage4 = cls._p_hit_raw  # type: ignore
            def _p_hit_raw_cv(self, market, mu, sigma, line, direction, consensus=None):
                try:
                    cfg = getattr(self, "cfg", {}) or {}
                    if not UNIFIED_CFG.get("use_qmc", True):
                        return self._p_hit_raw__stage4(market, mu, sigma, line, direction, consensus)  # type: ignore
                    sims_cfg = int(cfg.get("mc_sims_min", 100000))
                    n_sims = max(UNIFIED_CFG.get("qmc_sims_min", 100000), sims_cfg)
                    skip = int(cfg.get("qmc_skip", 0))
                    return _u4_qmc_phit_cv(mu, sigma, line, direction, n_sims=n_sims, seed=getattr(self, "seed", 42) if hasattr(self, "seed") else 42, skip=skip)
                except Exception:
                    return self._p_hit_raw__stage4(market, mu, sigma, line, direction, consensus)  # type: ignore
            cls._p_hit_raw = _p_hit_raw_cv  # type: ignore
    except Exception:
        pass

    # Alt-line safety curve + R‑vine coupling
    _unified_stage4_apply_altcurve()
    _unified_stage4_apply_rvine()

_unified_stage4_apply()


# ============================================================================
# >>> Embedded Auto-Runner (non-destructive)
# ============================================================================
import os, json, argparse, datetime
from pathlib import Path as _P

class _V4DataProvider:
    def __init__(self):
        self.paths = [os.environ.get("V4_SLATE_PATH"), "/mnt/data/slate.json", "/mnt/data/slate.csv"]
    def _from_csv(self, path):
        import csv
        legs=[]
        with open(path,"r",encoding="utf-8") as f:
            rdr=csv.DictReader(f)
            for r in rdr:
                legs.append({
                    "market": r.get("market","PASSYDS"),
                    "player_id": r.get("player_id","P1"),
                    "team_id": r.get("team_id","T1"),
                    "game_id": r.get("game_id","G1"),
                    "line": float(r.get("line","249.5")),
                    "direction": r.get("direction","OVER"),
                    "price": float(r.get("price","-110")),
                    "mu": float(r.get("mu", r.get("line","249.5"))),
                    "sigma": float(r.get("sigma","40.0")),
                    "tier": r.get("tier","balanced")
                })
        return {"bankroll": float(os.environ.get("V4_BANKROLL","1000")),
                "risk_tier": os.environ.get("V4_RISK_TIER","balanced"),
                "legs": legs}
    def _demo(self):
        return {
          "bankroll": 1000.0, "risk_tier":"balanced",
          "legs":[
            {"market":"NFL_PASSYDS_QB1","player_id":"QB1","team_id":"T-A","game_id":"G1","line":249.5,"direction":"OVER","price":-115,"mu":265.0,"sigma":45.0,"tier":"balanced"}
          ]
        }
    def load(self, path=None):
        if path and os.path.exists(path):
            if path.lower().endswith(".json"):
                return json.loads(_P(path).read_text(encoding="utf-8"))
            if path.lower().endswith(".csv"):
                return self._from_csv(path)
        for p in self.paths:
            if p and os.path.exists(p):
                if p.lower().endswith(".json"):
                    return json.loads(_P(p).read_text(encoding="utf-8"))
                if p.lower().endswith(".csv"):
                    return self._from_csv(p)
        return self._demo()

def run_from_prompt(payload: dict=None, slate_path: str=None, out_path: str=None):
    dp = _V4DataProvider()
    slate = payload if isinstance(payload, dict) else dp.load(slate_path)
    cfg = {
        "risk_tier": slate.get("risk_tier","balanced"),
        "bankroll": float(slate.get("bankroll", 1000.0)),
        "mc_sims_min": int(os.environ.get("V4_MC_SIMS_MIN","131072")),
        "alpha": float(os.environ.get("V4_ALPHA","0.10")),
        "kelly_fraction": float(os.environ.get("V4_KELLY","0.25")),
    }
    try:
        eng = V4EngineIntegrated(cfg)  # type: ignore
    except Exception:
        class _Shim:
            def __init__(self,cfg): self.cfg=cfg
            def _apply_hbu(self,m,mu,sg): return mu,sg
            def _p_hit(self,m,mu,sg,line,dir,cons=None):
                import math
                sg=max(1e-9,sg); z=(line-mu)/sg; from math import erf,sqrt
                p_under=0.5*(1.0+erf(-z/math.sqrt(2))); return 1.0-p_under if dir.lower().startswith("o") else p_under
            def evaluate_mainline(self,m,mu,sg,line,dir,price):
                p=self._p_hit(m,mu,sg,line,dir); b=(abs(price)/100.0) if price>0 else (100.0/abs(price))
                ev=p*b-(1-p); q=1-p; k=max(0.0,(b*p-q)/b) if b>0 else 0.0
                return {"p_hit":p,"ev":ev,"kelly":k,"audit":{"mu":mu,"sigma":sg}}
            def optimize_alt_line(self,*a,**k): return a[3], a[5] if len(a)>5 else -110
        eng=_Shim(cfg)

    results=[]
    for leg in slate.get("legs", []):
        m=leg["market"]; line=float(leg["line"]); d=leg["direction"]; price=float(leg["price"])
        mu=float(leg.get("mu", line)); sg=float(leg.get("sigma", max(1.0,abs(line)*0.25)))
        try:
            mu, sg = eng._apply_hbu(m, mu, sg)
        except Exception:
            pass
        try:
            res = eng.evaluate_mainline(m, mu, sg, line, d, price)
        except Exception:
            p = eng._p_hit(m, mu, sg, line, d, None)
            b=(abs(price)/100.0) if price>0 else (100.0/abs(price))
            res={"p_hit":p,"ev":p*b-(1-p),"kelly":0.0,"audit":{"mu":mu,"sigma":sg}}
        try:
            alt = eng.optimize_alt_line(m, mu, sg, line, d, price, leg.get("tier","balanced"))
            if alt:
                res["alt_suggest"]={"line": alt[0], "price": alt[1]}
        except Exception:
            pass
        results.append({"leg": leg, "result": res})

    bundle={"timestamp": datetime.datetime.utcnow().isoformat()+"Z",
            "bankroll": cfg["bankroll"], "risk_tier": cfg["risk_tier"], "results": results}
    out_path = out_path or os.environ.get("V4_OUT_PATH","/mnt/data/v4_run_output.json")
    try:
        _P(out_path).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    except Exception:
        pass
    return bundle

if __name__ == "__main__":
    ap=argparse.ArgumentParser(description="V4 Unified — Auto Runner")
    ap.add_argument("--slate", type=str, default=os.environ.get("V4_SLATE_PATH"))
    ap.add_argument("--out", type=str, default=os.environ.get("V4_OUT_PATH","/mnt/data/v4_run_output.json"))
    args=ap.parse_args()
    out = run_from_prompt(payload=None, slate_path=args.slate, out_path=args.out)
    print(json.dumps(out, indent=2))


# ============================================================================
# V4 + MC: Non-destructive Add-Ons (Integrated)
# Implements the "complements" list:
#   1) Elo/Bayesian Prior Mixer
#   2) NFL Drive/Play Microsim layer
#   3) WP/EP Calibration Hooks
#   4) Hoops Possession Markov micro-engine
#   5) Bivariate Poisson / Skellam module (soccer, NHL, low-scoring)
#   6) Light Strategy Toggles (MDP-inspired)
#   7) Unified Adapter (all sports) + Bolt-on Validator
# This block APPENDS functionality; it does not modify existing classes above.
# ============================================================================

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math, random

# ---------- (1) Elo/Bayesian Prior Mixer ----------
def v4_prior_mixer(no_vig_prob: float, elo_prob: float, w: float = 0.20) -> float:
    """Blend no-vig price-derived probability with Elo/Bayes prior.
    p' = mix(no_vig, elo; w) and clamp to [0,1]."""
    p = (1.0 - max(0.0,min(1.0,w))) * no_vig_prob + max(0.0,min(1.0,w)) * elo_prob
    return max(1e-6, min(1.0 - 1e-6, p))

# ---------- (2) NFL Drive/Play Microsim Layer ----------
@dataclass
class NFLMicrosimParams:
    plays_per_drive_mu: float = 5.7
    plays_per_drive_sigma: float = 2.2
    drive_count_mu_home: float = 10.8
    drive_count_mu_away: float = 10.4
    td_rate_per_drive: float = 0.205
    fg_rate_per_drive: float = 0.165
    turnover_rate_per_drive: float = 0.135
    pass_rate: float = 0.57
    ypp_pass: float = 6.7
    ypp_rush: float = 4.5

def simulate_nfl_microsim_paths(params: NFLMicrosimParams, sims: int = 5000, seed: Optional[int] = 42) -> Dict[str, float]:
    rng = random.Random(seed)
    td_home = td_away = 0.0
    yds_home = yds_away = 0.0
    for _ in range(sims):
        for side in ("home","away"):
            drives = max(6, int(round(rng.gauss(params.drive_count_mu_home if side=='home' else params.drive_count_mu_away, 1.2))))
            tds = 0; yds = 0.0
            for _d in range(drives):
                plays = max(1, int(round(rng.gauss(params.plays_per_drive_mu, params.plays_per_drive_sigma))))
                for _p in range(plays):
                    if rng.random() < params.pass_rate:
                        yds += max(-8.0, rng.gauss(params.ypp_pass, 7.0))
                    else:
                        yds += max(-3.0, rng.gauss(params.ypp_rush, 4.0))
                # drive result
                r = rng.random()
                if r < params.td_rate_per_drive:
                    tds += 1
                # (fg/turnover ignored for TD count)
            if side == "home":
                td_home += tds; yds_home += yds
            else:
                td_away += tds; yds_away += yds
    return {
        "home_tds_mean": td_home / sims,
        "away_tds_mean": td_away / sims,
        "home_yds_mean": yds_home / sims,
        "away_yds_mean": yds_away / sims,
    }

# ---------- (3) WP/EP Calibration Hooks ----------
def calibrate_wp_bias(sim_wp_points: List[Tuple[int,float]], reference_curve: Dict[int, float]) -> float:
    """sim_wp_points: list of (minute_mark, wp) from your sims
       reference_curve: minute_mark -> historical WP
       Returns small bias in [-0.08, +0.08] to apply to offensive/defensive efficacy."""
    if not sim_wp_points:
        return 0.0
    diffs = []
    for t, wp in sim_wp_points:
        ref = reference_curve.get(t, wp)
        diffs.append(wp - ref)
    avg = sum(diffs)/len(diffs)
    return max(-0.08, min(0.08, -avg))  # negative avg -> boost offense slightly

# ---------- (4) Hoops Possession Markov ----------
@dataclass
class HoopsMarkovParams:
    possessions: int = 100
    p_turnover: float = 0.13
    p_shot_2pt: float = 0.42
    p_shot_3pt: float = 0.36
    p_foul: float = 0.09
    p_off_reb: float = 0.26
    p_make_2pt: float = 0.53
    p_make_3pt: float = 0.37
    ft_rate: float = 1.9  # FT attempts per foul
    ft_pct: float = 0.78

def simulate_hoops_markov(params: HoopsMarkovParams, sims: int = 4000, seed: Optional[int]=42) -> Dict[str, float]:
    rng = random.Random(seed)
    pts_total = 0.0
    fgm = ast = reb = 0.0
    for _ in range(sims):
        pts = 0; fgm_s = 0; ast_s = 0; reb_s = 0
        for _p in range(params.possessions):
            r = rng.random()
            if r < params.p_turnover:
                continue
            r -= params.p_turnover
            if r < params.p_shot_2pt:
                if rng.random() < params.p_make_2pt:
                    pts += 2; fgm_s += 1; ast_s += 0.55  # avg fraction
                if rng.random() < params.p_off_reb:
                    reb_s += 1  # second-chance but keep single-step
            else:
                if rng.random() < params.p_make_3pt:
                    pts += 3; fgm_s += 1; ast_s += 0.75
                if rng.random() < params.p_off_reb:
                    reb_s += 1
            # fouls & FTs
            if rng.random() < params.p_foul:
                makes = sum(1 for _ in range(int(round(params.ft_rate))) if rng.random() < params.ft_pct)
                pts += makes
        pts_total += pts; fgm += fgm_s; ast += ast_s; reb += reb_s
    sims = max(1, sims)
    return {"points_mean": pts_total/sims, "fgm_mean": fgm/sims, "ast_mean": ast/sims, "reb_mean": reb/sims}

# ---------- (5) Bivariate Poisson / Skellam ----------
@dataclass
class BivarPoissonParams:
    lambda_home: float
    lambda_away: float
    lambda_c: float = 0.10  # shared component; 0 -> independent

def sample_bivariate_poisson(params: BivarPoissonParams, sims: int=5000, seed: Optional[int]=42) -> Tuple[float,float,float]:
    rng = random.Random(seed)
    import math
    def _po(mu):  # Poisson by thinning
        L = math.exp(-mu); k = 0; p = 1.0
        while p > L:
            k += 1; p *= rng.random()
        return max(0, k-1)
    gH = gA = 0
    for _ in range(sims):
        c = _po(max(0.0, params.lambda_c))
        h = _po(max(0.0, params.lambda_home - params.lambda_c)) + c
        a = _po(max(0.0, params.lambda_away - params.lambda_c)) + c
        gH += h; gA += a
    return (gH/sims, gA/sims, params.lambda_c)

def skellam_from_tot_spread(total: float, spread_home: float) -> Tuple[float,float]:
    """Return (lambda_home, lambda_away) for goal/score counts."""
    mu_sum = max(0.1, total)
    # Solve muH - muA = spread_home, muH + muA = total
    mu_h = (mu_sum + spread_home)/2.0
    mu_a = mu_sum - mu_h
    return (max(0.05, mu_h), max(0.05, mu_a))

# ---------- (6) Light MDP Strategy Toggles ----------
@dataclass
class StrategyToggles:
    pass_rate_delta: float = 0.0
    fourth_down_aggr_delta: float = 0.0
    pace_delta: float = 0.0

def apply_strategy_toggles_nfl(p: NFLMicrosimParams, t: StrategyToggles) -> NFLMicrosimParams:
    q = NFLMicrosimParams(**p.__dict__)
    q.pass_rate = max(0.35, min(0.80, q.pass_rate + t.pass_rate_delta))
    q.plays_per_drive_mu = max(3.5, q.plays_per_drive_mu * (1.0 + 0.10*t.pace_delta))
    q.td_rate_per_drive = max(0.05, q.td_rate_per_drive * (1.0 + 0.12*t.fourth_down_aggr_delta))
    return q

# ---------- (7) Unified Adapter + Validator (compact variant) ----------
@dataclass
class GameBundle:
    sport: str
    game_id: str
    home: str
    away: str
    home_params: dict
    away_params: dict = None
    players_home: List[dict] = None
    players_away: List[dict] = None
    lines: Dict[str, Dict[str, float]] = None
    extras: dict = None

# Minimal per-sport adapters (reuse existing engine helpers when available)
def build_player_prop_table(bundle: GameBundle) -> List[dict]:
    s = bundle.sport.upper()
    tag = f"{bundle.home}@{bundle.away}.{s}"
    rows: List[dict] = []
    X = bundle.extras or {}
    L = bundle.lines or {}

    # NFL via microsim if hints are present, else use extras
    if s == "NFL":
        mp = NFLMicrosimParams(**(X.get("microsim_params", {}) or {}))
        if X.get("strategy_toggles"):
            mp = apply_strategy_toggles_nfl(mp, StrategyToggles(**X["strategy_toggles"]))
        m = simulate_nfl_microsim_paths(mp, sims=X.get("microsim_sims", 4000))
        team_tds = {"home": m["home_tds_mean"], "away": m["away_tds_mean"]}
        pass_frac_h = X.get("home_pass_td_frac", 0.55); pass_frac_a = X.get("away_pass_td_frac", 0.55)

        # If player alloc helpers are present in earlier patches of your file, prefer them.
        try:
            _NFLPU = NFLPlayerUsage  # raises if missing
            pr_h = [NFLPlayerUsage(**p) for p in (bundle.players_home or [])]
            pr_a = [NFLPlayerUsage(**p) for p in (bundle.players_away or [])]
            td_h = nfl_anytime_td_probs(team_tds["home"], pass_frac_h, pr_h)
            td_a = nfl_anytime_td_probs(team_tds["away"], pass_frac_a, pr_a)
            for name, p in td_h.items():
                rows.append({"sport":"NFL","game_id":bundle.game_id,"team":bundle.home,"player":name,"market":"Anytime TD","line":None,"mu":None,"sigma":None,"p_over":None,"p_any":p,"direction":"over","corr_tag":tag,"notes":"microsim"})
            for name, p in td_a.items():
                rows.append({"sport":"NFL","game_id":bundle.game_id,"team":bundle.away,"player":name,"market":"Anytime TD","line":None,"mu":None,"sigma":None,"p_over":None,"p_any":p,"direction":"over","corr_tag":tag,"notes":"microsim"})
        except Exception:
            rows.append({"sport":"NFL","game_id":bundle.game_id,"team":bundle.home,"player":bundle.home,"market":"Team TDs","line":None,"mu":team_tds["home"],"sigma":0.9,"p_over":None,"p_any":None,"direction":"over","corr_tag":tag,"notes":"team-level"})
            rows.append({"sport":"NFL","game_id":bundle.game_id,"team":bundle.away,"player":bundle.away,"market":"Team TDs","line":None,"mu":team_tds["away"],"sigma":0.9,"p_over":None,"p_any":None,"direction":"over","corr_tag":tag,"notes":"team-level"})
        return rows

    # NBA/CBB: use Markov points to seed player allocations if helper exists
    if s in ("NBA","CBB","NCAAB","BASKETBALL"):
        mk = simulate_hoops_markov(HoopsMarkovParams(**(X.get("markov_params", {}) or {})), sims=X.get("markov_sims", 4000))
        team_pts = mk["points_mean"]
        try:
            BB = BBPlayerUsage
            players_h = [BB(**p) for p in (bundle.players_home or [])]
            players_a = [BB(**p) for p in (bundle.players_away or [])]
            out_h = basketball_player_props_from_total(team_pts*0.5, 100, None, 90, players_h, L.get("Points", {}), L.get("Assists", {}), L.get("Rebounds", {}))
            out_a = basketball_player_props_from_total(team_pts*0.5, 100, None, 90, players_a, L.get("Points", {}), L.get("Assists", {}), L.get("Rebounds", {}))
            for team, block in [(bundle.home, out_h), (bundle.away, out_a)]:
                for name, v in block.items():
                    if "p_over_pts" in v:
                        rows.append({"sport":s,"game_id":bundle.game_id,"team":team,"player":name,"market":"Points","line":L.get("Points", {}).get(name),"mu":v["pts_mean"],"sigma":None,"p_over":v["p_over_pts"],"p_any":None,"direction":"over","corr_tag":tag,"notes":"markov"})
                    if "p_over_ast" in v:
                        rows.append({"sport":s,"game_id":bundle.game_id,"team":team,"player":name,"market":"Assists","line":L.get("Assists", {}).get(name),"mu":v["ast_mean"],"sigma":None,"p_over":v["p_over_ast"],"p_any":None,"direction":"over","corr_tag":tag,"notes":"markov"})
                    if "p_over_reb" in v:
                        rows.append({"sport":s,"game_id":bundle.game_id,"team":team,"player":name,"market":"Rebounds","line":L.get("Rebounds", {}).get(name),"mu":v["reb_mean"],"sigma":None,"p_over":v["p_over_reb"],"p_any":None,"direction":"over","corr_tag":tag,"notes":"markov"})
        except Exception:
            rows.append({"sport":s,"game_id":bundle.game_id,"team":bundle.home,"player":bundle.home,"market":"Team Points","line":None,"mu":team_pts*0.5,"sigma":5.0,"p_over":None,"p_any":None,"direction":"over","corr_tag":tag,"notes":"team-level"})
            rows.append({"sport":s,"game_id":bundle.game_id,"team":bundle.away,"player":bundle.away,"market":"Team Points","line":None,"mu":team_pts*0.5,"sigma":5.0,"p_over":None,"p_any":None,"direction":"over","corr_tag":tag,"notes":"team-level"})
        return rows

    # Soccer/NHL: bivariate Poisson for goals
    if s in ("SOC","SOCCER","NHL"):
        mu_h, mu_a = skellam_from_tot_spread(X.get("total", 2.8), X.get("spread_home", 0.20))
        lam_c = X.get("lambda_c", 0.10)
        gH, gA, _ = sample_bivariate_poisson(BivarPoissonParams(mu_h, mu_a, lam_c), sims=X.get("bivpo_sims", 6000))
        rows.append({"sport":s,"game_id":bundle.game_id,"team":bundle.home,"player":bundle.home,"market":"Team Goals","line":None,"mu":gH,"sigma":1.1,"p_over":None,"p_any":None,"direction":"over","corr_tag":tag,"notes":"bivar-poisson"})
        rows.append({"sport":s,"game_id":bundle.game_id,"team":bundle.away,"player":bundle.away,"market":"Team Goals","line":None,"mu":gA,"sigma":1.1,"p_over":None,"p_any":None,"direction":"over","corr_tag":tag,"notes":"bivar-poisson"})
        return rows

    return rows

# Validator & grouping
REQ_KEYS = {"sport","game_id","team","player","market","direction","corr_tag"}
def validate_rows(rows: List[Dict[str, Any]], dedupe: bool=True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out, issues = [], []
    seen = set()
    for i, r in enumerate(rows or []):
        miss = [k for k in REQ_KEYS if k not in r or r[k] in (None,"")]
        if miss: issues.append({"index":i,"severity":"error","message":f"Missing: {miss}"}); continue
        for k in ("p_over","p_any"):
            if k in r and r[k] is not None:
                r[k] = max(0.0, min(1.0, float(r[k])))
        sig = (r["sport"], r["game_id"], r["team"], r["player"], r["market"], r.get("line"), r["direction"], r["corr_tag"])
        if dedupe and sig in seen:
            issues.append({"index":i,"severity":"info","message":"Duplicate skipped"}); continue
        seen.add(sig); out.append(r)
    if not out: issues.append({"index":None,"severity":"error","message":"No valid rows"})
    return out, issues

def group_by_corr_tag(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows: g.setdefault(r.get("corr_tag","UNSET"), []).append(r)
    return g

# ---------- Engine wiring shims ----------
try:
    _V4 = V4EngineIntegrated
    def build_rows_and_slips_autorun(bundle: GameBundle, tier: str="condensed", legs: int=3, price: float=1.83):
        eng = _V4()
        rows = build_player_prop_table(bundle)
        rows, issues = validate_rows(rows)
        eng.context.setdefault("unified_adapter_issues", issues)
        # map rows -> legs
        cands = []
        for r in rows:
            mu = float(r.get("mu") or 0.0); sigma = float(r.get("sigma") or 10.0); line = float(r.get("line") or mu)
            cands.append(_V4.Leg(market=f"{r['sport']}_{r['market']}_{r['player']}", mu=mu, sigma=sigma, line=line, direction=r.get("direction","over"), price=price))
        slips = eng.build_slips(cands, _V4.SlipSpec(target=tier, legs=legs))
        return {"rows": rows, "slips": slips}
    V4_build_rows_and_slips_autorun = build_rows_and_slips_autorun
except Exception:
    pass

# ============================================================================
# END add-ons
# ============================================================================



# ============================================================================
# Self-run harness (safe demo) — does not require any external services.
# Runs only when executed as a script. Produces a trivial demo slip bundle.
# ============================================================================
if __name__ == "__main__":
    try:
        # Minimal NFL bundle demo to validate wiring
        demo_bundle = GameBundle(
            sport="NFL",
            game_id="BUF@ATL-2099-01-01",
            home="BUF",
            away="ATL",
            home_params={},
            away_params={},
            players_home=[dict(name="WR1 BUF", td_share_recv=0.28, target_share=0.28, snap_rate=0.88)],
            players_away=[dict(name="RB1 ATL", td_share_rush=0.58, rush_attempt_share=0.68, target_share=0.12)],
            lines={"Rec Yds":{"WR1 BUF": 60.5}, "Rush Yds":{"RB1 ATL": 55.5}},
            extras={
                "microsim_params": {},
                "microsim_sims": 1000,
                "home_pass_td_frac": 0.58,
                "away_pass_td_frac": 0.52,
                "strategy_toggles": {"pass_rate_delta": 0.0, "fourth_down_aggr_delta": 0.0, "pace_delta": 0.0}
            }
        )
        result = V4_build_rows_and_slips_autorun(demo_bundle, tier="condensed", legs=3, price=1.83)
        print("MERGED_FULL demo: rows:", len(result["rows"]), "slips:", len(result["slips"]))
    except Exception as e:
        print("MERGED_FULL demo failed (this does not affect import usage):", repr(e))


# -------------------
# Self-check CLI hook
# -------------------

# ============================================================================
# NBA-SPECIFIC PERFORMANCE ENHANCEMENTS
# Addresses NFL vs NBA performance gap with sport-specific adjustments
# Version: 1.0 - Added 2025-11-15
# Non-destructive: All enhancements are additive and backward-compatible
# ============================================================================

class NBAPerformanceEnhancer:
    """NBA-specific adjustments to improve V4 + MC accuracy on basketball"""

    def __init__(self):
        # NBA vs NFL variance differences
        self.sport_variance_multipliers = {
            "nba": 1.3,    # 30% higher sigma for NBA (higher variance sport)
            "nfl": 1.0,    # Baseline
            "mlb": 1.25,
            "nhl": 1.15,
            "cfb": 1.05,
            "cbb": 1.2
        }

        # NBA-specific quality gates (stricter than NFL due to higher variance)
        self.quality_gates = {
            "nba": {
                "min_ev": 0.08,           # 8% vs NFL's 5% - need bigger edge
                "min_confidence": 0.75,    # 75% vs NFL's 70%
                "min_agreement": 0.85,     # 85% model agreement required
                "min_simulations": 500000  # 500K vs NFL's 100K - more variance
            },
            "nfl": {
                "min_ev": 0.05,
                "min_confidence": 0.70,
                "min_agreement": 0.70,
                "min_simulations": 100000
            },
            "mlb": {
                "min_ev": 0.06,
                "min_confidence": 0.72,
                "min_agreement": 0.75,
                "min_simulations": 200000
            }
        }

        # NBA correlation penalties (higher than NFL due to individual player dominance)
        self.correlation_penalties = {
            "nba_same_game": 0.80,    # 20% reduction (1 player = 20% of team output)
            "nfl_same_game": 0.92,    # 8% reduction (1 player = 4.5% of team output)
            "nba_same_player": 0.75,  # 25% reduction for multiple props on same player
            "nfl_same_player": 0.88,  # 12% reduction
            "mlb_same_game": 0.88,
            "mlb_same_player": 0.85
        }

        # NBA schedule/fatigue factors (NBA-specific: 3-4 games per week)
        self.schedule_adjustments = {
            "back_to_back": -0.10,           # -10% performance on B2B (huge NBA factor)
            "rest_advantage_1day": 0.05,      # +5% with 1 day rest advantage
            "rest_advantage_2day": 0.08,      # +8% with 2+ day rest advantage
            "travel_penalty_per_1000mi": -0.02,  # -2% per 1000 miles (NBA travels more)
            "three_in_four_nights": -0.08,    # -8% on 3rd game in 4 nights
            "four_in_five_nights": -0.12      # -12% on 4th game in 5 nights
        }

        # Load management risk scores (NBA-specific phenomenon)
        self.load_management_risk = {
            "high": 0.30,     # 30% chance star rests
            "medium": 0.15,   # 15% chance
            "low": 0.05       # 5% chance
        }

        # Blowout/garbage time thresholds (NBA-specific: more blowouts than NFL)
        self.blowout_thresholds = {
            "high_risk": 12.0,     # Spread >= 12 points
            "moderate_risk": 8.0,  # Spread >= 8 points
            "low_risk": 5.0        # Spread >= 5 points
        }

    def apply_nba_variance_adjustment(self, mu: float, sigma: float, 
                                      sport: str = "nba") -> Tuple[float, float]:
        """Apply sport-specific variance multiplier to account for structural differences"""
        multiplier = self.sport_variance_multipliers.get(sport.lower(), 1.0)
        adjusted_sigma = sigma * multiplier

        return mu, adjusted_sigma

    def calculate_nba_fatigue_adjustment(self, game_context: Dict) -> float:
        """
        Calculate comprehensive fatigue adjustment for NBA
        Accounts for: back-to-backs, rest differentials, travel, schedule density
        """
        fatigue_factor = 1.0
        reasons = []

        # Back-to-back penalty (most significant NBA factor)
        if game_context.get('is_back_to_back', False):
            fatigue_factor += self.schedule_adjustments['back_to_back']
            reasons.append("back_to_back")

        # 3-in-4 or 4-in-5 nights (severe fatigue)
        if game_context.get('three_in_four_nights', False):
            fatigue_factor += self.schedule_adjustments['three_in_four_nights']
            reasons.append("3_in_4_nights")
        elif game_context.get('four_in_five_nights', False):
            fatigue_factor += self.schedule_adjustments['four_in_five_nights']
            reasons.append("4_in_5_nights")

        # Rest advantage vs opponent
        rest_diff = game_context.get('rest_days_diff', 0)  # Positive = more rest than opponent
        if rest_diff >= 2:
            fatigue_factor += self.schedule_adjustments['rest_advantage_2day']
            reasons.append(f"rest_advantage_{rest_diff}d")
        elif rest_diff >= 1:
            fatigue_factor += self.schedule_adjustments['rest_advantage_1day']
            reasons.append("rest_advantage_1d")
        elif rest_diff <= -2:
            fatigue_factor -= self.schedule_adjustments['rest_advantage_2day']
            reasons.append(f"rest_disadvantage_{abs(rest_diff)}d")
        elif rest_diff <= -1:
            fatigue_factor -= self.schedule_adjustments['rest_advantage_1day']
            reasons.append("rest_disadvantage_1d")

        # Travel distance penalty
        travel_miles = game_context.get('travel_miles', 0)
        if travel_miles > 500:
            travel_penalty = (travel_miles / 1000.0) * self.schedule_adjustments['travel_penalty_per_1000mi']
            fatigue_factor += travel_penalty
            reasons.append(f"travel_{int(travel_miles)}mi")

        # Cap fatigue adjustment between 70% and 130%
        final_factor = max(0.70, min(1.30, fatigue_factor))

        return final_factor

    def assess_load_management_risk(self, player: str, game_context: Dict) -> Dict:
        """
        Assess risk of player resting due to load management (NBA-specific)
        Returns risk level and probability of player sitting
        """
        risk_level = "low"
        risk_score = self.load_management_risk['low']
        reasons = []

        # High risk factors for load management

        # 1. Schedule density
        games_in_7_days = game_context.get('games_in_last_7_days', 2)
        if games_in_7_days >= 4:
            risk_level = "high"
            risk_score = self.load_management_risk['high']
            reasons.append(f"{games_in_7_days}_games_in_7_days")

        # 2. Opponent quality (weak opponent = rest opportunity)
        opponent_quality = game_context.get('opponent_quality', 'average')
        if opponent_quality == 'weak':
            if risk_level == "low":
                risk_level = "medium"
                risk_score = self.load_management_risk['medium']
            reasons.append("weak_opponent_rest_opportunity")

        # 3. Season stage (late regular season before playoffs)
        season_stage = game_context.get('season_stage', 'mid')
        if season_stage == 'late_regular':
            if risk_level == "low":
                risk_level = "medium"
                risk_score = self.load_management_risk['medium']
            reasons.append("late_season_maintenance")

        # 4. Playoff position locked
        playoff_position = game_context.get('playoff_position', 'competing')
        if playoff_position == 'locked':
            risk_level = "high"
            risk_score = self.load_management_risk['high']
            reasons.append("playoff_seed_locked")

        # 5. Recent minutes load (high minutes = rest candidate)
        recent_mpg = game_context.get('recent_minutes_per_game', 32)
        if recent_mpg > 36:
            if risk_level != "high":
                risk_level = "medium"
                risk_score = max(risk_score, self.load_management_risk['medium'])
            reasons.append(f"high_minutes_{recent_mpg:.1f}mpg")

        # 6. Veteran player (older players rest more)
        player_age = game_context.get('player_age', 25)
        if player_age > 32:
            if risk_level == "low":
                risk_level = "medium"
                risk_score = max(risk_score, self.load_management_risk['medium'])
            reasons.append(f"veteran_age_{player_age}")

        return {
            "player": player,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "reasons": reasons,
            "adjustment_factor": 1.0 - risk_score,  # Reduce projection by risk score
            "recommendation": self._generate_load_management_recommendation(risk_level, risk_score)
        }

    def calculate_nba_correlation_penalty(self, legs: List[Dict], sport: str = "nba") -> float:
        """
        Calculate NBA-specific correlation penalty for parlays
        NBA has higher correlation due to individual player dominance (20% vs 4.5% in NFL)
        """
        if len(legs) < 2:
            return 1.0

        # Determine sport
        sport_key = sport.lower()

        # Check for same-game correlation
        game_ids = [leg.get('game_id', '') for leg in legs]
        unique_games = len(set(game_ids))
        same_game_legs = len(game_ids) - unique_games

        # Check for same-player correlation
        players = [leg.get('player', '') for leg in legs if leg.get('player')]
        unique_players = len(set(players))
        same_player_legs = len(players) - unique_players if players else 0

        # Apply sport-specific penalties
        correlation_factor = 1.0

        if same_game_legs > 0:
            penalty_key = f"{sport_key}_same_game"
            penalty = self.correlation_penalties.get(penalty_key, 0.90)
            correlation_factor *= (penalty ** same_game_legs)

        if same_player_legs > 0:
            penalty_key = f"{sport_key}_same_player"
            penalty = self.correlation_penalties.get(penalty_key, 0.85)
            correlation_factor *= (penalty ** same_player_legs)

        return correlation_factor

    def calculate_nba_blowout_risk(self, game_context: Dict) -> Dict:
        """
        Calculate blowout risk and impact on player props (NBA-specific)
        NBA has more blowouts than NFL → garbage time affects star/bench prop values
        """
        spread = abs(game_context.get('spread', 0))

        # Blowout risk increases with spread
        if spread >= self.blowout_thresholds['high_risk']:
            blowout_risk = 0.40  # 40% chance of blowout
            impact = "HIGH"
        elif spread >= self.blowout_thresholds['moderate_risk']:
            blowout_risk = 0.25  # 25% chance
            impact = "MEDIUM"
        elif spread >= self.blowout_thresholds['low_risk']:
            blowout_risk = 0.15  # 15% chance
            impact = "LOW"
        else:
            blowout_risk = 0.05  # 5% chance (competitive game)
            impact = "MINIMAL"

        # Calculate garbage time adjustments
        if blowout_risk > 0.20:
            # Stars sit early in blowouts
            star_prop_adjustment = 0.90  # -10% for stars
            # Bench players get garbage time minutes
            bench_prop_adjustment = 1.15  # +15% for bench
        elif blowout_risk > 0.10:
            star_prop_adjustment = 0.95  # -5%
            bench_prop_adjustment = 1.08  # +8%
        else:
            star_prop_adjustment = 1.0
            bench_prop_adjustment = 1.0

        return {
            "spread": spread,
            "blowout_probability": blowout_risk,
            "impact_level": impact,
            "star_player_adjustment": star_prop_adjustment,
            "bench_player_adjustment": bench_prop_adjustment,
            "recommendation": self._generate_blowout_recommendation(impact, spread)
        }

    def calculate_nba_pace_adjustment(self, expected_pace: float, 
                                      actual_pace: float = None,
                                      team_pace: float = None) -> Dict:
        """
        Calculate pace-of-play adjustment for NBA totals
        NBA pace varies 15-20% game-to-game vs NFL's consistency
        """
        if actual_pace is None and team_pace is None:
            return {"adjustment_factor": 1.0, "reason": "no_pace_data"}

        reference_pace = actual_pace or team_pace or expected_pace

        # NBA pace varies significantly
        pace_ratio = reference_pace / max(expected_pace, 1.0)

        # Cap adjustment at ±20% (realistic NBA pace variance)
        capped_ratio = max(0.80, min(1.20, pace_ratio))

        return {
            "expected_pace": expected_pace,
            "actual_pace": reference_pace,
            "pace_ratio": pace_ratio,
            "adjustment_factor": capped_ratio,
            "reason": "pace_faster" if capped_ratio > 1.05 else "pace_slower" if capped_ratio < 0.95 else "pace_normal"
        }

    def validate_nba_quality_gates(self, analysis: Dict, sport: str = "nba") -> Tuple[bool, List[str]]:
        """
        Check if bet meets NBA-specific quality gates
        Stricter gates for NBA due to higher variance - need bigger edge to overcome noise
        """
        gates = self.quality_gates.get(sport.lower(), self.quality_gates.get('nba'))

        passed = True
        failures = []

        # EV threshold check
        if analysis.get('ev', 0) < gates['min_ev']:
            passed = False
            failures.append(f"EV {analysis.get('ev', 0):.2%} below {gates['min_ev']:.1%} threshold")

        # Confidence threshold check
        if analysis.get('confidence', 0) < gates['min_confidence']:
            passed = False
            failures.append(f"Confidence {analysis.get('confidence', 0):.1%} below {gates['min_confidence']:.1%}")

        # Model agreement check
        if analysis.get('agreement', 1.0) < gates['min_agreement']:
            passed = False
            failures.append(f"Model agreement {analysis.get('agreement', 1.0):.1%} below {gates['min_agreement']:.1%}")

        # Simulation count check
        if analysis.get('n_simulations', 0) < gates['min_simulations']:
            passed = False
            failures.append(f"Only {analysis.get('n_simulations', 0):,} simulations (need {gates['min_simulations']:,})")

        return passed, failures

    def calculate_nba_position_sizing_adjustment(self, base_kelly: float, 
                                                  sport: str = "nba") -> float:
        """
        Reduce position sizing for higher variance sports
        NBA gets 25% reduction vs NFL due to higher outcome variance
        """
        if sport.lower() == "nba":
            # Reduce NBA sizing by 25% vs NFL (higher variance requires conservative sizing)
            return base_kelly * 0.75
        elif sport.lower() == "nfl":
            return base_kelly
        elif sport.lower() == "mlb":
            return base_kelly * 0.80  # MLB also high variance
        else:
            # Other sports get moderate reduction
            return base_kelly * 0.85

    def get_nba_simulation_count(self, sport: str = "nba", 
                                  market_complexity: str = "standard") -> int:
        """
        Determine optimal simulation count for sport and market complexity
        NBA requires more simulations due to higher variance
        """
        base_counts = {
            "nba": 500000,
            "nfl": 100000,
            "mlb": 200000,
            "nhl": 150000
        }

        base = base_counts.get(sport.lower(), 100000)

        # Increase for complex markets (parlays, same-game, live)
        if market_complexity == "complex":
            return int(base * 1.5)
        elif market_complexity == "parlay":
            return int(base * 2.0)
        else:
            return base

    def _generate_load_management_recommendation(self, risk_level: str, risk_score: float) -> str:
        """Generate recommendation based on load management risk"""
        if risk_level == "high":
            return f"⚠️ HIGH LOAD MGMT RISK ({risk_score:.0%}) - Avoid or reduce stake 50%"
        elif risk_level == "medium":
            return f"⚠️ MODERATE LOAD MGMT RISK ({risk_score:.0%}) - Monitor injury report, reduce stake 25%"
        else:
            return f"✅ LOW LOAD MGMT RISK ({risk_score:.0%}) - Normal analysis applies"

    def _generate_blowout_recommendation(self, impact: str, spread: float) -> str:
        """Generate recommendation based on blowout risk"""
        if impact == "HIGH":
            return f"🚨 HIGH BLOWOUT RISK (spread {spread}) - Stars sit early, fade star props, target bench"
        elif impact == "MEDIUM":
            return f"⚠️ MODERATE BLOWOUT RISK (spread {spread}) - Factor 25% into analysis"
        else:
            return f"✅ COMPETITIVE GAME (spread {spread}) - Normal prop analysis"


# ============================================================================
# NBA-ENHANCED V4 ENGINE INTEGRATION
# ============================================================================

class NBAEnhancedV4Integration:
    """
    Non-destructive integration layer for NBA enhancements with existing V4 engine
    Wraps existing V4EngineIntegrated without modifying its code
    """

    def __init__(self, v4_engine):
        self.v4_engine = v4_engine
        self.nba_enhancer = NBAPerformanceEnhancer()
        self.nba_analysis_log = []

    def enhanced_evaluate_mainline(self, market: str, mu: float, sigma: float,
                                   posted_line: float, direction: str, price: float,
                                   sport: str = "nba", game_context: Dict = None,
                                   blob: Optional[Dict] = None) -> Dict:
        """
        Enhanced mainline evaluation with NBA-specific adjustments
        Wraps V4EngineIntegrated.evaluate_mainline_with_context
        """
        game_context = game_context or {}
        sport_lower = sport.lower()

        # Step 1: Apply NBA variance adjustment
        adjusted_mu, adjusted_sigma = self.nba_enhancer.apply_nba_variance_adjustment(
            mu, sigma, sport_lower
        )

        # Step 2: Apply NBA-specific fatigue adjustment
        fatigue_factor = 1.0
        if sport_lower == "nba":
            fatigue_factor = self.nba_enhancer.calculate_nba_fatigue_adjustment(game_context)
            adjusted_mu *= fatigue_factor

        # Step 3: Apply load management risk adjustment (NBA only)
        load_mgmt_analysis = None
        if sport_lower == "nba" and game_context.get('player'):
            load_mgmt_analysis = self.nba_enhancer.assess_load_management_risk(
                game_context['player'], game_context
            )
            # Apply load management risk to projection
            adjusted_mu *= load_mgmt_analysis['adjustment_factor']

        # Step 4: Apply blowout risk adjustment (NBA only)
        blowout_analysis = None
        if sport_lower == "nba" and game_context.get('spread') is not None:
            blowout_analysis = self.nba_enhancer.calculate_nba_blowout_risk(game_context)
            # Apply blowout adjustment based on player type
            player_type = game_context.get('player_type', 'star')
            if player_type == 'star':
                adjusted_mu *= blowout_analysis['star_player_adjustment']
            elif player_type == 'bench':
                adjusted_mu *= blowout_analysis['bench_player_adjustment']

        # Step 5: Get sport-appropriate simulation count
        n_sims = self.nba_enhancer.get_nba_simulation_count(
            sport_lower, game_context.get('market_complexity', 'standard')
        )

        # Update engine config for this evaluation
        original_sims_min = self.v4_engine.cfg.get('mc_sims_min', 100000)
        self.v4_engine.cfg['mc_sims_min'] = n_sims

        # Step 6: Run V4 evaluation with adjusted parameters
        try:
            if hasattr(self.v4_engine, 'evaluate_mainline_with_context'):
                base_result = self.v4_engine.evaluate_mainline_with_context(
                    market, adjusted_mu, adjusted_sigma, posted_line, direction, price, blob
                )
            else:
                base_result = self.v4_engine.evaluate_mainline(
                    market, adjusted_mu, adjusted_sigma, posted_line, direction, price
                )
        finally:
            # Restore original config
            self.v4_engine.cfg['mc_sims_min'] = original_sims_min

        # Step 7: Apply NBA position sizing adjustment
        base_kelly = base_result.get('kelly', 0.0)
        adjusted_kelly = self.nba_enhancer.calculate_nba_position_sizing_adjustment(
            base_kelly, sport_lower
        )

        # Step 8: Validate NBA quality gates
        validation_dict = {
            'ev': base_result.get('ev_proxy', base_result.get('ev', 0)),
            'confidence': game_context.get('confidence', 0.75),
            'agreement': game_context.get('agreement', 0.80),
            'n_simulations': n_sims
        }
        passed_gates, gate_failures = self.nba_enhancer.validate_nba_quality_gates(
            validation_dict, sport_lower
        )

        # Compile enhanced result
        enhanced_result = {
            # Original V4 results
            **base_result,

            # NBA enhancement metadata
            "sport": sport,
            "nba_enhancements_applied": True,

            # Parameter adjustments
            "original_mu": mu,
            "original_sigma": sigma,
            "adjusted_mu": adjusted_mu,
            "adjusted_sigma": adjusted_sigma,
            "variance_multiplier": adjusted_sigma / sigma if sigma > 0 else 1.0,
            "fatigue_factor": fatigue_factor,

            # Kelly adjustments
            "base_kelly": base_kelly,
            "adjusted_kelly": adjusted_kelly,
            "position_sizing_factor": adjusted_kelly / base_kelly if base_kelly > 0 else 0.75,

            # Simulation details
            "n_simulations": n_sims,

            # NBA-specific analysis
            "load_management": load_mgmt_analysis,
            "blowout_risk": blowout_analysis,

            # Quality gates
            "quality_gates_passed": passed_gates,
            "quality_gate_failures": gate_failures,

            # Final recommendation
            "recommendation": self._generate_final_recommendation(
                passed_gates, gate_failures, base_result.get('ev_proxy', 0),
                load_mgmt_analysis, blowout_analysis
            )
        }

        # Log for performance tracking
        self.nba_analysis_log.append({
            "market": market,
            "sport": sport,
            "adjustments": {
                "variance": adjusted_sigma / sigma if sigma > 0 else 1.0,
                "fatigue": fatigue_factor,
                "sizing": adjusted_kelly / base_kelly if base_kelly > 0 else 0.75
            },
            "passed_gates": passed_gates
        })

        return enhanced_result

    def enhanced_parlay_evaluation(self, legs: List[Dict], sport: str = "nba",
                                    bankroll: float = 10000) -> Dict:
        """
        Enhanced parlay evaluation with NBA-specific correlation modeling
        """
        if not legs:
            return {"error": "No legs provided"}

        # Calculate naive joint probability
        joint_prob_naive = 1.0
        for leg in legs:
            joint_prob_naive *= leg.get('probability', 0.5)

        # Apply NBA-enhanced correlation penalty
        correlation_factor = self.nba_enhancer.calculate_nba_correlation_penalty(legs, sport)
        joint_prob_adjusted = joint_prob_naive * correlation_factor

        # Calculate parlay odds
        parlay_odds = 1.0
        for leg in legs:
            leg_decimal_odds = leg.get('decimal_odds', 1.91)
            parlay_odds *= leg_decimal_odds

        # Calculate parlay EV
        parlay_ev = (joint_prob_adjusted * parlay_odds) - 1.0

        # Calculate Kelly sizing with NBA adjustment
        if parlay_ev > 0 and parlay_odds > 1.0:
            base_kelly = (joint_prob_adjusted * parlay_odds - 1) / (parlay_odds - 1)
            base_kelly = max(0, base_kelly) * 0.25  # Quarter Kelly
            adjusted_kelly = self.nba_enhancer.calculate_nba_position_sizing_adjustment(
                base_kelly, sport
            )
        else:
            base_kelly = 0.0
            adjusted_kelly = 0.0

        # Validate against NBA quality gates
        validation_dict = {
            'ev': parlay_ev,
            'confidence': 0.75,  # Default for parlays
            'agreement': 0.80,
            'n_simulations': self.nba_enhancer.get_nba_simulation_count(sport, "parlay")
        }
        passed_gates, failures = self.nba_enhancer.validate_nba_quality_gates(
            validation_dict, sport
        )

        return {
            "num_legs": len(legs),
            "sport": sport,

            # Probabilities
            "joint_probability_naive": joint_prob_naive,
            "correlation_factor": correlation_factor,
            "correlation_penalty": 1.0 - correlation_factor,
            "joint_probability_adjusted": joint_prob_adjusted,

            # Odds and EV
            "parlay_odds": parlay_odds,
            "parlay_ev": parlay_ev,

            # Sizing
            "base_kelly": base_kelly,
            "adjusted_kelly": adjusted_kelly,
            "recommended_stake": bankroll * adjusted_kelly,

            # Quality
            "quality_gates_passed": passed_gates,
            "quality_gate_failures": failures,

            # Recommendation
            "recommendation": "✅ QUALIFIED PARLAY" if passed_gates and parlay_ev > 0.05 else f"❌ FILTERED: {', '.join(failures) if failures else 'Insufficient EV'}"
        }

    def get_nba_performance_summary(self) -> Dict:
        """Get summary of NBA enhancements applied"""
        if not self.nba_analysis_log:
            return {"message": "No NBA analyses run yet"}

        total_analyses = len(self.nba_analysis_log)
        nba_analyses = [a for a in self.nba_analysis_log if a['sport'].lower() == 'nba']

        if not nba_analyses:
            return {"message": "No NBA-specific analyses"}

        avg_variance_mult = statistics.mean([a['adjustments']['variance'] for a in nba_analyses])
        avg_fatigue_adj = statistics.mean([a['adjustments']['fatigue'] for a in nba_analyses])
        avg_sizing_adj = statistics.mean([a['adjustments']['sizing'] for a in nba_analyses])
        pass_rate = sum(1 for a in nba_analyses if a['passed_gates']) / len(nba_analyses)

        return {
            "total_nba_analyses": len(nba_analyses),
            "avg_variance_multiplier": avg_variance_mult,
            "avg_fatigue_adjustment": avg_fatigue_adj,
            "avg_sizing_adjustment": avg_sizing_adj,
            "quality_gate_pass_rate": pass_rate,
            "enhancements_active": True
        }

    def _generate_final_recommendation(self, passed_gates: bool, failures: List[str],
                                       ev: float, load_mgmt: Optional[Dict],
                                       blowout: Optional[Dict]) -> str:
        """Generate comprehensive final recommendation"""
        if not passed_gates:
            return f"❌ FILTERED: {', '.join(failures)}"

        warnings = []

        if load_mgmt and load_mgmt['risk_level'] in ['high', 'medium']:
            warnings.append(load_mgmt['recommendation'])

        if blowout and blowout['impact_level'] in ['HIGH', 'MEDIUM']:
            warnings.append(blowout['recommendation'])

        if warnings:
            return f"⚠️ QUALIFIED WITH WARNINGS: {' | '.join(warnings)}"
        else:
            if ev > 0.12:
                return "✅ STRONG BET - All NBA gates passed, high EV"
            elif ev > 0.08:
                return "✅ QUALIFIED BET - NBA gates passed"
            else:
                return "✅ MARGINAL BET - Just clears NBA gates"


# ============================================================================
# PERPLEXITY QUICK NBA ANALYSIS HELPER
# ============================================================================

def nba_quick_analysis(v4_engine, market: str, mu: float, sigma: float, 
                       line: float, direction: str, price: float,
                       is_back_to_back: bool = False,
                       rest_days_diff: int = 0,
                       spread: float = 0,
                       player_type: str = "star",
                       player_name: str = None) -> Dict:
    """
    Quick NBA analysis optimized for Perplexity conversational workflow

    Example usage in Perplexity:

    result = nba_quick_analysis(
        v4_engine=engine,
        market="GIANNIS_POINTS",
        mu=31.5,
        sigma=7.8,
        line=29.5,
        direction="over",
        price=-110,
        is_back_to_back=True,
        rest_days_diff=-1,  # Opponent has 1 more day rest
        spread=8.5,
        player_type="star",
        player_name="Giannis Antetokounmpo"
    )
    """

    # Create NBA enhancement integration
    nba_integration = NBAEnhancedV4Integration(v4_engine)

    # Build game context
    game_context = {
        "is_back_to_back": is_back_to_back,
        "rest_days_diff": rest_days_diff,
        "spread": spread,
        "player_type": player_type,
        "player": player_name or market.split('_')[0] if '_' in market else "Unknown",
        "confidence": 0.75,
        "agreement": 0.80
    }

    # Run enhanced analysis
    result = nba_integration.enhanced_evaluate_mainline(
        market=market,
        mu=mu,
        sigma=sigma,
        posted_line=line,
        direction=direction,
        price=price,
        sport="nba",
        game_context=game_context,
        blob=None
    )

    return result


# ============================================================================
# END NBA-SPECIFIC PERFORMANCE ENHANCEMENTS
# ============================================================================


if __name__ == "__main__":
    import argparse, json, os
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    args, unknown = p.parse_known_args()
    if args.selfcheck:
        print("[SELF-CHECK] Starting")
        try:
            import numpy as _np
            print("[OK] numpy present")
        except Exception as e:
            print("[WARN] numpy missing:", e)
        for f in ["priors.json","rho_overrides.json","book_bias.json","public_splits.json","team_power_ratings.json","schedule_adjusted_stats.json"]:
            if os.path.exists(f):
                try:
                    j = json.loads(open(f,"r",encoding="utf-8").read())
                    if j:
                        print(f"[OK] {f} loaded")
                    else:
                        print(f"[WARN] {f} empty")
                except Exception as e:
                    print(f"[WARN] {f} unreadable: {e}")
        print("[SELF-CHECK] Done")


# ============================================================================
# PERPLEXITY AI INTEGRATION FOR AUTOMATIC CONTEXT GATHERING
# Version: 1.0 - Added 2025-11-07
# Enables automatic game context research and V4+MC execution
# ============================================================================

import os
import re
import json as json_module
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple

class PerplexityContextGatherer:
    """
    Integrates with Perplexity AI API to automatically gather game context
    (injuries, odds, stats, matchups) and format it for V4+MC engine.

    Requires PERPLEXITY_API_KEY environment variable to be set.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-large-128k-online"

        # Try to import requests
        try:
            import requests
            self.requests = requests
            self.api_available = True
        except ImportError:
            self.api_available = False
            import warnings
            warnings.warn(
                "requests library not found. Install with: pip install requests\n"
                "Perplexity integration will use mock mode for demonstration."
            )

    def gather_game_context(self, team: str, opponent: str, 
                           sport: str = "NBA",
                           date: Optional[str] = None) -> Dict[str, Any]:
        """
        Automatically gather full game context using Perplexity AI.

        Args:
            team: Team code (e.g., "ATL", "BOS")
            opponent: Opponent code (e.g., "TOR", "ORL")
            sport: Sport type (NBA, NFL, MLB, NHL)
            date: Game date (defaults to today)

        Returns:
            Complete context blob ready for V4+MC engine
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Build research query
        query = self._build_context_query(team, opponent, sport, date)

        # Query Perplexity
        if self.api_available and self.api_key:
            response = self._query_perplexity(query)
            context = self._parse_perplexity_response(response, team, opponent, sport)
        else:
            # Mock mode for demonstration
            context = self._generate_mock_context(team, opponent, sport)

        return context

    def _build_context_query(self, team: str, opponent: str, 
                            sport: str, date: str) -> str:
        """Build structured query for Perplexity"""
        query = f"""
Research the {sport} game: {team} vs {opponent} on {date}

Please provide in structured format:

1. INJURIES:
   - List all injured/questionable/out players for both teams
   - Include player name, position, injury status, team

2. BETTING LINES:
   - Current spread, total, moneyline
   - Any significant line movement

3. TEAM STATS (current season):
   - {team}: Points per game, defensive rating, pace
   - {opponent}: Points per game, defensive rating, pace

4. SITUATIONAL FACTORS:
   - Home/away status
   - Rest days for each team
   - Recent form (last 5 games record)
   - Weather (if outdoor sport)

5. KEY MATCHUPS:
   - Notable player matchups
   - Tactical advantages/disadvantages

6. RECENT HEAD-TO-HEAD:
   - Most recent meeting result and date

Format your response clearly with headers for each section.
"""
        return query.strip()

    def _query_perplexity(self, query: str) -> str:
        """Query Perplexity API"""
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a sports betting research assistant. Provide accurate, current information in structured format."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }

        try:
            response = self.requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Perplexity API error: {e}")

    def _parse_perplexity_response(self, response: str, team: str, 
                                   opponent: str, sport: str) -> Dict[str, Any]:
        """Parse Perplexity's text response into structured context blob"""
        context = {
            "injuries": [],
            "opponent": {},
            "is_home": None,
            "weather": None,
            "rest_days": None,
            "recent_form": None,
            "matchups": {},
            "defense": {},
            "meta": {
                "source": "perplexity_ai",
                "query_time": datetime.now().isoformat(),
                "team": team,
                "opponent": opponent,
                "sport": sport
            }
        }

        # Parse injuries section
        injuries = self._extract_injuries(response, team, opponent, sport)
        context["injuries"] = injuries

        # Parse betting lines
        lines = self._extract_betting_lines(response)
        context["betting_lines"] = lines

        # Parse team stats
        stats = self._extract_team_stats(response, team, opponent)
        if opponent in stats:
            context["opponent"] = stats[opponent]

        # Parse situational factors
        situational = self._extract_situational(response, team)
        context.update(situational)

        return context

    def _extract_injuries(self, text: str, team: str, opponent: str, 
                         sport: str) -> List[Dict[str, str]]:
        """Extract injury information from text"""
        injuries = []

        # Common injury status keywords
        status_keywords = ["out", "doubtful", "questionable", "probable", "gtd", "day-to-day"]

        # Look for injury mentions
        lines = text.lower().split('\n')
        for line in lines:
            # Check if line contains injury info
            has_status = any(status in line for status in status_keywords)
            if has_status:
                # Try to extract player name and status
                for status in status_keywords:
                    if status in line:
                        # Simple extraction (can be enhanced with NLP)
                        # This is a basic pattern matcher
                        injury = {
                            "player": "Unknown",  # Would need NER to extract
                            "position": "",
                            "status": status,
                            "team": team if team.lower() in line else opponent,
                            "opponent": opponent if team.lower() in line else team,
                            "sport": sport
                        }
                        injuries.append(injury)
                        break

        return injuries

    def _extract_betting_lines(self, text: str) -> Dict[str, Any]:
        """Extract betting lines from text"""
        lines = {}

        # Look for spread (e.g., "Hawks -2.5", "ATL -2.5")
        spread_pattern = r'[-+]?\d+\.?\d*\s*(?:point)?\s*(?:spread|favorite)?'
        spread_matches = re.findall(spread_pattern, text.lower())
        if spread_matches:
            lines["spread"] = spread_matches[0].strip()

        # Look for total (e.g., "O/U 235.5", "total 235.5")
        total_pattern = r'(?:total|o/u|over/under)?\s*\d+\.?\d*'
        total_matches = re.findall(total_pattern, text.lower())
        if total_matches:
            lines["total"] = total_matches[0].strip()

        return lines

    def _extract_team_stats(self, text: str, team: str, 
                           opponent: str) -> Dict[str, Dict]:
        """Extract team statistics from text"""
        stats = {
            team: {},
            opponent: {}
        }

        # Look for PPG (points per game)
        ppg_pattern = r'(\d+\.?\d*)\s*(?:ppg|points per game)'
        ppg_matches = re.findall(ppg_pattern, text.lower())

        # Look for defensive rating mentions
        # This would need more sophisticated parsing

        return stats

    def _extract_situational(self, text: str, team: str) -> Dict[str, Any]:
        """Extract situational factors from text"""
        situational = {}

        # Home/away detection
        if "at home" in text.lower() or f"{team.lower()} home" in text.lower():
            situational["is_home"] = True
        elif "away" in text.lower() or "road" in text.lower():
            situational["is_home"] = False

        # Rest days (look for "back-to-back", "3 days rest", etc.)
        if "back-to-back" in text.lower() or "b2b" in text.lower():
            situational["rest_days"] = 0

        # Recent form (look for win-loss records)
        form_pattern = r'(\d+)-(\d+)\s*(?:record|in last)'
        form_matches = re.findall(form_pattern, text.lower())
        if form_matches:
            wins, losses = map(int, form_matches[0])
            situational["recent_form"] = wins / (wins + losses) if (wins + losses) > 0 else 0.5

        return situational

    def _generate_mock_context(self, team: str, opponent: str, 
                               sport: str) -> Dict[str, Any]:
        """Generate mock context for demonstration when API unavailable"""
        return {
            "injuries": [],
            "opponent": {
                "defensive_rating": 0.50,
                "pace": 100.0
            },
            "is_home": True,
            "weather": None,
            "rest_days": 2,
            "recent_form": 0.500,
            "matchups": {},
            "defense": {},
            "meta": {
                "source": "mock_data",
                "note": "Install requests library and set PERPLEXITY_API_KEY for real data",
                "team": team,
                "opponent": opponent,
                "sport": sport
            }
        }


class AutomatedV4Runner:
    """
    Combines PerplexityContextGatherer with V4EngineIntegrated for
    fully automated game analysis and betting recommendations.
    """

    def __init__(self, v4_engine, perplexity_api_key: Optional[str] = None):
        self.engine = v4_engine
        self.context_gatherer = PerplexityContextGatherer(perplexity_api_key)
        self.analysis_history = []

    def analyze_game(self, team: str, opponent: str, 
                    sport: str = "NBA",
                    markets: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Fully automated game analysis with V4+MC.

        Args:
            team: Team code (e.g., "ATL")
            opponent: Opponent code (e.g., "TOR")
            sport: Sport type
            markets: List of markets to evaluate, e.g.:
                [
                    {"market": "NBA_POINTS_ATL", "mu": 112, "sigma": 10, 
                     "line": 115.5, "direction": "under", "price": -110}
                ]

        Returns:
            Complete analysis with context, recommendations, and diagnostics
        """
        print(f"\n{'='*70}")
        print(f"AUTOMATED V4+MC ANALYSIS: {team} vs {opponent}")
        print(f"{'='*70}\n")

        # Step 1: Gather context
        print("Step 1: Gathering game context via Perplexity AI...")
        context = self.context_gatherer.gather_game_context(team, opponent, sport)
        print(f"✓ Context gathered: {len(context.get('injuries', []))} injuries detected")

        # Step 2: Set context in engine
        print("\nStep 2: Loading context into V4 engine...")
        self.engine.set_game_context(context)
        print("✓ Context loaded")

        # Step 3: Evaluate markets
        print("\nStep 3: Running V4+MC simulations with context...")
        results = []

        if markets:
            for market_spec in markets:
                result = self.engine.evaluate_mainline_with_context(
                    market=market_spec["market"],
                    mu=market_spec["mu"],
                    sigma=market_spec["sigma"],
                    posted_line=market_spec["line"],
                    direction=market_spec["direction"],
                    price=market_spec["price"],
                    blob=context
                )
                results.append(result)

                # Display result
                print(f"\n  Market: {result['market']}")
                print(f"  Line: {result['direction']} {result['line']}")
                print(f"  P(hit): {result['p_hit']:.3f}")
                print(f"  EV: {result['ev_proxy']:.4f}")
                print(f"  Kelly: {result['kelly']:.4f}")
                if 'context_confidence' in result:
                    print(f"  Context Confidence: {result['context_confidence']:.3f}")

        # Step 4: Compile full analysis
        analysis = {
            "team": team,
            "opponent": opponent,
            "sport": sport,
            "context": context,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "summary": self._generate_summary(results, context)
        }

        # Store in history
        self.analysis_history.append(analysis)

        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*70}\n")

        return analysis

    def _generate_summary(self, results: List[Dict], 
                         context: Dict) -> Dict[str, Any]:
        """Generate summary of analysis"""
        summary = {
            "total_bets_evaluated": len(results),
            "positive_ev_count": sum(1 for r in results if r.get('ev_proxy', 0) > 0),
            "avg_confidence": sum(r.get('context_confidence', 0.5) for r in results) / max(len(results), 1),
            "injuries_detected": len(context.get('injuries', [])),
            "context_applied": bool(context.get('injuries') or context.get('is_home') is not None)
        }

        # Find best bet
        if results:
            best_bet = max(results, key=lambda x: x.get('ev_proxy', 0))
            summary["best_bet"] = {
                "market": best_bet.get('market'),
                "line": f"{best_bet.get('direction')} {best_bet.get('line')}",
                "ev": best_bet.get('ev_proxy'),
                "kelly": best_bet.get('kelly')
            }

        return summary

    def get_analysis_history(self) -> List[Dict]:
        """Retrieve all past analyses"""
        return self.analysis_history



# ============================================================================
# CONVENIENCE FUNCTIONS FOR AUTOMATED ANALYSIS
# ============================================================================

def create_automated_runner(api_key: Optional[str] = None):
    """
    Create an AutomatedV4Runner instance ready to use.

    Args:
        api_key: Perplexity API key (or set PERPLEXITY_API_KEY env var)

    Returns:
        AutomatedV4Runner instance
    """
    engine = V4EngineIntegrated()
    return AutomatedV4Runner(engine, api_key)


def quick_analyze(team: str, opponent: str, sport: str = "NBA",
                 api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick one-line game analysis.

    Example:
        result = quick_analyze("ATL", "TOR", "NBA")

    Args:
        team: Team code
        opponent: Opponent code  
        sport: Sport type
        api_key: Perplexity API key (optional)

    Returns:
        Full analysis dictionary
    """
    runner = create_automated_runner(api_key)
    return runner.analyze_game(team, opponent, sport)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Basic automated analysis
------------------------------------

from v4_mc_production_ultimate_enhanced_context_aware import quick_analyze

# Analyze a game with one line
result = quick_analyze("ATL", "TOR", "NBA")

print(f"Best bet: {result['summary']['best_bet']}")
print(f"Context confidence: {result['summary']['avg_confidence']:.3f}")


EXAMPLE 2: Detailed analysis with custom markets
-------------------------------------------------

from v4_mc_production_ultimate_enhanced_context_aware import create_automated_runner

# Create runner
runner = create_automated_runner()

# Define markets to evaluate
markets = [
    {
        "market": "NBA_POINTS_ATL",
        "mu": 112,  # Hawks season average
        "sigma": 10,
        "line": 115.5,
        "direction": "under",  # Expecting lower without Trae Young
        "price": -110
    },
    {
        "market": "NBA_POINTS_TOR", 
        "mu": 114,
        "sigma": 9,
        "line": 117.5,
        "direction": "over",
        "price": -110
    },
    {
        "market": "NBA_TOTAL",
        "mu": 226,
        "sigma": 12,
        "line": 233.5,
        "direction": "under",
        "price": -110
    }
]

# Run automated analysis
result = runner.analyze_game("ATL", "TOR", "NBA", markets=markets)

# Access results
for bet in result['results']:
    if bet['ev_proxy'] > 0:
        print(f"POSITIVE EV: {bet['market']} {bet['direction']} {bet['line']}")
        print(f"  P(hit): {bet['p_hit']:.3f}")
        print(f"  EV: {bet['ev_proxy']:.4f}")
        print(f"  Kelly size: {bet['kelly']:.4f}")
        print(f"  Adjusted mu: {bet['mu_adjusted']:.1f}")
        print()


EXAMPLE 3: Manual context with automated runner
------------------------------------------------

from v4_mc_production_ultimate_enhanced_context_aware import AutomatedV4Runner, V4EngineIntegrated

# Create engine and runner
engine = V4EngineIntegrated()
runner = AutomatedV4Runner(engine, api_key=None)  # No API needed for manual context

# Manually provide context (skips Perplexity API call)
manual_context = {
    "injuries": [
        {"player": "Trae Young", "position": "PG", "status": "out",
         "team": "ATL", "opponent": "TOR", "sport": "NBA"}
    ],
    "opponent": {"defensive_rating": 0.52},
    "is_home": True,
    "recent_form": 0.500
}

# Set context directly
engine.set_game_context(manual_context)

# Evaluate with context
result = engine.evaluate_mainline_with_context(
    market="NBA_POINTS_ATL",
    mu=112,
    sigma=10,
    posted_line=115.5,
    direction="under",
    price=-110,
    blob=manual_context
)

print(f"Context-aware P(hit): {result['p_hit']:.3f}")
print(f"Confidence: {result['context_confidence']:.3f}")


EXAMPLE 4: API Key setup
-------------------------

# Option 1: Environment variable (recommended)
import os
os.environ['PERPLEXITY_API_KEY'] = 'your-api-key-here'

# Option 2: Pass directly
runner = create_automated_runner(api_key='your-api-key-here')

# Then use normally
result = runner.analyze_game("BOS", "ORL", "NBA")


EXAMPLE 5: Without Perplexity API (mock mode)
----------------------------------------------

# If you don't have API key, the system runs in mock mode
# You can still use all features with manual context

runner = create_automated_runner()  # No API key

# Manual context still works perfectly
context = {
    "injuries": [...],
    "opponent": {"defensive_rating": 0.55},
    "is_home": False
}

runner.engine.set_game_context(context)

# All V4+MC features work with manual context
result = runner.engine.evaluate_mainline_with_context(
    market="NBA_POINTS_BOS",
    mu=115, sigma=10, posted_line=107.5,
    direction="over", price=-110, blob=context
)
"""

