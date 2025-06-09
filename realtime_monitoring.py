# Real-Time Odds Monitoring & Advanced Edge Detection System
# Professional-grade betting intelligence

import asyncio
import httpx
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
import uuid

class RealTimeOddsMonitor:
    """Real-time odds monitoring across multiple bookmakers"""
    
    def __init__(self, odds_api_key: str):
        self.odds_api_key = odds_api_key
        self.monitored_bets = {}
        self.odds_history = {}
        self.alerts = []
        
    async def start_monitoring(self, bet_configurations: List[Dict]):
        """Start monitoring specific bets for value opportunities"""
        for config in bet_configurations:
            bet_id = str(uuid.uuid4())
            self.monitored_bets[bet_id] = {
                "config": config,
                "created_at": datetime.now(),
                "alerts_sent": 0,
                "best_odds_seen": None,
                "current_value_rating": None
            }
        
        # Start monitoring loop
        await self._monitoring_loop()
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                await self._check_all_monitored_bets()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_all_monitored_bets(self):
        """Check all monitored bets for value opportunities"""
        async with httpx.AsyncClient() as client:
            try:
                # Get current AFL odds
                response = await client.get(
                    "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds",
                    params={
                        "apiKey": self.odds_api_key,
                        "regions": "au",
                        "markets": "h2h,spreads,totals,player_props",
                        "oddsFormat": "decimal"
                    },
                    timeout=30
                )
                response.raise_for_status()
                current_odds = response.json()
                
                # Process each monitored bet
                for bet_id, bet_data in self.monitored_bets.items():
                    await self._analyze_bet_odds(bet_id, bet_data, current_odds)
                    
            except Exception as e:
                print(f"Odds API error: {e}")
    
    async def _analyze_bet_odds(self, bet_id: str, bet_data: Dict, market_odds: List[Dict]):
        """Analyze specific bet against current market"""
        config = bet_data["config"]
        
        # Find relevant odds for this bet
        relevant_odds = self._extract_relevant_odds(config, market_odds)
        
        if not relevant_odds:
            return
        
        # Calculate value metrics
        value_analysis = self._calculate_value_metrics(config, relevant_odds)
        
        # Update bet tracking
        bet_data["current_value_rating"] = value_analysis["value_rating"]
        
        # Store odds history
        timestamp = datetime.now().isoformat()
        if bet_id not in self.odds_history:
            self.odds_history[bet_id] = []
        
        self.odds_history[bet_id].append({
            "timestamp": timestamp,
            "odds_data": relevant_odds,
            "value_rating": value_analysis["value_rating"],
            "best_bookmaker": value_analysis["best_bookmaker"]
        })
        
        # Check for alerts
        await self._check_value_alerts(bet_id, bet_data, value_analysis)
    
    def _extract_relevant_odds(self, config: Dict, market_odds: List[Dict]) -> Dict:
        """Extract odds relevant to the monitored bet"""
        bet_type = config.get("type", "")
        player_name = config.get("player", "")
        
        # For demo, return mock odds structure
        return {
            "sportsbet": {"odds": 2.40, "last_update": datetime.now()},
            "tab": {"odds": 2.35, "last_update": datetime.now()},
            "betfair": {"odds": 2.45, "last_update": datetime.now()},
            "bet365": {"odds": 2.50, "last_update": datetime.now()},  # Best odds
            "ladbrokes": {"odds": 2.38, "last_update": datetime.now()}
        }
    
    def _calculate_value_metrics(self, config: Dict, odds_data: Dict) -> Dict:
        """Calculate value metrics across all bookmakers"""
        if not odds_data:
            return {"value_rating": 0, "best_bookmaker": None}
        
        # Find best odds
        best_bookmaker = max(odds_data.keys(), key=lambda k: odds_data[k]["odds"])
        best_odds = odds_data[best_bookmaker]["odds"]
        
        # Calculate our predicted probability (from config)
        predicted_prob = config.get("predicted_probability", 0.5)
        
        # Calculate value rating
        implied_prob = 1 / best_odds
        value_rating = (predicted_prob - implied_prob) / implied_prob
        
        return {
            "value_rating": value_rating,
            "best_bookmaker": best_bookmaker,
            "best_odds": best_odds,
            "line_shopping_edge": self._calculate_line_shopping_edge(odds_data),
            "market_efficiency": self._calculate_market_efficiency(odds_data)
        }
    
    def _calculate_line_shopping_edge(self, odds_data: Dict) -> Dict:
        """Calculate advantage from line shopping across bookmakers"""
        odds_values = [data["odds"] for data in odds_data.values()]
        
        best_odds = max(odds_values)
        worst_odds = min(odds_values)
        average_odds = statistics.mean(odds_values)
        
        line_shopping_advantage = ((best_odds - worst_odds) / worst_odds) * 100
        
        return {
            "best_odds": best_odds,
            "worst_odds": worst_odds,
            "average_odds": round(average_odds, 2),
            "shopping_advantage": round(line_shopping_advantage, 2),
            "recommended_book": max(odds_data.keys(), key=lambda k: odds_data[k]["odds"])
        }
    
    def _calculate_market_efficiency(self, odds_data: Dict) -> str:
        """Assess market efficiency based on odds spread"""
        odds_values = [data["odds"] for data in odds_data.values()]
        spread = max(odds_values) - min(odds_values)
        
        if spread > 0.20:
            return "Inefficient - Large odds spread"
        elif spread > 0.10:
            return "Moderate - Some inefficiency"
        else:
            return "Efficient - Tight market"
    
    async def _check_value_alerts(self, bet_id: str, bet_data: Dict, value_analysis: Dict):
        """Check if any alerts should be triggered"""
        value_rating = value_analysis["value_rating"]
        config = bet_data["config"]
        
        alerts_to_send = []
        
        # Value threshold alerts
        if value_rating > 0.15 and bet_data["alerts_sent"] == 0:
            alerts_to_send.append({
                "type": "EXCELLENT_VALUE",
                "message": f"🔥 EXCELLENT VALUE DETECTED: {config['description']} @ {value_analysis['best_bookmaker']} ({value_analysis['best_odds']})",
                "urgency": "HIGH",
                "value_rating": value_rating
            })
        
        # Line shopping alerts
        shopping_advantage = value_analysis["line_shopping_edge"]["shopping_advantage"]
        if shopping_advantage > 5:
            alerts_to_send.append({
                "type": "LINE_SHOPPING",
                "message": f"💰 LINE SHOPPING OPPORTUNITY: {shopping_advantage:.1f}% better odds at {value_analysis['best_bookmaker']}",
                "urgency": "MEDIUM"
            })
        
        # Store alerts
        for alert in alerts_to_send:
            alert["bet_id"] = bet_id
            alert["timestamp"] = datetime.now()
            self.alerts.append(alert)
            bet_data["alerts_sent"] += 1
    
    def get_active_alerts(self, max_age_hours: int = 2) -> List[Dict]:
        """Get active alerts within specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        return [
            alert for alert in self.alerts 
            if alert["timestamp"] > cutoff_time
        ]
    
    def get_odds_history(self, bet_id: str) -> List[Dict]:
        """Get odds history for specific bet"""
        return self.odds_history.get(bet_id, [])

class SteamMoveDetector:
    """Detect 'steam moves' - when sharp money hits the market"""
    
    def __init__(self):
        self.odds_snapshots = {}
        self.steam_moves = []
    
    async def detect_steam_moves(self, current_odds: Dict, previous_odds: Dict) -> List[Dict]:
        """Detect steam moves based on odds movement patterns"""
        steam_moves = []
        
        for bet_key in current_odds:
            if bet_key in previous_odds:
                current = current_odds[bet_key]
                previous = previous_odds[bet_key]
                
                # Calculate odds movement
                movement = self._calculate_odds_movement(current, previous)
                
                # Check for steam move indicators
                if self._is_steam_move(movement):
                    steam_moves.append({
                        "bet": bet_key,
                        "movement": movement,
                        "detected_at": datetime.now(),
                        "severity": self._classify_steam_severity(movement),
                        "recommendation": self._get_steam_recommendation(movement)
                    })
        
        self.steam_moves.extend(steam_moves)
        return steam_moves
    
    def _calculate_odds_movement(self, current: Dict, previous: Dict) -> Dict:
        """Calculate odds movement metrics"""
        # Mock calculation - in production would analyze multiple bookmakers
        current_avg = 2.40  # Mock current average
        previous_avg = 2.60  # Mock previous average
        
        movement_pct = ((current_avg - previous_avg) / previous_avg) * 100
        
        return {
            "percentage_change": movement_pct,
            "direction": "shortening" if movement_pct < 0 else "drifting",
            "magnitude": abs(movement_pct),
            "speed": "fast",  # Could calculate based on time window
            "volume_indicator": "high"  # Would need volume data
        }
    
    def _is_steam_move(self, movement: Dict) -> bool:
        """Determine if movement qualifies as steam move"""
        # Steam move criteria:
        # 1. Significant odds movement (>5%)
        # 2. Fast movement (within short time period)
        # 3. Against public money direction
        
        return (
            movement["magnitude"] > 5 and
            movement["speed"] == "fast" and
            movement["volume_indicator"] == "high"
        )
    
    def _classify_steam_severity(self, movement: Dict) -> str:
        """Classify severity of steam move"""
        magnitude = movement["magnitude"]
        
        if magnitude > 15:
            return "EXTREME"
        elif magnitude > 10:
            return "STRONG"
        elif magnitude > 5:
            return "MODERATE"
        else:
            return "MILD"
    
    def _get_steam_recommendation(self, movement: Dict) -> str:
        """Get recommendation based on steam move"""
        if movement["direction"] == "shortening":
            return "⚠️ SHARP MONEY DETECTED - Consider following the move"
        else:
            return "📈 PUBLIC MONEY FADE - Sharp money going other way"

class AdvancedValueDetector:
    """Advanced value detection using multiple market inefficiencies"""
    
    @staticmethod
    def detect_correlation_mispricing(sgm_outcomes: List[Dict], market_odds: float) -> Dict:
        """Detect when market has mispriced correlation in SGM"""
        
        # Calculate true correlation-adjusted probability
        from advanced_analytics import TeammateSymergyAnalyzer
        synergy_analysis = TeammateSymergyAnalyzer.calculate_synergy_impact(sgm_outcomes)
        
        # Get individual outcome probabilities
        individual_probs = []
        for outcome in sgm_outcomes:
            # Mock individual probability - in production get from player models
            individual_probs.append(0.65)  # 65% chance for each outcome
        
        # Calculate naive probability (market assumption)
        naive_prob = 1.0
        for prob in individual_probs:
            naive_prob *= prob
        
        # Calculate correlation-adjusted probability
        correlation_adjustment = synergy_analysis["total_synergy_impact"]
        adjusted_prob = naive_prob * (1 + correlation_adjustment)
        
        # Compare to market probability
        market_prob = 1 / market_odds
        
        return {
            "naive_probability": round(naive_prob, 3),
            "correlation_adjusted": round(adjusted_prob, 3),
            "market_probability": round(market_prob, 3),
            "correlation_edge": round((adjusted_prob - market_prob) / market_prob, 3),
            "verdict": "POSITIVE EDGE" if adjusted_prob > market_prob else "NEGATIVE EDGE",
            "confidence": "High" if abs(adjusted_prob - market_prob) > 0.05 else "Medium"
        }
    
    @staticmethod
    def detect_recency_bias(player_stats: Dict, market_odds: float) -> Dict:
        """Detect when market over-reacts to recent performance"""
        
        # Get recent vs long-term performance
        recent_avg = player_stats.get("recent_avg", 0)
        season_avg = player_stats.get("season_avg", 0)
        
        if season_avg == 0:
            return {"error": "Insufficient data"}
        
        # Calculate performance deviation
        deviation = (recent_avg - season_avg) / season_avg
        
        # Estimate market adjustment to recent form
        estimated_market_adjustment = deviation * 0.8  # Market typically over-weights recent form
        
        # Calculate what odds should be based on true talent
        true_talent_prob = 0.60  # Mock baseline probability
        market_prob = 1 / market_odds
        
        return {
            "recent_deviation": round(deviation * 100, 1),
            "estimated_market_bias": round(estimated_market_adjustment * 100, 1),
            "true_talent_probability": true_talent_prob,
            "market_probability": round(market_prob, 3),
            "recency_edge": round((true_talent_prob - market_prob) / market_prob, 3),
            "verdict": "FADE RECENT FORM" if deviation > 0.15 and market_prob < true_talent_prob else "FOLLOW FORM"
        }
    
    @staticmethod
    def detect_weather_mispricing(weather_impact: Dict, market_odds: float) -> Dict:
        """Detect when market hasn't properly priced weather impact"""
        
        total_weather_impact = weather_impact.get("total_impact", 0)
        
        # Most recreational markets don't adjust for weather
        market_weather_adjustment = 0.02  # Market typically under-adjusts
        true_weather_impact = total_weather_impact
        
        weather_edge = true_weather_impact - market_weather_adjustment
        
        return {
            "true_weather_impact": round(true_weather_impact * 100, 1),
            "estimated_market_adjustment": round(market_weather_adjustment * 100, 1),
            "weather_edge": round(weather_edge * 100, 1),
            "verdict": "WEATHER EDGE" if abs(weather_edge) > 0.03 else "NO EDGE",
            "recommendation": "AVOID" if weather_edge < -0.05 else "BET" if weather_edge > 0.05 else "NEUTRAL"
        }

class PortfolioOptimizer:
    """Optimize betting portfolio for maximum returns"""
    
    @staticmethod
    def optimize_bankroll_allocation(bets: List[Dict], total_bankroll: float) -> Dict:
        """Optimize bankroll allocation across multiple bets using Kelly Criterion"""
        
        total_allocation = 0
        allocations = {}
        
        for bet in bets:
            # Kelly Criterion: f = (bp - q) / b
            # where: f = fraction to bet, b = odds-1, p = probability, q = 1-p
            
            probability = bet.get("probability", 0.5)
            odds = bet.get("odds", 2.0)
            
            b = odds - 1  # Net odds
            p = probability
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Apply Kelly multiplier for safety (typically 0.25x to 0.5x)
            kelly_multiplier = 0.25  # Conservative
            safe_fraction = max(0, kelly_fraction * kelly_multiplier)
            
            # Cap maximum bet size
            max_bet_fraction = 0.05  # Never bet more than 5% on single bet
            final_fraction = min(safe_fraction, max_bet_fraction)
            
            allocations[bet["id"]] = {
                "kelly_fraction": round(kelly_fraction, 4),
                "safe_fraction": round(final_fraction, 4),
                "bet_amount": round(total_bankroll * final_fraction, 2),
                "expected_value": round((probability * (odds - 1) - (1 - probability)) * final_fraction, 4)
            }
            
            total_allocation += final_fraction
        
        return {
            "individual_allocations": allocations,
            "total_allocation": round(total_allocation, 4),
            "remaining_bankroll": round((1 - total_allocation) * total_bankroll, 2),
            "portfolio_ev": round(sum([alloc["expected_value"] for alloc in allocations.values()]), 4),
            "risk_assessment": "Conservative" if total_allocation < 0.10 else "Moderate" if total_allocation < 0.20 else "Aggressive"
        }
    
    @staticmethod
    def calculate_portfolio_correlations(bets: List[Dict]) -> Dict:
        """Calculate correlations between different bets in portfolio"""
        
        correlations = {}
        risk_warnings = []
        
        for i, bet1 in enumerate(bets):
            for j, bet2 in enumerate(bets[i+1:], i+1):
                
                # Check for same game correlations
                if bet1.get("game_id") == bet2.get("game_id"):
                    correlation = 0.3  # Same game bets are correlated
                    correlations[f"{bet1['id']}-{bet2['id']}"] = {
                        "correlation": correlation,
                        "type": "same_game",
                        "risk_level": "Medium"
                    }
                
                # Check for same player correlations
                elif bet1.get("player") == bet2.get("player"):
                    correlation = 0.7  # Same player bets highly correlated
                    correlations[f"{bet1['id']}-{bet2['id']}"] = {
                        "correlation": correlation,
                        "type": "same_player",
                        "risk_level": "High"
                    }
                    risk_warnings.append(f"High correlation risk: {bet1['player']} bets")
        
        return {
            "correlations": correlations,
            "risk_warnings": risk_warnings,
            "portfolio_risk": "High" if len(risk_warnings) > 2 else "Medium" if len(risk_warnings) > 0 else "Low"
        }

class LiveGameTracker:
    """Track live game data for in-play adjustments"""
    
    def __init__(self):
        self.live_games = {}
        self.quarter_stats = {}
    
    async def track_live_performance(self, game_id: str, player_stats: Dict):
        """Track live player performance during game"""
        
        if game_id not in self.live_games:
            self.live_games[game_id] = {
                "start_time": datetime.now(),
                "quarter": 1,
                "player_stats": {}
            }
        
        game_data = self.live_games[game_id]
        
        # Update player stats
        for player, stats in player_stats.items():
            if player not in game_data["player_stats"]:
                game_data["player_stats"][player] = {
                    "disposals": 0,
                    "goals": 0,
                    "marks": 0,
                    "quarters": {}
                }
            
            # Update cumulative stats
            game_data["player_stats"][player].update(stats)
            
            # Track quarter-by-quarter performance
            current_quarter = game_data["quarter"]
            game_data["player_stats"][player]["quarters"][current_quarter] = stats
    
    def predict_final_stats(self, game_id: str, player: str) -> Dict:
        """Predict final stats based on current performance"""
        
        if game_id not in self.live_games or player not in self.live_games[game_id]["player_stats"]:
            return {"error": "No data available"}
        
        game_data = self.live_games[game_id]
        player_data = game_data["player_stats"][player]
        current_quarter = game_data["quarter"]
        
        if current_quarter == 0:
            return {"error": "Game not started"}
        
        # Simple linear projection
        current_disposals = player_data.get("disposals", 0)
        projected_final = (current_disposals / current_quarter) * 4
        
        # Adjust for typical quarter-by-quarter patterns
        quarter_weightings = [1.0, 1.1, 0.9, 0.8]  # Players often fade in 4th quarter
        weighted_projection = current_disposals * sum(quarter_weightings[current_quarter:]) / current_quarter
        
        return {
            "current_stats": player_data,
            "linear_projection": round(projected_final, 1),
            "weighted_projection": round(weighted_projection, 1),
            "confidence": "High" if current_quarter >= 2 else "Medium",
            "recommendation": self._get_live_recommendation(current_disposals, weighted_projection)
        }
    
    def _get_live_recommendation(self, current: float, projected: float) -> str:
        """Get live betting recommendation"""
        if projected > current * 1.5:
            return "🔥 STRONG LIVE BET - Tracking well ahead of pace"
        elif projected > current * 1.2:
            return "✅ LIVE BET - Tracking above pace"
        elif projected < current * 0.8:
            return "❌ AVOID - Tracking below pace"
        else:
            return "⚠️ MONITOR - On pace but close"