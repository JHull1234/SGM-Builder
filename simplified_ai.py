# Simplified AI SGM Picker - Working without heavy ML dependencies
# Professional-grade betting intelligence

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
import itertools
from scipy import stats
import uuid
import asyncio
import json

class SimplifiedMLPredictor:
    """Simplified ML predictor using statistical methods"""
    
    def __init__(self):
        self.prediction_weights = {
            "season_avg": 0.4,
            "recent_form": 0.3,
            "venue_performance": 0.2,
            "weather_adjustment": 0.05,
            "opponent_adjustment": 0.05
        }
    
    def predict_performance(self, player_data: Dict, match_context: Dict) -> Dict:
        """Predict player performance using weighted statistical model"""
        
        predictions = {}
        confidence_scores = {}
        
        for stat_type in ["disposals", "goals", "marks", "tackles"]:
            # Base prediction from season average
            season_avg = player_data.get(f"avg_{stat_type}", 0)
            
            # Recent form adjustment
            recent_form_factor = self._get_recent_form_factor(player_data.get("name", ""), stat_type)
            
            # Venue adjustment
            venue_factor = self._get_venue_factor(player_data, match_context.get("venue", "MCG"), stat_type)
            
            # Weather adjustment
            weather_factor = self._get_weather_factor(match_context.get("weather", {}), stat_type)
            
            # Opponent adjustment
            opponent_factor = self._get_opponent_factor(match_context.get("opponent_team", ""), stat_type)
            
            # Calculate weighted prediction
            base_pred = season_avg * self.prediction_weights["season_avg"]
            recent_pred = season_avg * recent_form_factor * self.prediction_weights["recent_form"]
            venue_pred = season_avg * venue_factor * self.prediction_weights["venue_performance"]
            weather_adj = season_avg * weather_factor * self.prediction_weights["weather_adjustment"]
            opponent_adj = season_avg * opponent_factor * self.prediction_weights["opponent_adjustment"]
            
            prediction = base_pred + recent_pred + venue_pred + weather_adj + opponent_adj
            predictions[stat_type] = max(0, prediction)
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(
                player_data, stat_type, recent_form_factor, venue_factor
            )
            confidence_scores[stat_type] = confidence
        
        return {
            "predictions": predictions,
            "confidence": confidence_scores,
            "model_agreement": {stat: 0.1 for stat in predictions.keys()}
        }
    
    def _get_recent_form_factor(self, player_name: str, stat_type: str) -> float:
        """Get recent form factor for player"""
        recent_form_data = {
            "Clayton Oliver": {"disposals": 1.08, "goals": 1.15, "marks": 1.02, "tackles": 1.05},
            "Christian Petracca": {"disposals": 0.98, "goals": 1.12, "marks": 1.08, "tackles": 0.95},
            "Marcus Bontempelli": {"disposals": 1.05, "goals": 1.18, "marks": 1.12, "tackles": 1.02},
            "Jeremy Cameron": {"disposals": 1.02, "goals": 1.22, "marks": 1.08, "tackles": 0.98}
        }
        
        return recent_form_data.get(player_name, {}).get(stat_type, 1.0)
    
    def _get_venue_factor(self, player_data: Dict, venue: str, stat_type: str) -> float:
        """Get venue performance factor"""
        venue_performance = player_data.get("venue_performance", {})
        
        if venue in venue_performance:
            venue_stat = venue_performance[venue].get(stat_type.replace("avg_", ""), 0)
            season_avg = player_data.get(f"avg_{stat_type}", 0)
            
            if season_avg > 0:
                return venue_stat / season_avg
        
        return 1.0
    
    def _get_weather_factor(self, weather: Dict, stat_type: str) -> float:
        """Get weather impact factor"""
        factor = 0.0
        
        wind_speed = weather.get("wind_speed", 10)
        precipitation = weather.get("precipitation", 0)
        temperature = weather.get("temperature", 20)
        
        if wind_speed > 20:
            if stat_type == "goals":
                factor -= 0.15
            elif stat_type == "marks":
                factor -= 0.10
            elif stat_type == "disposals":
                factor -= 0.05
        
        if precipitation > 1:
            if stat_type in ["disposals", "marks"]:
                factor -= 0.12
            elif stat_type == "goals":
                factor -= 0.18
        
        if temperature < 10 or temperature > 30:
            factor -= 0.03
        
        return factor
    
    def _get_opponent_factor(self, opponent_team: str, stat_type: str) -> float:
        """Get opponent defensive impact factor"""
        defensive_impact = {
            "Brisbane": {"disposals": -0.05, "goals": -0.08, "marks": -0.03, "tackles": 0.02},
            "Melbourne": {"disposals": -0.08, "goals": -0.12, "marks": -0.05, "tackles": 0.05},
            "Fremantle": {"disposals": -0.12, "goals": -0.15, "marks": -0.08, "tackles": 0.08},
            "Sydney": {"disposals": -0.10, "goals": -0.13, "marks": -0.06, "tackles": 0.06}
        }
        
        return defensive_impact.get(opponent_team, {}).get(stat_type, 0.0)
    
    def _calculate_prediction_confidence(self, player_data: Dict, stat_type: str, 
                                       recent_form_factor: float, venue_factor: float) -> float:
        """Calculate confidence in prediction"""
        
        base_confidence = 0.7
        
        games_played = player_data.get("games_played", 0)
        if games_played > 20:
            base_confidence += 0.1
        elif games_played < 10:
            base_confidence -= 0.1
        
        if 0.9 <= recent_form_factor <= 1.1:
            base_confidence += 0.05
        elif recent_form_factor < 0.8 or recent_form_factor > 1.2:
            base_confidence -= 0.1
        
        if venue_factor != 1.0:
            base_confidence += 0.05
        
        return max(0.3, min(0.95, base_confidence))

class SimplifiedSGMPicker:
    """Simplified automated SGM picker"""
    
    def __init__(self, ml_predictor: SimplifiedMLPredictor):
        self.ml_predictor = ml_predictor
        self.min_players = 2
        self.max_players = 4
        self.stat_types = ["disposals", "goals", "marks", "tackles"]
    
    async def recommend_sgm(self, target_odds: float, match_context: Dict, available_players: List[Dict]) -> Dict:
        """Recommend optimal SGM based on target odds"""
        
        print(f"🎯 AI analyzing {len(available_players)} players for target odds: {target_odds}")
        
        # Get predictions for all players
        player_predictions = []
        for player in available_players:
            prediction = self.ml_predictor.predict_performance(player, match_context)
            player_predictions.append({
                "player": player,
                "predictions": prediction["predictions"],
                "confidence": prediction["confidence"]
            })
        
        # Generate SGM combinations
        best_sgms = []
        combinations_analyzed = 0
        
        # Limit combinations for performance
        for num_players in range(self.min_players, min(self.max_players + 1, 4)):
            for player_combo in itertools.combinations(player_predictions[:8], num_players):
                
                sgm_combinations = self._generate_stat_combinations(player_combo)
                
                for sgm in sgm_combinations[:50]:
                    combinations_analyzed += 1
                    sgm_analysis = await self._analyze_sgm_combination(sgm, match_context, target_odds)
                    
                    if sgm_analysis["meets_criteria"]:
                        best_sgms.append(sgm_analysis)
                    
                    if len(best_sgms) >= 20:
                        break
                
                if len(best_sgms) >= 20:
                    break
        
        # Sort by value rating and confidence
        best_sgms.sort(key=lambda x: (x["value_rating"], x["confidence_score"]), reverse=True)
        
        return {
            "target_odds": target_odds,
            "recommendations": best_sgms[:5],
            "total_combinations_analyzed": combinations_analyzed,
            "analysis_timestamp": datetime.now().isoformat(),
            "match_context": match_context
        }
    
    def _generate_stat_combinations(self, player_combo: Tuple) -> List[Dict]:
        """Generate stat combinations for SGM"""
        
        combinations = []
        
        for player_data in player_combo:
            player = player_data["player"]
            predictions = player_data["predictions"]
            
            player_options = []
            
            key_stats = ["disposals", "goals"] if player["position"] == "Forward" else ["disposals", "marks"]
            
            for stat_type in key_stats:
                predicted_value = predictions.get(stat_type, 0)
                
                if predicted_value > 5:
                    target_options = self._generate_target_lines(stat_type, predicted_value)
                    
                    for target, implied_prob in target_options[:2]:
                        player_options.append({
                            "player": player["name"],
                            "stat_type": stat_type,
                            "target": target,
                            "predicted": predicted_value,
                            "implied_probability": implied_prob
                        })
            
            if player_options:
                combinations.append(player_options)
        
        sgm_combinations = []
        if len(combinations) >= 2:
            for combo in itertools.product(*combinations):
                if len(combo) <= 4:
                    sgm_combinations.append(list(combo))
        
        return sgm_combinations[:20]
    
    def _generate_target_lines(self, stat_type: str, predicted_value: float) -> List[Tuple[float, float]]:
        """Generate realistic target lines for a stat"""
        
        target_lines = []
        
        if stat_type == "disposals":
            lines = [20.5, 25.5, 30.5, 35.5]
        elif stat_type == "goals":
            lines = [0.5, 1.5, 2.5, 3.5]
        elif stat_type == "marks":
            lines = [4.5, 6.5, 8.5]
        elif stat_type == "tackles":
            lines = [4.5, 6.5, 8.5]
        else:
            return []
        
        for line in lines:
            if abs(line - predicted_value) < predicted_value * 0.5:
                prob = self._calculate_over_probability(predicted_value, line, stat_type)
                if 0.2 <= prob <= 0.85:
                    target_lines.append((line, prob))
        
        return target_lines
    
    def _calculate_over_probability(self, predicted: float, line: float, stat_type: str) -> float:
        """Calculate probability of going over the line"""
        
        std_devs = {"disposals": 4.0, "goals": 1.0, "marks": 1.5, "tackles": 1.2}
        std_dev = std_devs.get(stat_type, 2.0)
        
        if std_dev > 0:
            z_score = (line - predicted) / std_dev
            probability = 1 - stats.norm.cdf(z_score)
        else:
            probability = 0.5
        
        return max(0.05, min(0.95, probability))
    
    async def _analyze_sgm_combination(self, sgm: List[Dict], match_context: Dict, target_odds: float) -> Dict:
        """Analyze a specific SGM combination"""
        
        individual_probs = [outcome["implied_probability"] for outcome in sgm]
        naive_prob = np.prod(individual_probs)
        
        correlation_adjustment = self._calculate_correlation_adjustment(sgm)
        adjusted_prob = naive_prob * (1 + correlation_adjustment)
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))
        
        implied_odds = 1 / adjusted_prob
        value_rating = (implied_odds - target_odds) / target_odds
        
        confidence_score = np.mean([0.7, 0.8, 0.75])
        
        meets_criteria = (
            0.8 <= implied_odds <= target_odds * 1.3 and
            value_rating > -0.15 and
            confidence_score > 0.5
        )
        
        return {
            "sgm_outcomes": sgm,
            "individual_probabilities": individual_probs,
            "naive_probability": round(naive_prob, 4),
            "correlation_adjustment": round(correlation_adjustment, 4),
            "adjusted_probability": round(adjusted_prob, 4),
            "implied_odds": round(implied_odds, 2),
            "target_odds": target_odds,
            "value_rating": round(value_rating, 4),
            "confidence_score": round(confidence_score, 3),
            "meets_criteria": meets_criteria,
            "recommendation": self._generate_recommendation(value_rating, confidence_score, implied_odds)
        }
    
    def _calculate_correlation_adjustment(self, sgm: List[Dict]) -> float:
        """Calculate correlation adjustments"""
        adjustment = 0.0
        
        players = [outcome["player"] for outcome in sgm]
        for player in set(players):
            player_outcomes = [o for o in sgm if o["player"] == player]
            if len(player_outcomes) > 1:
                adjustment -= 0.15 * (len(player_outcomes) - 1)
        
        teammate_pairs = self._get_teammate_pairs(players)
        adjustment += len(teammate_pairs) * 0.05
        
        return adjustment
    
    def _get_teammate_pairs(self, players: List[str]) -> List[Tuple]:
        """Get teammate pairs"""
        teams = {
            "Clayton Oliver": "Melbourne", "Christian Petracca": "Melbourne",
            "Marcus Bontempelli": "Western Bulldogs", "Adam Treloar": "Western Bulldogs",
            "Jeremy Cameron": "Geelong", "Tom Hawkins": "Geelong"
        }
        
        pairs = []
        for i, p1 in enumerate(players):
            for p2 in players[i+1:]:
                if teams.get(p1) == teams.get(p2) and teams.get(p1):
                    pairs.append((p1, p2))
        
        return pairs
    
    def _generate_recommendation(self, value_rating: float, confidence: float, implied_odds: float) -> str:
        """Generate recommendation"""
        if value_rating > 0.15 and confidence > 0.7:
            return f"🔥 EXCELLENT SGM - Strong value at {implied_odds:.2f}"
        elif value_rating > 0.05 and confidence > 0.6:
            return f"✅ GOOD SGM - Solid value at {implied_odds:.2f}"
        elif value_rating > -0.05:
            return f"⚠️ MARGINAL SGM - Fair value, proceed with caution"
        else:
            return f"❌ AVOID - Poor value"