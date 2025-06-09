# Advanced Machine Learning & Automated SGM Picker
# The most sophisticated AFL betting AI system

import numpy as np
import pandas as pd
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ ML libraries not available - using fallback predictions")
    ML_AVAILABLE = False

import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

import statistics
import itertools
from scipy import stats
import uuid

class MachineLearningPredictor:
    """Advanced ML models for AFL player performance prediction"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_accuracy = {}

    def create_features(self, player_data: Dict, match_context: Dict) -> np.array:
        """Create comprehensive feature set for ML models"""

        features = []

        # Player performance features
        features.extend([
            player_data.get("avg_disposals", 0),
            player_data.get("avg_goals", 0),
            player_data.get("avg_marks", 0),
            player_data.get("avg_tackles", 0),
            player_data.get("avg_kicks", 0),
            player_data.get("avg_handballs", 0),
            player_data.get("games_played", 0)
        ])

        # Recent form features (last 5 games)
        recent_games = player_data.get("recent_form", {}).get("last_5_games", [])
        if recent_games:
            recent_disposals = [g.get("disposals", 0) for g in recent_games]
            recent_goals = [g.get("goals", 0) for g in recent_games]

            features.extend([
                np.mean(recent_disposals),
                np.std(recent_disposals),
                np.mean(recent_goals),
                np.std(recent_goals),
                len(recent_games)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])

        # Venue features
        venue_performance = player_data.get("venue_performance", {})
        current_venue = match_context.get("venue", "MCG")
        if current_venue in venue_performance:
            features.extend([
                venue_performance[current_venue].get("disposals", player_data.get("avg_disposals", 0)),
                venue_performance[current_venue].get("goals", player_data.get("avg_goals", 0))
            ])
        else:
            features.extend([player_data.get("avg_disposals", 0), player_data.get("avg_goals", 0)])

        # Weather features
        weather = match_context.get("weather", {})
        features.extend([
            weather.get("temperature", 20),
            weather.get("wind_speed", 10),
            weather.get("humidity", 60),
            weather.get("precipitation", 0),
            1 if weather.get("conditions", "").lower() in ["rain", "storm"] else 0
        ])

        # Opposition features
        opponent_team = match_context.get("opponent_team", "Unknown")
        opponent_defense = match_context.get("opponent_defense", {})
        features.extend([
            opponent_defense.get("midfielder_disposals_allowed", 125),
            opponent_defense.get("forward_goals_allowed", 8.5),
            opponent_defense.get("tackles_per_game", 65)
        ])

        # Injury/fitness features
        injury_impact = match_context.get("injury_impact", 0)
        features.extend([
            injury_impact,
            1 if injury_impact < -0.05 else 0  # Binary injury flag
        ])

        # Time-based features
        match_date = match_context.get("date", datetime.now())
        if isinstance(match_date, str):
            match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))

        features.extend([
            match_date.weekday(),  # Day of week
            match_date.month,      # Season time
            1 if match_date.weekday() in [5, 6] else 0  # Weekend flag
        ])

        return np.array(features).reshape(1, -1)

    def train_ensemble_models(self, training_data: List[Dict]):
        """Train ensemble of ML models for different stats"""

        stat_types = ["disposals", "goals", "marks", "tackles"]

        for stat_type in stat_types:
            print(f"Training models for {stat_type}...")

            # Prepare training data
            X_data = []
            y_data = []

            for record in training_data:
                features = self.create_features(record["player_data"], record["match_context"])
                target = record["actual_performance"].get(stat_type, 0)

                X_data.append(features[0])
                y_data.append(target)

            X = np.array(X_data)
            y = np.array(y_data)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[stat_type] = scaler

            # Train multiple models
            models = {
                "random_forest": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42
                ),
                "gradient_boost": GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                "neural_network": MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    max_iter=1000,
                    alpha=0.01,
                    random_state=42
                )
            }

            trained_models = {}
            model_scores = {}

            for model_name, model in models.items():
                # Train model
                model.fit(X_scaled, y)

                # Cross-validation score
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
                model_scores[model_name] = -cv_scores.mean()

                trained_models[model_name] = model

                print(f"  {model_name}: MAE = {model_scores[model_name]:.3f}")

            # Store best model and ensemble
            self.models[stat_type] = trained_models
            self.model_accuracy[stat_type] = model_scores

            # Feature importance (from Random Forest)
            if hasattr(trained_models["random_forest"], "feature_importances_"):
                self.feature_importance[stat_type] = trained_models["random_forest"].feature_importances_

    def predict_performance(self, player_data: Dict, match_context: Dict) -> Dict:
        """Predict player performance using ensemble models"""

        features = self.create_features(player_data, match_context)
        predictions = {}
        confidence_scores = {}

        for stat_type in ["disposals", "goals", "marks", "tackles"]:
            if stat_type in self.models:
                # Scale features
                X_scaled = self.scalers[stat_type].transform(features)

                # Get predictions from all models
                model_predictions = []
                for model_name, model in self.models[stat_type].items():
                    pred = model.predict(X_scaled)[0]
                    model_predictions.append(pred)

                # Ensemble prediction (weighted average based on model accuracy)
                weights = []
                for model_name in self.models[stat_type].keys():
                    # Higher accuracy = lower MAE = higher weight
                    mae = self.model_accuracy[stat_type][model_name]
                    weight = 1 / (mae + 0.1)  # Add small constant to avoid division by zero
                    weights.append(weight)

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Weighted ensemble prediction
                ensemble_pred = sum(pred * weight for pred, weight in zip(model_predictions, weights))
                predictions[stat_type] = max(0, ensemble_pred)  # Ensure non-negative

                # Confidence based on model agreement
                pred_std = np.std(model_predictions)
                confidence_scores[stat_type] = max(0.5, 1 - (pred_std / np.mean(model_predictions)))
            else:
                # Fallback to average if no model trained
                predictions[stat_type] = player_data.get(f"avg_{stat_type}", 0)
                confidence_scores[stat_type] = 0.5

        return {
            "predictions": predictions,
            "confidence": confidence_scores,
            "model_agreement": {
                stat: np.std([model.predict(self.scalers[stat].transform(features))[0]
                            for model in self.models[stat].values()])
                for stat in predictions.keys() if stat in self.models
            }
        }

class AutomatedSGMPicker:
    """Automated SGM picker based on target odds and advanced analytics"""

    def __init__(self, ml_predictor: MachineLearningPredictor):
        self.ml_predictor = ml_predictor
        self.min_players = 2
        self.max_players = 4
        self.stat_types = ["disposals", "goals", "marks", "tackles"]

    async def recommend_sgm(self, target_odds: float, match_context: Dict, available_players: List[Dict]) -> Dict:
        """Recommend optimal SGM based on target odds"""

        print(f"🎯 Finding optimal SGM for target odds: {target_odds}")

        # Get ML predictions for all players
        player_predictions = []
        for player in available_players:
            ml_prediction = self.ml_predictor.predict_performance(player, match_context)

            player_predictions.append({
                "player": player,
                "predictions": ml_prediction["predictions"],
                "confidence": ml_prediction["confidence"]
            })

        # Generate all possible SGM combinations
        best_sgms = []

        for num_players in range(self.min_players, min(self.max_players + 1, len(available_players) + 1)):
            for player_combo in itertools.combinations(player_predictions, num_players):

                # Generate stat combinations for each player
                sgm_combinations = self._generate_stat_combinations(player_combo)

                for sgm in sgm_combinations:
                    sgm_analysis = await self._analyze_sgm_combination(sgm, match_context, target_odds)

                    if sgm_analysis["meets_criteria"]:
                        best_sgms.append(sgm_analysis)

        # Sort by value rating and confidence
        best_sgms.sort(key=lambda x: (x["value_rating"], x["confidence_score"]), reverse=True)

        # Return top recommendations
        return {
            "target_odds": target_odds,
            "recommendations": best_sgms[:5],  # Top 5 recommendations
            "total_combinations_analyzed": len(best_sgms),
            "analysis_timestamp": datetime.now().isoformat(),
            "market_context": match_context
        }

    def _generate_stat_combinations(self, player_combo: Tuple) -> List[Dict]:
        """Generate stat combinations for SGM"""

        combinations = []

        for player_data in player_combo:
            player = player_data["player"]
            predictions = player_data["predictions"]

            # For each player, try different stat targets
            player_options = []

            for stat_type in self.stat_types:
                predicted_value = predictions.get(stat_type, 0)

                if predicted_value > 0:
                    # Generate different target lines
                    target_options = self._generate_target_lines(stat_type, predicted_value)

                    for target, implied_prob in target_options:
                        player_options.append({
                            "player": player["name"],
                            "stat_type": stat_type,
                            "target": target,
                            "predicted": predicted_value,
                            "implied_probability": implied_prob
                        })

            combinations.append(player_options)

        # Create all combinations across players
        sgm_combinations = []
        for combo in itertools.product(*combinations):
            sgm_combinations.append(list(combo))

        return sgm_combinations

    def _generate_target_lines(self, stat_type: str, predicted_value: float) -> List[Tuple[float, float]]:
        """Generate realistic target lines for a stat"""

        target_lines = []

        if stat_type == "disposals":
            # Common disposal lines
            lines = [15.5, 20.5, 25.5, 30.5, 35.5]
            for line in lines:
                if abs(line - predicted_value) < 10:  # Within reasonable range
                    prob = self._calculate_over_probability(predicted_value, line, stat_type)
                    target_lines.append((line, prob))

        elif stat_type == "goals":
            # Common goal lines
            lines = [0.5, 1.5, 2.5, 3.5]
            for line in lines:
                if line <= predicted_value + 2:  # Reasonable for goals
                    prob = self._calculate_over_probability(predicted_value, line, stat_type)
                    target_lines.append((line, prob))

        elif stat_type == "marks":
            # Common mark lines
            lines = [3.5, 5.5, 7.5, 9.5]
            for line in lines:
                if abs(line - predicted_value) < 5:
                    prob = self._calculate_over_probability(predicted_value, line, stat_type)
                    target_lines.append((line, prob))

        elif stat_type == "tackles":
            # Common tackle lines
            lines = [3.5, 5.5, 7.5, 9.5]
            for line in lines:
                if abs(line - predicted_value) < 4:
                    prob = self._calculate_over_probability(predicted_value, line, stat_type)
                    target_lines.append((line, prob))

        return target_lines

    def _calculate_over_probability(self, predicted: float, line: float, stat_type: str) -> float:
        """Calculate probability of going over the line"""

        # Estimate standard deviation based on stat type
        std_devs = {
            "disposals": 5.0,
            "goals": 1.2,
            "marks": 2.0,
            "tackles": 1.5
        }

        std_dev = std_devs.get(stat_type, 2.0)

        # Use normal distribution
        z_score = (line - predicted) / std_dev
        probability = 1 - stats.norm.cdf(z_score)

        return max(0.05, min(0.95, probability))  # Cap between 5% and 95%

    async def _analyze_sgm_combination(self, sgm: List[Dict], match_context: Dict, target_odds: float) -> Dict:
        """Analyze a specific SGM combination"""

        # Calculate individual probabilities
        individual_probs = [outcome["implied_probability"] for outcome in sgm]

        # Calculate naive probability (independence assumption)
        naive_prob = np.prod(individual_probs)

        # Apply correlation adjustments
        correlation_adjustment = self._calculate_sgm_correlations(sgm)
        adjusted_prob = naive_prob * (1 + correlation_adjustment)
        adjusted_prob = max(0.01, min(0.99, adjusted_prob))

        # Calculate implied odds
        implied_odds = 1 / adjusted_prob

        # Value analysis
        odds_difference = implied_odds - target_odds
        value_rating = odds_difference / target_odds

        # Confidence analysis
        confidence_factors = self._calculate_confidence_factors(sgm, match_context)
        overall_confidence = np.mean(list(confidence_factors.values()))

        # Check if meets criteria
        meets_criteria = (
            0.9 <= implied_odds <= target_odds * 1.2 and  # Within reasonable range of target
            value_rating > -0.1 and  # Not terrible value
            overall_confidence > 0.6  # Reasonable confidence
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
            "confidence_factors": confidence_factors,
            "confidence_score": round(overall_confidence, 3),
            "meets_criteria": meets_criteria,
            "recommendation": self._generate_sgm_recommendation(value_rating, overall_confidence, implied_odds, target_odds)
        }

    def _calculate_sgm_correlations(self, sgm: List[Dict]) -> float:
        """Calculate correlation adjustments for SGM"""

        correlation_adjustment = 0.0

        # Check for same player correlations
        players = [outcome["player"] for outcome in sgm]
        for player in set(players):
            player_outcomes = [o for o in sgm if o["player"] == player]
            if len(player_outcomes) > 1:
                # Same player outcomes are correlated
                correlation_adjustment -= 0.15 * (len(player_outcomes) - 1)

        # Check for teammate synergies
        # This would use the teammate synergy analysis from advanced_analytics.py
        teammate_pairs = self._get_teammate_pairs(players)
        for pair in teammate_pairs:
            correlation_adjustment += 0.05  # Positive teammate correlation

        return correlation_adjustment

    def _get_teammate_pairs(self, players: List[str]) -> List[Tuple[str, str]]:
        """Get teammate pairs from player list"""
        team_mapping = {
            "Clayton Oliver": "Melbourne",
            "Christian Petracca": "Melbourne",
            "Marcus Bontempelli": "Western Bulldogs",
            "Adam Treloar": "Western Bulldogs",
            "Jeremy Cameron": "Geelong",
            "Tom Hawkins": "Geelong"
        }

        teammate_pairs = []
        for i, player1 in enumerate(players):
            for player2 in players[i+1:]:
                team1 = team_mapping.get(player1)
                team2 = team_mapping.get(player2)
                if team1 and team2 and team1 == team2:
                    teammate_pairs.append((player1, player2))

        return teammate_pairs

    def _calculate_confidence_factors(self, sgm: List[Dict], match_context: Dict) -> Dict:
        """Calculate various confidence factors"""

        confidence_factors = {}

        # Player form confidence
        predictions = [outcome["predicted"] for outcome in sgm]
        targets = [outcome["target"] for outcome in sgm]

        # How far are predictions from targets?
        prediction_accuracy = []
        for pred, target in zip(predictions, targets):
            if pred > 0:
                accuracy = 1 - abs(pred - target) / pred
                prediction_accuracy.append(max(0, accuracy))

        confidence_factors["prediction_accuracy"] = np.mean(prediction_accuracy) if prediction_accuracy else 0.5

        # Weather confidence
        weather = match_context.get("weather", {})
        if weather.get("wind_speed", 0) > 25 or weather.get("precipitation", 0) > 2:
            confidence_factors["weather_stability"] = 0.6  # Lower confidence in extreme weather
        else:
            confidence_factors["weather_stability"] = 0.9

        # Data completeness
        data_completeness = 0.8  # Mock - would calculate based on available data
        confidence_factors["data_completeness"] = data_completeness

        # Market efficiency
        confidence_factors["market_efficiency"] = 0.7  # AFL markets are reasonably efficient

        return confidence_factors

    def _generate_sgm_recommendation(self, value_rating: float, confidence: float, implied_odds: float, target_odds: float) -> str:
        """Generate recommendation for SGM"""

        if value_rating > 0.15 and confidence > 0.8:
            return f"🔥 EXCELLENT SGM - Strong value at {implied_odds:.2f} (target: {target_odds:.2f})"
        elif value_rating > 0.05 and confidence > 0.7:
            return f"✅ GOOD SGM - Solid value at {implied_odds:.2f}"
        elif value_rating > -0.05 and confidence > 0.6:
            return f"⚠️ MARGINAL SGM - Fair value, proceed with caution"
        else:
            return f"❌ AVOID - Poor value or low confidence"
