# Advanced Machine Learning & Automated SGM Picker
# The most sophisticated AFL betting AI system

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
from textblob import TextBlob
import statistics
import itertools
from scipy import stats
import joblib

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

class ForumMonitor:
    """Monitor AFL betting forums and communities for sharp insights"""
    
    def __init__(self):
        self.monitored_sources = [
            "https://www.bigfooty.com/forum/forums/afl-betting.159/",
            "https://www.reddit.com/r/AFLbetting/",
            "https://www.punters.com.au/forum/afl/",
            "https://www.sportsbetting.com.au/forum/afl/"
        ]
        self.sharp_indicators = [
            "value", "edge", "line", "steam", "sharp", "syndicate",
            "overlay", "expected value", "EV", "kelly", "bankroll",
            "correlation", "variance", "unit", "ROI"
        ]
        self.sentiment_cache = {}
    
    async def monitor_forums(self) -> Dict:
        """Monitor forums for betting intelligence"""
        
        insights = {
            "sharp_plays": [],
            "consensus_picks": [],
            "contrarian_opportunities": [],
            "injury_intel": [],
            "sentiment_analysis": {},
            "last_updated": datetime.now().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            tasks = [self._scrape_source(session, source) for source in self.monitored_sources]
            source_data = await asyncio.gather(*tasks, return_exceptions=True)
            
            for data in source_data:
                if isinstance(data, dict) and not isinstance(data, Exception):
                    insights = self._merge_insights(insights, data)
        
        return insights
    
    async def _scrape_source(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Scrape individual forum source"""
        try:
            # Mock forum data for demonstration
            return {
                "source": url,
                "posts": [
                    {
                        "content": "Clayton Oliver 25+ disposals looks like value at 2.40. He's averaging 35 at MCG this year.",
                        "author": "SharpBettor123",
                        "timestamp": datetime.now() - timedelta(hours=2),
                        "upvotes": 15,
                        "sharp_score": 0.8
                    },
                    {
                        "content": "Jeremy Cameron under 2.5 goals. Weather forecast shows 30+ km/h winds at GMHBA.",
                        "author": "WeatherAnalyst",
                        "timestamp": datetime.now() - timedelta(hours=1),
                        "upvotes": 8,
                        "sharp_score": 0.6
                    },
                    {
                        "content": "Hearing Petracca might be rested this week. Ribs still bothering him.",
                        "author": "InsiderTips",
                        "timestamp": datetime.now() - timedelta(minutes=30),
                        "upvotes": 23,
                        "sharp_score": 0.9
                    }
                ]
            }
        except Exception as e:
            return {"error": str(e), "source": url}
    
    def _merge_insights(self, main_insights: Dict, source_data: Dict) -> Dict:
        """Merge insights from different sources"""
        
        if "error" in source_data:
            return main_insights
        
        for post in source_data.get("posts", []):
            content = post["content"].lower()
            sharp_score = post.get("sharp_score", 0)
            
            # Classify post type
            if any(indicator in content for indicator in self.sharp_indicators):
                if sharp_score > 0.7:
                    main_insights["sharp_plays"].append({
                        "content": post["content"],
                        "author": post["author"],
                        "source": source_data["source"],
                        "confidence": sharp_score,
                        "upvotes": post.get("upvotes", 0)
                    })
            
            # Check for injury intel
            if any(word in content for word in ["injury", "hurt", "sore", "rested", "managed"]):
                main_insights["injury_intel"].append({
                    "content": post["content"],
                    "author": post["author"],
                    "timestamp": post["timestamp"],
                    "reliability": sharp_score
                })
            
            # Sentiment analysis
            sentiment = TextBlob(post["content"]).sentiment
            player_mentions = self._extract_player_mentions(post["content"])
            
            for player in player_mentions:
                if player not in main_insights["sentiment_analysis"]:
                    main_insights["sentiment_analysis"][player] = []
                
                main_insights["sentiment_analysis"][player].append({
                    "polarity": sentiment.polarity,
                    "subjectivity": sentiment.subjectivity,
                    "source": source_data["source"]
                })
        
        return main_insights
    
    def _extract_player_mentions(self, text: str) -> List[str]:
        """Extract AFL player mentions from text"""
        # Common AFL player names
        players = [
            "Clayton Oliver", "Christian Petracca", "Marcus Bontempelli",
            "Jeremy Cameron", "Tom Hawkins", "Patrick Cripps",
            "Nick Daicos", "Lachie Neale", "Zach Merrett"
        ]
        
        mentioned = []
        text_lower = text.lower()
        
        for player in players:
            if player.lower() in text_lower:
                mentioned.append(player)
        
        return mentioned
    
    def get_consensus_sentiment(self, player: str) -> Dict:
        """Get consensus sentiment for specific player"""
        if player not in self.sentiment_cache:
            return {"sentiment": "neutral", "confidence": 0.5}
        
        sentiments = self.sentiment_cache[player]
        avg_polarity = np.mean([s["polarity"] for s in sentiments])
        avg_subjectivity = np.mean([s["subjectivity"] for s in sentiments])
        
        if avg_polarity > 0.2:
            sentiment = "positive"
        elif avg_polarity < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        confidence = 1 - avg_subjectivity  # Lower subjectivity = higher confidence
        
        return {
            "sentiment": sentiment,
            "polarity": round(avg_polarity, 3),
            "confidence": round(confidence, 3),
            "sample_size": len(sentiments)
        }

class AdvancedStatistics:
    """Advanced statistical analysis for AFL betting"""
    
    @staticmethod
    def monte_carlo_simulation(player_predictions: List[Dict], num_simulations: int = 10000) -> Dict:
        """Run Monte Carlo simulation for SGM success probability"""
        
        successes = 0
        outcome_distributions = []
        
        for player_pred in player_predictions:
            predictions = player_pred["predictions"]
            confidence = player_pred["confidence"]
            
            # Create probability distributions for each stat
            for stat_type, predicted_value in predictions.items():
                # Use normal distribution with std based on confidence
                std_dev = predicted_value * (1 - confidence.get(stat_type, 0.7)) * 0.3
                distribution = stats.norm(predicted_value, std_dev)
                outcome_distributions.append({
                    "player": player_pred.get("player", "Unknown"),
                    "stat": stat_type,
                    "distribution": distribution,
                    "target": player_pred.get("targets", {}).get(stat_type, predicted_value)
                })
        
        # Run simulations
        for _ in range(num_simulations):
            simulation_success = True
            
            for outcome in outcome_distributions:
                simulated_value = outcome["distribution"].rvs()
                target_value = outcome["target"]
                
                if simulated_value < target_value:
                    simulation_success = False
                    break
            
            if simulation_success:
                successes += 1
        
        probability = successes / num_simulations
        
        # Calculate confidence intervals
        std_error = np.sqrt(probability * (1 - probability) / num_simulations)
        confidence_95 = 1.96 * std_error
        
        return {
            "success_probability": round(probability, 4),
            "confidence_interval_95": [
                round(max(0, probability - confidence_95), 4),
                round(min(1, probability + confidence_95), 4)
            ],
            "simulations_run": num_simulations,
            "expected_outcomes": successes
        }
    
    @staticmethod
    def bayesian_player_rating(historical_performance: List[float], recent_performance: List[float]) -> Dict:
        """Bayesian updating of player performance rating"""
        
        # Prior (based on historical performance)
        if historical_performance:
            prior_mean = np.mean(historical_performance)
            prior_var = np.var(historical_performance)
            prior_precision = 1 / prior_var if prior_var > 0 else 1
        else:
            prior_mean = 20.0  # Default prior
            prior_precision = 0.1
        
        # Likelihood (recent performance)
        if recent_performance:
            likelihood_mean = np.mean(recent_performance)
            likelihood_var = np.var(recent_performance)
            likelihood_precision = len(recent_performance) / likelihood_var if likelihood_var > 0 else len(recent_performance)
        else:
            return {"error": "No recent performance data"}
        
        # Posterior (Bayesian update)
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * likelihood_mean) / posterior_precision
        posterior_var = 1 / posterior_precision
        
        # Confidence in prediction
        confidence = min(0.95, posterior_precision / (posterior_precision + 1))
        
        return {
            "updated_rating": round(posterior_mean, 2),
            "confidence": round(confidence, 3),
            "variance": round(posterior_var, 3),
            "prior_influence": round(prior_precision / posterior_precision, 3),
            "recent_influence": round(likelihood_precision / posterior_precision, 3)
        }
    
    @staticmethod
    def regression_analysis(features: np.array, targets: np.array) -> Dict:
        """Advanced regression analysis to find hidden patterns"""
        
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import r2_score
        
        # Polynomial features for non-linear relationships
        poly = PolynomialFeatures(degree=2, include_bias=False)
        features_poly = poly.fit_transform(features)
        
        # Ridge regression to handle multicollinearity
        ridge = Ridge(alpha=1.0)
        ridge.fit(features_poly, targets)
        
        # Predictions and R²
        predictions = ridge.predict(features_poly)
        r2 = r2_score(targets, predictions)
        
        # Feature importance (absolute coefficients)
        feature_names = poly.get_feature_names_out()
        feature_importance = np.abs(ridge.coef_)
        
        # Sort by importance
        importance_indices = np.argsort(feature_importance)[::-1]
        top_features = [(feature_names[i], feature_importance[i]) for i in importance_indices[:10]]
        
        return {
            "r_squared": round(r2, 4),
            "top_predictive_features": top_features,
            "model_accuracy": "High" if r2 > 0.8 else "Medium" if r2 > 0.6 else "Low",
            "hidden_patterns": AdvancedStatistics._interpret_features(top_features)
        }
    
    @staticmethod
    def _interpret_features(top_features: List[Tuple]) -> List[str]:
        """Interpret what the top features mean"""
        interpretations = []
        
        for feature_name, importance in top_features[:5]:
            if "temperature" in feature_name.lower():
                interpretations.append(f"Temperature has {importance:.3f} impact on performance")
            elif "wind" in feature_name.lower():
                interpretations.append(f"Wind conditions significantly affect outcomes ({importance:.3f})")
            elif "venue" in feature_name.lower():
                interpretations.append(f"Venue familiarity is crucial ({importance:.3f})")
            elif "recent" in feature_name.lower():
                interpretations.append(f"Recent form strongly predicts performance ({importance:.3f})")
        
        return interpretations

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