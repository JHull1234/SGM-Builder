# SportDevs AFL API Integration - Live 2025 Season Data
# Professional AFL statistics for real SGM analysis

import httpx
import asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import os

class SportDevsAPIService:
    """Professional AFL data integration with SportDevs API"""
    
    def __init__(self):
        self.api_key = os.environ.get('SPORTDEVS_API_KEY')
        self.base_url = "https://aussie-rules.sportdevs.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.session = httpx.AsyncClient(timeout=30.0)
        
        if not self.api_key:
            logging.warning("SPORTDEVS_API_KEY not found in environment variables")
    
    async def get_player_season_stats(self, player_name: str, season: int = 2025) -> Dict:
        """Get player information and available data from SportDevs API"""
        try:
            # Search for player by name
            response = await self.session.get(
                f"{self.base_url}/players",
                headers=self.headers,
                params={
                    "name": f"ilike.*{player_name}*",
                    "limit": 5
                }
            )
            
            if response.status_code != 200:
                return {"error": f"API returned status {response.status_code}: {response.text}"}
            
            players = response.json()
            
            if not players:
                return {"error": f"No player found matching '{player_name}'"}
            
            # Take the best match
            player = players[0]
            
            # Try to get additional statistics if available in other endpoints
            # Note: Full statistics may require higher subscription tier
            
            return {
                "player_name": player_name,
                "actual_player_name": player.get("name", "Unknown"),
                "player_id": player.get("id"),
                "team": player.get("team_name", "Unknown"),
                "position": player.get("player_position", "Unknown"),
                "jersey_number": player.get("player_jersey_number", "Unknown"),
                "height": player.get("player_height", 0),
                "date_of_birth": player.get("date_of_birth", "Unknown"),
                "data_source": "SportDevs API",
                "api_note": "Player profile data available. Full statistics may require higher subscription tier.",
                "subscription_info": "Current access: Basic player profiles. Upgrade for detailed statistics.",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error fetching player data for {player_name}: {str(e)}")
            return {"error": str(e)}
    
    async def get_player_recent_games(self, player_name: str, num_games: int = 5) -> Dict:
        """Get last N games for a player with detailed stats"""
        try:
            # Get matches-players-statistics for recent games
            response = await self.session.get(
                f"{self.base_url}/matches-players-statistics",
                headers=self.headers,
                params={
                    "player_name": f"ilike.*{player_name}*",
                    "order": "match_date.desc",
                    "limit": num_games
                }
            )
            
            if response.status_code != 200:
                return {"error": f"API returned status {response.status_code}: {response.text}"}
            
            games_data = response.json()
            
            if not games_data:
                return {"error": f"No recent games found for {player_name}"}
            
            # Calculate recent form averages
            recent_stats = {
                "disposals": [game.get("disposals", 0) for game in games_data],
                "goals": [game.get("goals", 0) for game in games_data],
                "marks": [game.get("marks", 0) for game in games_data],
                "tackles": [game.get("tackles", 0) for game in games_data]
            }
            
            recent_averages = {
                "disposals": round(sum(recent_stats["disposals"]) / len(games_data), 1),
                "goals": round(sum(recent_stats["goals"]) / len(games_data), 1),
                "marks": round(sum(recent_stats["marks"]) / len(games_data), 1),
                "tackles": round(sum(recent_stats["tackles"]) / len(games_data), 1)
            }
            
            return {
                "player_name": player_name,
                "actual_player_name": games_data[0].get("player_name", "Unknown"),
                "games_analyzed": len(games_data),
                "recent_games": games_data,
                "recent_averages": recent_averages,
                "recent_stats_lists": recent_stats,
                "form_trend": self._calculate_form_trend(recent_stats["disposals"]),
                "data_source": "SportDevs API",
                "last_updated": datetime.now().isoformat()
            }
                
        except Exception as e:
            logging.error(f"Error fetching recent games for {player_name}: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_form_trend(self, recent_values: List[float]) -> str:
        """Calculate if player is improving, declining, or stable"""
        if len(recent_values) < 3:
            return "insufficient_data"
        
        # Compare recent 2 games vs older games
        recent_avg = sum(recent_values[:2]) / 2
        older_avg = sum(recent_values[2:]) / len(recent_values[2:])
        
        if recent_avg > older_avg * 1.15:
            return "improving"
        elif recent_avg < older_avg * 0.85:
            return "declining"
        else:
            return "stable"
    
    async def get_team_defensive_stats(self, team_name: str, season: int = 2025) -> Dict:
        """Get team defensive statistics for 2025 season"""
        try:
            team_id = await self._search_team_id(team_name)
            if not team_id:
                return {"error": f"Team {team_name} not found"}
            
            response = await self.session.get(
                f"{self.base_url}/teams/{team_id}/stats/defensive/{season}",
                headers=self.headers
            )
            
            if response.status_code != 200:
                return {"error": f"API returned status {response.status_code}"}
            
            data = response.json()
            
            return {
                "team_name": team_name,
                "team_id": team_id,
                "season": season,
                "defensive_stats": {
                    "points_against_per_game": data.get("points_against_per_game", 0),
                    "disposals_allowed_per_game": data.get("disposals_allowed_per_game", 0),
                    "tackles_per_game": data.get("tackles_per_game", 0),
                    "intercepts_per_game": data.get("intercepts_per_game", 0),
                    "pressure_acts_per_game": data.get("pressure_acts_per_game", 0),
                    "defensive_efficiency": data.get("defensive_efficiency", 0)
                },
                "ranking": {
                    "defensive_rank": data.get("defensive_rank", 0),
                    "total_teams": 18
                },
                "data_source": "SportDevs API",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error fetching team defensive stats for {team_name}: {str(e)}")
            return {"error": str(e)}
    
    async def get_injury_reports(self, season: int = 2025) -> List[Dict]:
        """Get current injury reports across the league"""
        try:
            response = await self.session.get(
                f"{self.base_url}/injuries/current/{season}",
                headers=self.headers
            )
            
            if response.status_code != 200:
                return [{"error": f"API returned status {response.status_code}"}]
            
            injuries_data = response.json().get("injuries", [])
            
            formatted_injuries = []
            for injury in injuries_data:
                formatted_injuries.append({
                    "player_name": injury.get("player_name", "Unknown"),
                    "team": injury.get("team", "Unknown"),
                    "injury_type": injury.get("injury_type", "Unknown"),
                    "status": injury.get("status", "Unknown"),  # injured, test, available
                    "expected_return": injury.get("expected_return"),
                    "weeks_out": injury.get("weeks_out", 0),
                    "data_source": "SportDevs API",
                    "last_updated": datetime.now().isoformat()
                })
            
            return formatted_injuries
            
        except Exception as e:
            logging.error(f"Error fetching injury reports: {str(e)}")
            return [{"error": str(e)}]
    
    async def get_live_match_data(self, season: int = 2025) -> List[Dict]:
        """Get current round matches and live data"""
        try:
            response = await self.session.get(
                f"{self.base_url}/matches",
                headers=self.headers,
                params={
                    "season_id": f"eq.{season}",
                    "order": "match_date.desc",
                    "limit": 20
                }
            )
            
            if response.status_code != 200:
                return [{"error": f"API returned status {response.status_code}: {response.text}"}]
            
            matches_data = response.json()
            
            formatted_matches = []
            for match in matches_data:
                formatted_matches.append({
                    "match_id": match.get("match_id"),
                    "home_team": match.get("home_team_name", "Unknown"),
                    "away_team": match.get("away_team_name", "Unknown"),
                    "venue": match.get("venue_name", "Unknown"),
                    "date": match.get("match_date"),
                    "round": match.get("round_name", "Unknown"),
                    "season": match.get("season_name", season),
                    "status": match.get("status", "Unknown"),
                    "home_score": match.get("home_team_score", 0),
                    "away_score": match.get("away_team_score", 0),
                    "data_source": "SportDevs API",
                    "last_updated": datetime.now().isoformat()
                })
            
            return formatted_matches
            
        except Exception as e:
            logging.error(f"Error fetching live match data: {str(e)}")
            return [{"error": str(e)}]
    
    async def _search_player_id(self, player_name: str) -> Optional[int]:
        """Search for player ID by name"""
        try:
            response = await self.session.get(
                f"{self.base_url}/players/search",
                headers=self.headers,
                params={"name": player_name}
            )
            
            if response.status_code == 200:
                search_results = response.json().get("players", [])
                if search_results:
                    # Return the first match (could be improved with fuzzy matching)
                    return search_results[0].get("player_id")
            
            return None
            
        except Exception as e:
            logging.error(f"Error searching for player {player_name}: {str(e)}")
            return None
    
    async def _search_team_id(self, team_name: str) -> Optional[int]:
        """Search for team ID by name"""
        try:
            response = await self.session.get(
                f"{self.base_url}/teams/search",
                headers=self.headers,
                params={"name": team_name}
            )
            
            if response.status_code == 200:
                search_results = response.json().get("teams", [])
                if search_results:
                    return search_results[0].get("team_id")
            
            return None
            
        except Exception as e:
            logging.error(f"Error searching for team {team_name}: {str(e)}")
            return None
    
    async def close(self):
        """Close the HTTP session"""
        await self.session.aclose()

class LiveAFLAnalyzer:
    """Advanced SGM analysis using live AFL data from SportDevs"""
    
    def __init__(self):
        self.sportdevs = SportDevsAPIService()
    
    async def analyze_live_sgm(self, selections: List[Dict], venue: str = "MCG") -> Dict:
        """Analyze SGM using live 2025 AFL data"""
        try:
            enhanced_predictions = []
            
            for selection in selections:
                player_name = selection["player"]
                stat_type = selection["stat_type"]
                threshold = selection["threshold"]
                
                # Get live player data
                season_stats = await self.sportdevs.get_player_season_stats(player_name)
                recent_games = await self.sportdevs.get_player_recent_games(player_name, 5)
                
                if "error" not in season_stats and "error" not in recent_games:
                    # Calculate probability using real data
                    season_avg = season_stats["season_averages"].get(stat_type, 0)
                    recent_avg = recent_games["recent_averages"].get(stat_type, season_avg)
                    
                    # Use normal distribution for probability calculation
                    import numpy as np
                    from scipy import stats
                    
                    # Adjust for recent form vs season form
                    form_factor = recent_avg / season_avg if season_avg > 0 else 1.0
                    adjusted_avg = recent_avg if form_factor > 0.8 else (recent_avg + season_avg) / 2
                    
                    # Estimate standard deviation based on recent games variability
                    recent_values = recent_games["recent_stats_lists"].get(stat_type, [])
                    if len(recent_values) > 2:
                        std_dev = np.std(recent_values)
                    else:
                        std_dev = adjusted_avg * 0.3  # 30% coefficient of variation
                    
                    # Calculate probability of exceeding threshold
                    if std_dev > 0:
                        z_score = (threshold - adjusted_avg) / std_dev
                        probability = 1 - stats.norm.cdf(z_score)
                        probability = max(0.05, min(0.95, probability))  # Clamp between 5-95%
                    else:
                        probability = 0.5
                    
                    enhanced_predictions.append({
                        "player": player_name,
                        "stat_type": stat_type,
                        "threshold": threshold,
                        "probability": round(probability, 3),
                        "season_average": round(season_avg, 1),
                        "recent_average": round(recent_avg, 1),
                        "form_factor": round(form_factor, 3),
                        "games_played": season_stats["games_played"],
                        "recent_games_analyzed": recent_games["games_analyzed"],
                        "recent_values": recent_values,
                        "data_quality": "Live AFL Data",
                        "confidence": "High" if season_stats["games_played"] > 10 else "Medium"
                    })
                else:
                    # Fallback if API data unavailable
                    enhanced_predictions.append({
                        "player": player_name,
                        "stat_type": stat_type,
                        "threshold": threshold,
                        "probability": 0.5,
                        "data_quality": "API Error",
                        "confidence": "Low",
                        "error": season_stats.get("error", "Unknown error")
                    })
            
            # Calculate combined probability
            individual_probs = [pred["probability"] for pred in enhanced_predictions]
            combined_prob = 1.0
            for prob in individual_probs:
                combined_prob *= prob
            
            # Apply correlation adjustments for same-team players
            correlation_factor = 1.0
            teams = [pred.get("team", "") for pred in enhanced_predictions]
            if len(set(teams)) < len(teams):  # Same team players
                correlation_factor = 0.95  # Slight positive correlation
            
            final_combined_prob = combined_prob * correlation_factor
            implied_odds = 1 / final_combined_prob if final_combined_prob > 0 else 999
            
            return {
                "selections": selections,
                "enhanced_predictions": enhanced_predictions,
                "analysis": {
                    "individual_probabilities": individual_probs,
                    "combined_probability": round(final_combined_prob, 4),
                    "implied_odds": round(implied_odds, 2),
                    "correlation_factor": correlation_factor,
                    "data_source": "SportDevs Live AFL API",
                    "recommendation": self._get_recommendation(final_combined_prob, implied_odds),
                    "confidence_level": "High" if all(p.get("confidence") == "High" for p in enhanced_predictions) else "Medium"
                },
                "venue": venue,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Live SGM analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _get_recommendation(self, probability: float, implied_odds: float) -> str:
        """Generate betting recommendation"""
        if probability > 0.30:
            return f"🔥 EXCELLENT VALUE - High probability ({probability*100:.1f}%) at ${implied_odds:.2f}"
        elif probability > 0.20:
            return f"✅ GOOD VALUE - Decent probability ({probability*100:.1f}%) at ${implied_odds:.2f}"
        elif probability > 0.15:
            return f"⚠️ FAIR VALUE - Moderate probability ({probability*100:.1f}%) at ${implied_odds:.2f}"
        else:
            return f"❌ POOR VALUE - Low probability ({probability*100:.1f}%) at ${implied_odds:.2f}"
    
    async def close(self):
        """Close SportDevs connection"""
        await self.sportdevs.close()
