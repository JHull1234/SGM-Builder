from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import httpx
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import uuid
from pydantic import BaseModel
import statistics
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="AFL Same Game Multi Analytics", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
client = AsyncIOMotorClient(os.environ.get('MONGO_URL'))
db = client[os.environ.get('DB_NAME', 'afl_betting_analytics')]

# API Keys
WEATHERAPI_KEY = os.environ.get('WEATHERAPI_KEY')
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
SQUIGGLE_API_URL = os.environ.get('SQUIGGLE_API_URL')

# Debug print to verify environment variables
print(f"DEBUG: WEATHERAPI_KEY loaded: {WEATHERAPI_KEY[:10]}..." if WEATHERAPI_KEY else "None")
print(f"DEBUG: ODDS_API_KEY loaded: {ODDS_API_KEY[:10]}..." if ODDS_API_KEY else "None")
print(f"DEBUG: SQUIGGLE_API_URL loaded: {SQUIGGLE_API_URL}")

# AFL Venues with coordinates
AFL_VENUES = {
    "MCG": {"lat": -37.8200, "lon": 144.9834, "city": "Melbourne", "state": "VIC"},
    "Marvel Stadium": {"lat": -37.8164, "lon": 144.9475, "city": "Melbourne", "state": "VIC"},
    "Adelaide Oval": {"lat": -34.9155, "lon": 138.5959, "city": "Adelaide", "state": "SA"},
    "Optus Stadium": {"lat": -31.9505, "lon": 115.8605, "city": "Perth", "state": "WA"},
    "Gabba": {"lat": -27.4858, "lon": 153.0389, "city": "Brisbane", "state": "QLD"},
    "SCG": {"lat": -33.8915, "lon": 151.2244, "city": "Sydney", "state": "NSW"},
    "ANZ Stadium": {"lat": -33.8474, "lon": 151.0628, "city": "Sydney", "state": "NSW"},
    "GMHBA Stadium": {"lat": -38.1579, "lon": 144.3544, "city": "Geelong", "state": "VIC"}
}

class SGMPrediction(BaseModel):
    match_id: str
    player_outcomes: List[Dict]
    team_outcomes: List[Dict]
    weather_impact: Dict
    correlation_score: float
    value_rating: float
    recommended_stake: float

class PlayerPerformance(BaseModel):
    player_name: str
    team: str
    position: str
    predicted_disposals: float
    predicted_goals: float
    predicted_marks: float
    predicted_tackles: float
    confidence: float

# Classes for real-time monitoring
class RealTimeOddsMonitor:
    def __init__(self, api_key):
        self.api_key = api_key

class SteamMoveDetector:
    def __init__(self):
        pass

# Initialize venues collection
async def init_venues_collection():
    venues_collection = db["afl_venues"]
    for venue_name, data in AFL_VENUES.items():
        await venues_collection.update_one(
            {"name": venue_name},
            {"$set": {
                "name": venue_name,
                "coordinates": {"lat": data["lat"], "lon": data["lon"]},
                "city": data["city"],
                "state": data["state"]
            }},
            upsert=True
        )

# Import ML and advanced features
from datetime import datetime
import uuid
import asyncio

# Global ML instances
ml_predictor = None
forum_monitor = None
sgm_picker = None

@app.on_event("startup")
async def startup_event():
    global odds_monitor, steam_detector, ml_predictor, forum_monitor, sgm_picker
    await init_venues_collection()
    
    # Initialize real-time monitoring
    odds_monitor = RealTimeOddsMonitor(ODDS_API_KEY)
    steam_detector = SteamMoveDetector()
    
    # Initialize ML components
    try:
        import sys
        sys.path.append('/app')
        from ml_sgm_picker import MachineLearningPredictor, ForumMonitor, AutomatedSGMPicker
        
        ml_predictor = MachineLearningPredictor()
        forum_monitor = ForumMonitor()
        sgm_picker = AutomatedSGMPicker(ml_predictor)
        
        print("🤖 ML SGM Picker and Forum Monitor initialized")
    except Exception as e:
        print(f"ML initialization warning: {e}")
    
    print("🚀 AFL Analytics Platform started with full AI capabilities")

class AFLDataService:
    @staticmethod
    async def get_current_matches():
        """Get current AFL matches from Squiggle API"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{SQUIGGLE_API_URL}/?q=games;year=2025")
                response.raise_for_status()
                matches = response.json()['games']
                
                # Filter for upcoming matches in next 7 days
                upcoming_matches = []
                for match in matches:
                    if match.get('complete') == 0:  # Match not completed
                        upcoming_matches.append({
                            "match_id": str(uuid.uuid4()),
                            "home_team": match.get('hteam'),
                            "away_team": match.get('ateam'),
                            "venue": match.get('venue'),
                            "date": match.get('date'),
                            "round": match.get('round'),
                            "year": match.get('year')
                        })
                
                return upcoming_matches[:10]  # Return next 10 matches
            except Exception as e:
                raise HTTPException(500, f"Failed to fetch AFL matches: {str(e)}")

    @staticmethod
    async def get_team_stats(team_name: str):
        """Get team statistics from Squiggle API"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{SQUIGGLE_API_URL}/?q=standings;year=2025;team={team_name}")
                response.raise_for_status()
                return response.json().get('standings', [])
            except Exception as e:
                return {"error": f"Failed to fetch team stats: {str(e)}"}

    @staticmethod
    async def get_player_stats():
        """Get comprehensive player statistics for SGM analysis"""
        # Import enhanced player data
        import sys
        import os
        sys.path.append('/app')
        from enhanced_player_data import COMPREHENSIVE_AFL_PLAYERS, TEAM_DEFENSIVE_STATS
        
        return COMPREHENSIVE_AFL_PLAYERS

class WeatherService:
    @staticmethod
    async def get_venue_weather(venue_name: str):
        """Get current weather for AFL venue"""
        if venue_name not in AFL_VENUES:
            raise HTTPException(404, f"Venue {venue_name} not found")
        
        venue = AFL_VENUES[venue_name]
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "http://api.weatherapi.com/v1/current.json",
                    params={
                        "key": WEATHERAPI_KEY,
                        "q": f"{venue['lat']},{venue['lon']}",
                        "aqi": "yes"
                    },
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                return {
                    "venue": venue_name,
                    "temperature": data["current"]["temp_c"],
                    "humidity": data["current"]["humidity"],
                    "wind_speed": data["current"]["wind_kph"],
                    "wind_direction": data["current"]["wind_dir"],
                    "precipitation": data["current"]["precip_mm"],
                    "conditions": data["current"]["condition"]["text"],
                    "pressure": data["current"]["pressure_mb"],
                    "visibility": data["current"]["vis_km"],
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(500, f"Weather API error: {str(e)}")

class BettingOddsService:
    @staticmethod
    async def get_afl_odds():
        """Get AFL betting odds from The Odds API"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds",
                    params={
                        "apiKey": ODDS_API_KEY,
                        "regions": "au",
                        "markets": "h2h,spreads,totals",
                        "dateFormat": "iso"
                    },
                    timeout=15
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                raise HTTPException(500, f"Betting odds API error: {str(e)}")

class SGMAnalytics:
    @staticmethod
    def calculate_correlation_score(outcomes: List[Dict]) -> float:
        """Calculate sophisticated correlation between multiple outcomes in SGM"""
        import sys
        sys.path.append('/app')
        from enhanced_player_data import TEAM_DEFENSIVE_STATS
        
        base_score = 0.8
        
        # Group outcomes by player and team
        player_outcomes = {}
        team_outcomes = {}
        
        for outcome in outcomes:
            player = outcome.get('player')
            if player:
                if player not in player_outcomes:
                    player_outcomes[player] = []
                player_outcomes[player].append(outcome)
                
                # Get player's team
                team = SGMAnalytics._get_player_team(player)
                if team not in team_outcomes:
                    team_outcomes[team] = []
                team_outcomes[team].append(outcome)
        
        # 1. Same player penalty (outcomes are less independent)
        for player, player_outs in player_outcomes.items():
            if len(player_outs) > 1:
                # Different penalty based on stat correlation
                correlation_penalty = SGMAnalytics._calculate_stat_correlation_penalty(player_outs)
                base_score -= correlation_penalty
        
        # 2. Teammate synergy bonus (positive correlation)
        for team, team_outs in team_outcomes.items():
            if len(team_outs) > 1:
                synergy_bonus = SGMAnalytics._calculate_teammate_synergy(team_outs)
                base_score += synergy_bonus
        
        # 3. Position-based correlation adjustments
        position_adjustment = SGMAnalytics._calculate_position_correlation(outcomes)
        base_score *= position_adjustment
        
        return max(0.1, min(1.0, base_score))
    
    @staticmethod
    def _get_player_team(player_name: str) -> str:
        """Get team for a player"""
        team_mapping = {
            "Clayton Oliver": "Melbourne",
            "Christian Petracca": "Melbourne", 
            "Marcus Bontempelli": "Western Bulldogs",
            "Adam Treloar": "Western Bulldogs",
            "Jeremy Cameron": "Geelong",
            "Tom Hawkins": "Geelong",
            "Patrick Cripps": "Carlton",
            "Sam Walsh": "Carlton",
            "Nick Daicos": "Collingwood",
            "Scott Pendlebury": "Collingwood",
            "Zach Merrett": "Essendon",
            "Darcy Parish": "Essendon",
            "Caleb Serong": "Fremantle",
            "Andrew Brayshaw": "Fremantle",
            "Lachie Neale": "Brisbane",
            "Hugh McCluggage": "Brisbane",
            "Rory Laird": "Adelaide",
            "Jordan Dawson": "Adelaide"
        }
        return team_mapping.get(player_name, "Unknown")
    
    @staticmethod
    def _calculate_stat_correlation_penalty(player_outcomes: List[Dict]) -> float:
        """Calculate penalty for multiple outcomes from same player"""
        stat_types = [outcome.get('type', '') for outcome in player_outcomes]
        
        # Different stats have different correlation levels
        correlation_matrix = {
            ('disposals', 'disposals'): 0.95,  # Very high correlation
            ('disposals', 'marks'): 0.65,     # Positive correlation
            ('disposals', 'goals'): -0.25,    # Slight negative correlation
            ('disposals', 'tackles'): 0.45,   # Moderate correlation
            ('goals', 'marks'): 0.70,         # High correlation for forwards
            ('goals', 'tackles'): -0.15,      # Slight negative correlation
            ('marks', 'tackles'): 0.20        # Low correlation
        }
        
        total_penalty = 0
        for i, stat1 in enumerate(stat_types):
            for j, stat2 in enumerate(stat_types[i+1:], i+1):
                correlation = correlation_matrix.get((stat1, stat2), 0.5)
                correlation = correlation_matrix.get((stat2, stat1), correlation)
                penalty = abs(correlation) * 0.2  # Convert correlation to penalty
                total_penalty += penalty
        
        return total_penalty
    
    @staticmethod
    def _calculate_teammate_synergy(team_outcomes: List[Dict]) -> float:
        """Calculate synergy bonus for teammate outcomes"""
        # Teammates often help each other achieve outcomes
        # Example: Clayton Oliver + Christian Petracca both performing well
        
        if len(team_outcomes) == 2:
            return 0.05  # Small positive correlation for teammates
        elif len(team_outcomes) > 2:
            return 0.03  # Diminishing returns for more teammates
        return 0
    
    @staticmethod
    def _calculate_position_correlation(outcomes: List[Dict]) -> float:
        """Adjust correlation based on player positions"""
        position_mapping = {
            "Clayton Oliver": "Midfielder",
            "Christian Petracca": "Midfielder",
            "Marcus Bontempelli": "Midfielder", 
            "Jeremy Cameron": "Forward",
            "Tom Hawkins": "Forward",
            "Patrick Cripps": "Midfielder"
        }
        
        positions = []
        for outcome in outcomes:
            player = outcome.get('player')
            position = position_mapping.get(player, 'Midfielder')
            positions.append(position)
        
        # Same position outcomes are more correlated
        unique_positions = len(set(positions))
        if unique_positions == 1:
            return 0.95  # Same position = higher correlation
        else:
            return 1.0   # Different positions = more independent
    
    @staticmethod
    def calculate_weather_impact(weather: Dict, outcomes: List[Dict]) -> Dict:
        """Calculate sophisticated weather impact on SGM outcomes"""
        impact = {
            "total_impact": 0.0,
            "wind_impact": 0.0,
            "rain_impact": 0.0,
            "temperature_impact": 0.0,
            "detailed_breakdown": {}
        }
        
        for outcome in outcomes:
            stat_type = outcome.get("type", "").lower()
            player = outcome.get("player", "")
            
            # Position-specific weather impact
            position = SGMAnalytics._get_player_position(player)
            
            # Wind impact (non-linear)
            if weather["wind_speed"] > 20:
                if "goals" in stat_type:
                    wind_penalty = -0.02 * (weather["wind_speed"] - 20)  # -2% per km/h above 20
                    impact["wind_impact"] += wind_penalty
                elif "marks" in stat_type:
                    wind_penalty = -0.015 * (weather["wind_speed"] - 20)  # -1.5% per km/h above 20
                    impact["wind_impact"] += wind_penalty
                elif "disposals" in stat_type:
                    wind_penalty = -0.008 * (weather["wind_speed"] - 20)  # -0.8% per km/h above 20
                    impact["wind_impact"] += wind_penalty
            
            # Rain impact (threshold-based)
            if weather["precipitation"] > 1:
                rain_severity = min(weather["precipitation"] / 10, 1.0)  # Cap at 10mm
                if "disposals" in stat_type:
                    rain_penalty = -0.10 * rain_severity  # -10% max in heavy rain
                    impact["rain_impact"] += rain_penalty
                elif "marks" in stat_type:
                    rain_penalty = -0.18 * rain_severity  # -18% max in heavy rain
                    impact["rain_impact"] += rain_penalty
                elif "goals" in stat_type:
                    rain_penalty = -0.20 * rain_severity  # -20% max in heavy rain
                    impact["rain_impact"] += rain_penalty
            
            # Temperature impact
            temp = weather["temperature"]
            if temp < 10:
                cold_penalty = -0.015 * (10 - temp)  # -1.5% per degree below 10°C
                impact["temperature_impact"] += cold_penalty
            elif temp > 30:
                hot_penalty = -0.01 * (temp - 30)  # -1% per degree above 30°C
                impact["temperature_impact"] += hot_penalty
            
            # Store detailed breakdown
            impact["detailed_breakdown"][f"{player}_{stat_type}"] = {
                "wind_factor": 1 + (impact["wind_impact"] / len(outcomes)),
                "rain_factor": 1 + (impact["rain_impact"] / len(outcomes)),
                "temperature_factor": 1 + (impact["temperature_impact"] / len(outcomes))
            }
        
        impact["total_impact"] = impact["wind_impact"] + impact["rain_impact"] + impact["temperature_impact"]
        
        return impact
    
    @staticmethod
    def _get_player_position(player_name: str) -> str:
        """Get position for weather impact calculations"""
        position_mapping = {
            "Clayton Oliver": "Midfielder",
            "Christian Petracca": "Midfielder",
            "Marcus Bontempelli": "Midfielder",
            "Jeremy Cameron": "Forward",
            "Tom Hawkins": "Forward",
            "Patrick Cripps": "Midfielder",
            "Sam Walsh": "Midfielder",
            "Nick Daicos": "Midfielder",
            "Zach Merrett": "Midfielder",
            "Caleb Serong": "Midfielder"
        }
        return position_mapping.get(player_name, "Midfielder")
    
    @staticmethod
    def calculate_value_rating(predicted_prob: float, market_odds: float) -> float:
        """Calculate sophisticated value rating for SGM bet"""
        if market_odds <= 0:
            return 0.0
        
        # Convert odds to implied probability (including bookmaker margin)
        implied_prob = 1 / market_odds
        
        # Calculate value (our edge over the market)
        value = (predicted_prob - implied_prob) / implied_prob
        
        # Adjust for confidence in our prediction
        confidence_adjustment = min(predicted_prob, 0.9)  # Cap confidence at 90%
        adjusted_value = value * confidence_adjustment
        
        return max(-1.0, min(2.0, adjusted_value))  # Cap between -100% and +200%
    
    @staticmethod
    def _calculate_defensive_rank(team: str, team_defense: Dict) -> Dict:
        """Calculate where team ranks defensively in the league"""
        # This would use full league data in production
        return {
            "disposals_allowed_rank": "5th best",  # Mock ranking
            "goals_allowed_rank": "8th best",
            "overall_defense_rank": "6th best"
        }
    
    @staticmethod
    def _calculate_matchup_difficulty(player: Dict, team_defense: Dict) -> str:
        """Calculate overall matchup difficulty"""
        # Simple calculation based on team defense vs league average
        midfielder_allowed = team_defense["midfielder_disposals_allowed"]
        league_avg = 375.0 / 3  # Rough estimate for midfielder share
        
        if midfielder_allowed > league_avg * 1.1:
            return "Favorable"
        elif midfielder_allowed < league_avg * 0.9:
            return "Difficult"
        else:
            return "Average"
    
    @staticmethod
    def _project_stat_vs_defense(player_avg: float, defense_allowed: float, league_avg: float) -> float:
        """Project player stat against specific team defense"""
        defensive_factor = defense_allowed / league_avg
        return round(player_avg * defensive_factor, 1)
    
    @staticmethod
    def _generate_matchup_insights(player: Dict, opponent: str, team_defense: Dict) -> List[str]:
        """Generate key insights for the matchup"""
        insights = []
        
        # Defensive strength insight
        if team_defense["midfielder_disposals_allowed"] < 115:
            insights.append(f"{opponent} has a strong midfield defense - expect reduced disposal numbers")
        elif team_defense["midfielder_disposals_allowed"] > 130:
            insights.append(f"{opponent} allows high disposal counts - favorable matchup for midfielders")
        
        # Goal scoring insight
        if team_defense["forward_goals_allowed"] < 7:
            insights.append(f"{opponent} has excellent forward defense - goals will be hard to come by")
        elif team_defense["forward_goals_allowed"] > 9:
            insights.append(f"{opponent} defense leaks goals - good scoring opportunities expected")
        
        # Player-specific insight
        if player["position"] == "Midfielder" and team_defense["tackles_per_game"] > 70:
            insights.append(f"High tackling pressure from {opponent} may impact disposal efficiency")
        
        # Venue insight (if available)
        player_venues = player.get("venue_performance", {})
        if len(insights) < 3:
            insights.append("Consider venue-specific performance when finalizing bet")
        
    @staticmethod
    def _generate_matchup_insights(player: Dict, opponent: str, team_defense: Dict) -> List[str]:
        """Generate key insights for the matchup"""
        insights = []
        
        # Defensive strength insight
        if team_defense["midfielder_disposals_allowed"] < 115:
            insights.append(f"{opponent} has a strong midfield defense - expect reduced disposal numbers")
        elif team_defense["midfielder_disposals_allowed"] > 130:
            insights.append(f"{opponent} allows high disposal counts - favorable matchup for midfielders")
        
        # Goal scoring insight
        if team_defense["forward_goals_allowed"] < 7:
            insights.append(f"{opponent} has excellent forward defense - goals will be hard to come by")
        elif team_defense["forward_goals_allowed"] > 9:
            insights.append(f"{opponent} defense leaks goals - good scoring opportunities expected")
        
        # Player-specific insight
        if player["position"] == "Midfielder" and team_defense["tackles_per_game"] > 70:
            insights.append(f"High tackling pressure from {opponent} may impact disposal efficiency")
        
        # Venue insight (if available)
        player_venues = player.get("venue_performance", {})
        if len(insights) < 3:
            insights.append("Consider venue-specific performance when finalizing bet")
        
        return insights[:3]  # Return top 3 insights
    
    @staticmethod
    def _identify_key_strengths(synergy_analysis: Dict, form_analysis: Dict, injury_analysis: Dict) -> List[str]:
        """Identify key strengths of the SGM"""
        strengths = []
        
        # Synergy strengths
        if synergy_analysis["synergy_rating"] in ["Excellent", "Good"]:
            strengths.append(f"Strong teammate synergy ({synergy_analysis['synergy_rating'].lower()})")
        
        # Form strengths
        hot_players = [key.split('_')[0] for key, data in form_analysis.items() if data["trend"] == "Hot"]
        if hot_players:
            strengths.append(f"Players in hot form: {', '.join(set(hot_players))}")
        
        # Health strengths
        healthy_players = [player for player, data in injury_analysis.items() if data["impact_rating"] == "None"]
        if len(healthy_players) == len(injury_analysis):
            strengths.append("All players healthy - no injury concerns")
        
        return strengths[:3]
    
    @staticmethod
    def _identify_key_concerns(form_analysis: Dict, injury_analysis: Dict, weather_impact: Dict) -> List[str]:
        """Identify key concerns with the SGM"""
        concerns = []
        
        # Form concerns
        cold_players = [key.split('_')[0] for key, data in form_analysis.items() if data["trend"] == "Cold"]
        if cold_players:
            concerns.append(f"Players in cold form: {', '.join(set(cold_players))}")
        
        # Injury concerns
        injured_players = [player for player, data in injury_analysis.items() if data["impact_rating"] in ["Medium", "High"]]
        if injured_players:
            concerns.append(f"Injury concerns: {', '.join(injured_players)}")
        
        # Weather concerns
        if weather_impact["total_impact"] < -0.1:
            concerns.append("Weather conditions may negatively impact performance")
        
        return concerns[:3]
    
    @staticmethod
    def _generate_final_recommendation(value_rating: float, correlation: float, confidence: float) -> str:
        """Generate final betting recommendation"""
        if value_rating > 0.3 and correlation > 0.7 and confidence > 0.8:
            return "STRONG BET - Excellent value with high confidence"
        elif value_rating > 0.1 and correlation > 0.6 and confidence > 0.7:
            return "GOOD BET - Solid value with reasonable confidence"
        elif value_rating > 0 and correlation > 0.5:
            return "CONSIDER - Positive value but monitor closely"
        elif value_rating > -0.1:
            return "MARGINAL - Limited value, proceed with caution"
        else:
            return "AVOID - Poor value or low confidence"

@app.get("/api/player/{player_name}/recent-form")
async def get_player_recent_form(player_name: str):
    """Get player's recent form analysis"""
    try:
        import sys
        sys.path.append('/app')
        from advanced_analytics import RecentFormAnalyzer
        
        # Get form analysis for all stat types
        form_analysis = {
            "disposals": RecentFormAnalyzer.calculate_form_factor(player_name, "disposals"),
            "goals": RecentFormAnalyzer.calculate_form_factor(player_name, "goals"),
            "marks": RecentFormAnalyzer.calculate_form_factor(player_name, "marks"),
            "tackles": RecentFormAnalyzer.calculate_form_factor(player_name, "tackles")
        }
        
        # Overall form assessment
        overall_trend = RecentFormAnalyzer.RECENT_FORM_DATA.get(player_name, {}).get("form_trend", "Unknown")
        injury_concerns = RecentFormAnalyzer.RECENT_FORM_DATA.get(player_name, {}).get("injury_concerns")
        
        return {
            "player": player_name,
            "overall_trend": overall_trend,
            "injury_concerns": injury_concerns,
            "stat_analysis": form_analysis,
            "recommendation": _get_form_recommendation(form_analysis, overall_trend)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Recent form analysis error: {str(e)}")

def _get_form_recommendation(form_analysis: Dict, overall_trend: str) -> str:
    """Generate recommendation based on recent form"""
    if overall_trend == "Hot":
        return "POSITIVE - Player in excellent recent form"
    elif overall_trend == "Cold":
        return "CAUTION - Player struggling for form"
    else:
        return "NEUTRAL - Average recent form"

@app.get("/api/synergy/{player1}/{player2}")
async def get_teammate_synergy(player1: str, player2: str):
    """Get synergy analysis between two players"""
    try:
        import sys
        sys.path.append('/app')
        from advanced_analytics import TeammateSymergyAnalyzer
        
        # Create mock outcomes to test synergy
        mock_outcomes = [
            {"player": player1, "type": "disposals"},
            {"player": player2, "type": "goals"}
        ]
        
        synergy_analysis = TeammateSymergyAnalyzer.calculate_synergy_impact(mock_outcomes)
        
        return {
            "players": [player1, player2],
            "synergy_analysis": synergy_analysis,
            "recommendation": _get_synergy_recommendation(synergy_analysis)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Synergy analysis error: {str(e)}")

def _get_synergy_recommendation(synergy_analysis: Dict) -> str:
    """Generate synergy recommendation"""
    rating = synergy_analysis["synergy_rating"]
    if rating in ["Excellent", "Good"]:
        return f"POSITIVE - {rating} synergy between players"
    elif rating == "Neutral":
        return "NEUTRAL - No significant synergy impact"
    else:
        return f"NEGATIVE - {rating} synergy may hurt SGM chances"

@app.get("/api/market-timing/{bet_type}")
async def get_market_timing(bet_type: str):
    """Get market timing strategy for bet type"""
    try:
        import sys
        sys.path.append('/app')
        from advanced_analytics import MarketMonitor
        
        timing_strategy = MarketMonitor.get_market_timing_strategy(bet_type)
        
        return {
            "bet_type": bet_type,
            "timing_strategy": timing_strategy,
            "market_opening_schedule": MarketMonitor.MARKET_OPENING_TIMES
        }
        
    except Exception as e:
        raise HTTPException(500, f"Market timing analysis error: {str(e)}")

@app.post("/api/odds/track-movement")
async def track_odds_movement(request: Dict):
    """Track odds movement for a specific bet"""
    try:
        import sys
        sys.path.append('/app')
        from advanced_analytics import MarketMonitor
        
        bet_type = request.get("bet_type")
        current_odds = request.get("current_odds")
        
        if not bet_type or not current_odds:
            raise HTTPException(400, "bet_type and current_odds required")
        
        movement_analysis = MarketMonitor.track_line_movement(bet_type, current_odds)
        
        return {
            "bet_type": bet_type,
            "current_odds": current_odds,
            "movement_analysis": movement_analysis
        }
        
    except Exception as e:
        raise HTTPException(500, f"Odds tracking error: {str(e)}")

@app.get("/api/injury-report")
async def get_injury_report():
    """Get comprehensive injury report for all monitored players"""
    try:
        import sys
        sys.path.append('/app')
        from advanced_analytics import InjuryImpactAnalyzer
        
        injury_report = {}
        for player_name in InjuryImpactAnalyzer.INJURY_IMPACT_DATA.keys():
            injury_data = InjuryImpactAnalyzer.get_injury_impact(player_name)
            injury_report[player_name] = injury_data
        
        # Summary statistics
        total_players = len(injury_report)
        injured_players = len([p for p in injury_report.values() if p["impact_rating"] != "None"])
        high_risk = len([p for p in injury_report.values() if p["impact_rating"] == "High"])
        
        return {
            "injury_report": injury_report,
            "summary": {
                "total_monitored": total_players,
                "players_with_concerns": injured_players,
                "high_risk_players": high_risk,
                "overall_status": "Concerning" if high_risk > 0 else "Manageable" if injured_players > 0 else "Healthy"
            },
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Injury report error: {str(e)}")

@app.post("/api/monitoring/start")
async def start_odds_monitoring(request: Dict):
    """Start monitoring specific bets for value opportunities"""
    try:
        import sys
        sys.path.append('/app')
        from realtime_monitoring import RealTimeOddsMonitor
        
        bet_configurations = request.get("bets", [])
        
        if not bet_configurations:
            raise HTTPException(400, "No bets provided for monitoring")
        
        # Initialize monitoring if not already started
        global odds_monitor
        if not odds_monitor:
            odds_monitor = RealTimeOddsMonitor(ODDS_API_KEY)
        
        # Start monitoring (in background)
        monitoring_id = str(uuid.uuid4())
        
        # Store monitoring configuration
        monitoring_collection = db["monitoring_configs"]
        await monitoring_collection.insert_one({
            "monitoring_id": monitoring_id,
            "bet_configurations": bet_configurations,
            "started_at": datetime.now().isoformat(),
            "status": "active"
        })
        
        return {
            "monitoring_id": monitoring_id,
            "status": "started",
            "monitoring_bets": len(bet_configurations),
            "message": "Real-time odds monitoring started"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Monitoring start error: {str(e)}")

@app.get("/api/monitoring/alerts")
async def get_value_alerts():
    """Get current value alerts from monitoring"""
    try:
        import sys
        sys.path.append('/app')
        from realtime_monitoring import AdvancedValueDetector
        
        # Mock alerts for demonstration
        current_alerts = [
            {
                "id": str(uuid.uuid4()),
                "type": "EXCELLENT_VALUE",
                "message": "🔥 EXCELLENT VALUE: Clayton Oliver 25+ Disposals @ bet365 (2.50)",
                "urgency": "HIGH",
                "value_rating": 0.18,
                "bookmaker": "bet365",
                "odds": 2.50,
                "detected_at": datetime.now().isoformat(),
                "action_required": "BET NOW"
            },
            {
                "id": str(uuid.uuid4()),
                "type": "LINE_SHOPPING",
                "message": "💰 LINE SHOPPING: 8.2% better odds at Sportsbet vs TAB",
                "urgency": "MEDIUM",
                "advantage": 8.2,
                "best_bookmaker": "Sportsbet",
                "detected_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "action_required": "COMPARE ODDS"
            },
            {
                "id": str(uuid.uuid4()),
                "type": "STEAM_MOVE",
                "message": "⚠️ SHARP MONEY: Jeremy Cameron goals line moving from 2.8 to 2.4",
                "urgency": "HIGH",
                "movement": -14.3,
                "detected_at": (datetime.now() - timedelta(minutes=2)).isoformat(),
                "action_required": "FOLLOW SHARP MONEY"
            }
        ]
        
        return {
            "alerts": current_alerts,
            "total_alerts": len(current_alerts),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Alerts retrieval error: {str(e)}")

@app.post("/api/advanced-analysis/correlation-mispricing")
async def detect_correlation_mispricing(request: Dict):
    """Detect when market has mispriced correlation in SGM"""
    try:
        import sys
        sys.path.append('/app')
        from realtime_monitoring import AdvancedValueDetector
        
        sgm_outcomes = request.get("outcomes", [])
        market_odds = request.get("market_odds", 3.50)
        
        correlation_analysis = AdvancedValueDetector.detect_correlation_mispricing(
            sgm_outcomes, market_odds
        )
        
        return {
            "analysis_type": "correlation_mispricing",
            "market_odds": market_odds,
            "correlation_analysis": correlation_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Correlation analysis error: {str(e)}")

@app.post("/api/advanced-analysis/recency-bias")
async def detect_recency_bias(request: Dict):
    """Detect when market over-reacts to recent performance"""
    try:
        import sys
        sys.path.append('/app')
        from realtime_monitoring import AdvancedValueDetector
        
        player_stats = request.get("player_stats", {})
        market_odds = request.get("market_odds", 3.50)
        
        recency_analysis = AdvancedValueDetector.detect_recency_bias(
            player_stats, market_odds
        )
        
        return {
            "analysis_type": "recency_bias",
            "market_odds": market_odds,
            "recency_analysis": recency_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Recency bias analysis error: {str(e)}")

@app.post("/api/portfolio/optimize")
async def optimize_betting_portfolio(request: Dict):
    """Optimize betting portfolio for maximum returns"""
    try:
        import sys
        sys.path.append('/app')
        from realtime_monitoring import PortfolioOptimizer
        
        bets = request.get("bets", [])
        total_bankroll = request.get("bankroll", 1000.0)
        
        if not bets:
            raise HTTPException(400, "No bets provided for optimization")
        
        # Optimize bankroll allocation
        allocation_result = PortfolioOptimizer.optimize_bankroll_allocation(bets, total_bankroll)
        
        # Calculate portfolio correlations
        correlation_analysis = PortfolioOptimizer.calculate_portfolio_correlations(bets)
        
        return {
            "bankroll_optimization": allocation_result,
            "correlation_analysis": correlation_analysis,
            "total_bankroll": total_bankroll,
            "optimization_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Portfolio optimization error: {str(e)}")

@app.get("/api/live-tracking/{game_id}")
async def get_live_game_tracking(game_id: str):
    """Get live game tracking data"""
    try:
        import sys
        sys.path.append('/app')
        from realtime_monitoring import LiveGameTracker
        
        # Mock live game data
        live_tracker = LiveGameTracker()
        
        # Simulate some live stats
        mock_player_stats = {
            "Clayton Oliver": {"disposals": 18, "goals": 1, "marks": 3},
            "Christian Petracca": {"disposals": 14, "goals": 0, "marks": 4}
        }
        
        await live_tracker.track_live_performance(game_id, mock_player_stats)
        
        # Get predictions for key players
        predictions = {}
        for player in mock_player_stats.keys():
            prediction = live_tracker.predict_final_stats(game_id, player)
            predictions[player] = prediction
        
        return {
            "game_id": game_id,
            "current_quarter": 2,
            "live_stats": mock_player_stats,
            "final_projections": predictions,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Live tracking error: {str(e)}")

@app.post("/api/steam-detection")
async def detect_steam_moves(request: Dict):
    """Detect steam moves in the market"""
    try:
        import sys
        sys.path.append('/app')
        from realtime_monitoring import SteamMoveDetector
        
        current_odds = request.get("current_odds", {})
        previous_odds = request.get("previous_odds", {})
        
        steam_detector = SteamMoveDetector()
        steam_moves = await steam_detector.detect_steam_moves(current_odds, previous_odds)
        
        return {
            "steam_moves": steam_moves,
            "total_moves_detected": len(steam_moves),
            "detection_timestamp": datetime.now().isoformat(),
            "recommendations": [move["recommendation"] for move in steam_moves]
        }
        
    except Exception as e:
        raise HTTPException(500, f"Steam detection error: {str(e)}")

@app.get("/api/venues")
async def get_all_venues():
    """Get all AFL venues"""
    return [{"name": name, **data} for name, data in AFL_VENUES.items()]

# API Endpoints
@app.get("/")
async def root():
    return {"message": "AFL Same Game Multi Analytics API", "version": "1.0.0"}

@app.get("/api/matches")
async def get_upcoming_matches():
    """Get upcoming AFL matches"""
    return await AFLDataService.get_current_matches()

@app.get("/api/players")
async def get_player_stats():
    """Get player statistics for SGM analysis"""
    return await AFLDataService.get_player_stats()

@app.get("/api/weather/{venue_name}")
async def get_venue_weather(venue_name: str):
    """Get weather conditions for specific venue"""
    return await WeatherService.get_venue_weather(venue_name)

@app.get("/api/odds")
async def get_betting_odds():
    """Get current AFL betting odds"""
    return await BettingOddsService.get_afl_odds()

@app.post("/api/sgm/analyze")
async def analyze_sgm(request: Dict):
    """Analyze Same Game Multi combination with advanced analytics"""
    try:
        # Import advanced analytics
        import sys
        sys.path.append('/app')
        from advanced_analytics import (
            RecentFormAnalyzer, 
            TeammateSymergyAnalyzer,
            DefensiveMatchupAnalyzer,
            InjuryImpactAnalyzer,
            MarketMonitor
        )
        
        match_id = request.get("match_id")
        outcomes = request.get("outcomes", [])
        venue = request.get("venue")
        
        if not outcomes:
            raise HTTPException(400, "No outcomes provided")
        
        # Get weather data for venue
        weather_data = await WeatherService.get_venue_weather(venue)
        
        # 1. RECENT FORM ANALYSIS
        form_analysis = {}
        for outcome in outcomes:
            player = outcome.get("player")
            stat_type = outcome.get("type")
            if player and stat_type:
                form_data = RecentFormAnalyzer.calculate_form_factor(player, stat_type)
                form_analysis[f"{player}_{stat_type}"] = form_data
        
        # 2. TEAMMATE SYNERGY ANALYSIS
        synergy_analysis = TeammateSymergyAnalyzer.calculate_synergy_impact(outcomes)
        
        # 3. ENHANCED CORRELATION WITH SYNERGY
        base_correlation = SGMAnalytics.calculate_correlation_score(outcomes)
        synergy_adjusted_correlation = base_correlation + synergy_analysis["total_synergy_impact"]
        synergy_adjusted_correlation = max(0.1, min(1.0, synergy_adjusted_correlation))
        
        # 4. DEFENSIVE MATCHUP ANALYSIS
        matchup_analysis = {}
        for outcome in outcomes:
            player_name = outcome.get("player")
            if player_name:
                # Get player data for matchup analysis
                import sys
                sys.path.append('/app')
                from enhanced_player_data import COMPREHENSIVE_AFL_PLAYERS
                
                player_data = None
                for player in COMPREHENSIVE_AFL_PLAYERS:
                    if player["name"] == player_name:
                        player_data = player
                        break
                
                if player_data:
                    # Mock opponent team for analysis
                    opponent_team = "Brisbane"  # In production, get from match data
                    detailed_matchup = DefensiveMatchupAnalyzer.get_detailed_matchup_analysis(
                        player_data, opponent_team, venue
                    )
                    matchup_analysis[player_name] = detailed_matchup
        
        # 5. INJURY IMPACT ANALYSIS
        injury_analysis = {}
        total_injury_impact = 0.0
        for outcome in outcomes:
            player = outcome.get("player")
            if player:
                injury_data = InjuryImpactAnalyzer.get_injury_impact(player)
                injury_analysis[player] = injury_data
                total_injury_impact += injury_data["performance_adjustment"]
        
        # 6. FORM-ADJUSTED CORRELATION
        form_adjustment = 0.0
        for outcome in outcomes:
            player = outcome.get("player")
            stat_type = outcome.get("type")
            if player and stat_type:
                form_key = f"{player}_{stat_type}"
                if form_key in form_analysis:
                    form_factor = form_analysis[form_key]["factor"]
                    # Convert form factor to correlation adjustment
                    if form_factor > 1.15:  # Hot form
                        form_adjustment += 0.05
                    elif form_factor < 0.85:  # Cold form
                        form_adjustment -= 0.05
        
        # 7. CALCULATE FINAL CORRELATION SCORE
        final_correlation = synergy_adjusted_correlation + form_adjustment
        final_correlation = max(0.1, min(1.0, final_correlation))
        
        # 8. ENHANCED WEATHER IMPACT
        weather_impact = SGMAnalytics.calculate_weather_impact(weather_data, outcomes)
        
        # 9. CALCULATE PREDICTED PROBABILITY
        base_probability = final_correlation
        weather_adjusted_prob = base_probability * (1 + weather_impact["total_impact"])
        injury_adjusted_prob = weather_adjusted_prob * (1 + total_injury_impact)
        final_probability = max(0.05, min(0.95, injury_adjusted_prob))
        
        # 10. MARKET TIMING ANALYSIS
        market_timing = {}
        for outcome in outcomes:
            player = outcome.get("player")
            stat_type = outcome.get("type")
            value = outcome.get("value")
            if player and stat_type and value:
                bet_description = f"{player} {value}+ {stat_type}"
                timing_strategy = MarketMonitor.get_market_timing_strategy(bet_description)
                market_timing[bet_description] = timing_strategy
        
        # 11. VALUE RATING AND STAKE RECOMMENDATION
        mock_market_odds = 3.50  # Mock odds - in production get from odds API
        value_rating = SGMAnalytics.calculate_value_rating(final_probability, mock_market_odds)
        
        # Enhanced stake recommendation considering form and injuries
        base_stake = max(0, min(0.1, value_rating * 0.25)) if value_rating > 0 else 0
        
        # Adjust stake based on confidence factors
        confidence_factors = {
            "form_confidence": sum([1 for fa in form_analysis.values() if fa["confidence"] == "High"]) / len(form_analysis) if form_analysis else 0.5,
            "synergy_confidence": 1.0 if synergy_analysis["synergy_rating"] in ["Excellent", "Good"] else 0.7,
            "injury_confidence": 1.0 if total_injury_impact > -0.05 else 0.6  # Reduce confidence if significant injuries
        }
        
        avg_confidence = sum(confidence_factors.values()) / len(confidence_factors)
        recommended_stake = base_stake * avg_confidence
        
        # 12. COMPILE COMPREHENSIVE RESULT
        result = {
            "match_id": match_id,
            "outcomes": outcomes,
            "advanced_analysis": {
                "correlation_analysis": {
                    "base_correlation": round(base_correlation, 3),
                    "synergy_adjusted": round(synergy_adjusted_correlation, 3),
                    "form_adjusted": round(final_correlation, 3),
                    "final_score": round(final_correlation, 3)
                },
                "recent_form": form_analysis,
                "teammate_synergy": synergy_analysis,
                "defensive_matchups": matchup_analysis,
                "injury_impact": {
                    "player_analysis": injury_analysis,
                    "total_impact": round(total_injury_impact, 3),
                    "overall_rating": "Healthy" if total_injury_impact > -0.05 else "Concerns"
                },
                "weather_impact": weather_impact,
                "market_timing": market_timing,
                "predicted_probability": round(final_probability, 3),
                "value_rating": round(value_rating, 3),
                "confidence_factors": confidence_factors,
                "recommended_stake": round(recommended_stake, 3)
            },
            "summary": {
                "confidence": "High" if final_correlation > 0.7 and avg_confidence > 0.8 else 
                           "Medium" if final_correlation > 0.5 and avg_confidence > 0.6 else "Low",
                "key_strengths": SGMAnalytics._identify_key_strengths(synergy_analysis, form_analysis, injury_analysis),
                "key_concerns": SGMAnalytics._identify_key_concerns(form_analysis, injury_analysis, weather_impact),
                "recommendation": SGMAnalytics._generate_final_recommendation(value_rating, final_correlation, avg_confidence)
            },
            "weather_conditions": weather_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store analysis in database
        analysis_collection = db["sgm_analyses"]
        result_copy = result.copy()
        await analysis_collection.insert_one(result_copy)
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Advanced SGM analysis error: {str(e)}")

@app.get("/api/sgm/history")
async def get_sgm_history():
    """Get historical SGM analyses"""
    analysis_collection = db["sgm_analyses"]
    analyses = await analysis_collection.find({}).sort("timestamp", -1).limit(20).to_list(length=20)
    
    # Convert ObjectId to string for JSON serialization
    for analysis in analyses:
        analysis["_id"] = str(analysis["_id"])
    
    return analyses

@app.get("/api/team/{team_name}/stats")
async def get_team_statistics(team_name: str):
    """Get team defensive/offensive statistics"""
    return await AFLDataService.get_team_stats(team_name)

@app.get("/api/matchup/{player_name}/{opponent_team}")
async def get_defensive_matchup(player_name: str, opponent_team: str):
    """Get detailed defensive matchup analysis for player vs team"""
    try:
        import sys
        sys.path.append('/app')
        from enhanced_player_data import COMPREHENSIVE_AFL_PLAYERS, TEAM_DEFENSIVE_STATS
        
        # Find player data
        player_data = None
        for player in COMPREHENSIVE_AFL_PLAYERS:
            if player["name"].lower() == player_name.lower():
                player_data = player
                break
        
        if not player_data:
            raise HTTPException(404, f"Player {player_name} not found")
        
        if opponent_team not in TEAM_DEFENSIVE_STATS:
            raise HTTPException(404, f"Team {opponent_team} not found")
        
        team_defense = TEAM_DEFENSIVE_STATS[opponent_team]
        
        # Calculate matchup analysis
        analysis = {
            "player": player_data["name"],
            "opponent": opponent_team,
            "base_averages": {
                "disposals": player_data["avg_disposals"],
                "goals": player_data["avg_goals"],
                "marks": player_data["avg_marks"],
                "tackles": player_data["avg_tackles"]
            },
            "defensive_matchup": {
                "opponent_allows_per_game": {
                    "total_disposals": team_defense["disposals_allowed_per_game"],
                    "midfielder_disposals": team_defense["midfielder_disposals_allowed"],
                    "goals": team_defense["goals_allowed_per_game"],
                    "forward_goals": team_defense["forward_goals_allowed"]
                },
                "league_rank": SGMAnalytics._calculate_defensive_rank(opponent_team, team_defense),
                "matchup_rating": SGMAnalytics._calculate_matchup_difficulty(player_data, team_defense)
            },
            "projected_performance": {
                "disposals": SGMAnalytics._project_stat_vs_defense(
                    player_data["avg_disposals"], 
                    team_defense["midfielder_disposals_allowed"], 
                    375.0  # League average
                ),
                "goals": SGMAnalytics._project_stat_vs_defense(
                    player_data["avg_goals"],
                    team_defense["forward_goals_allowed"],
                    8.5  # League average
                )
            },
            "key_insights": SGMAnalytics._generate_matchup_insights(player_data, opponent_team, team_defense)
        }
        
        return analysis
        
    except Exception as e:
        raise HTTPException(500, f"Matchup analysis error: {str(e)}")

@app.get("/api/player/{player_name}/venue-performance")
async def get_player_venue_performance(player_name: str):
    """Get player's performance breakdown by venue"""
    try:
        import sys
        sys.path.append('/app')
        from enhanced_player_data import COMPREHENSIVE_AFL_PLAYERS
        
        # Find player data
        player_data = None
        for player in COMPREHENSIVE_AFL_PLAYERS:
            if player["name"].lower() == player_name.lower():
                player_data = player
                break
        
        if not player_data:
            raise HTTPException(404, f"Player {player_name} not found")
        
        venue_analysis = {
            "player": player_data["name"],
            "team": player_data["team"],
            "season_averages": {
                "disposals": player_data["avg_disposals"],
                "goals": player_data["avg_goals"],
                "marks": player_data["avg_marks"]
            },
            "venue_breakdown": {},
            "best_venues": [],
            "worst_venues": []
        }
        
        # Process venue performance
        venue_performances = []
        for venue, stats in player_data["venue_performance"].items():
            venue_analysis["venue_breakdown"][venue] = {
                "disposals": stats["disposals"],
                "goals": stats["goals"],
                "disposal_difference": stats["disposals"] - player_data["avg_disposals"],
                "goal_difference": stats["goals"] - player_data["avg_goals"]
            }
            
            # Calculate overall performance score
            disposal_factor = stats["disposals"] / player_data["avg_disposals"]
            goal_factor = stats["goals"] / player_data["avg_goals"] if player_data["avg_goals"] > 0 else 1
            overall_score = (disposal_factor + goal_factor) / 2
            
            venue_performances.append({
                "venue": venue,
                "score": overall_score,
                "disposals": stats["disposals"],
                "goals": stats["goals"]
            })
        
        # Sort venues by performance
        venue_performances.sort(key=lambda x: x["score"], reverse=True)
        
        venue_analysis["best_venues"] = venue_performances[:3]
        venue_analysis["worst_venues"] = venue_performances[-3:]
        
        return venue_analysis
        
    except Exception as e:
        raise HTTPException(500, f"Venue analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
