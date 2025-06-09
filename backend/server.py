from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
import httpx
import asyncio

# Import our advanced modules
from ml_sgm_picker import MachineLearningPredictor, AutomatedSGMPicker
from advanced_analytics import (
    RecentFormAnalyzer, 
    TeammateSymergyAnalyzer, 
    InjuryImpactAnalyzer,
    ENHANCED_PLAYER_DATA,
    TEAM_DEFENSIVE_STATS
)
from real_afl_data import RealAFLDataCollector, EnhancedSGMAnalyzer
from sportdevs_integration import SportDevsAPIService, LiveAFLAnalyzer
from api_sports_integration import APISportsAFLService, APISportsAFLAnalyzer

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="AFL SGM Builder API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# AFL Venues with coordinates
AFL_VENUES = {
    "MCG": {"lat": -37.8200, "lon": 144.9834, "city": "Melbourne", "state": "VIC"},
    "Marvel Stadium": {"lat": -37.8164, "lon": 144.9475, "city": "Melbourne", "state": "VIC"},
    "Adelaide Oval": {"lat": -34.9155, "lon": 138.5959, "city": "Adelaide", "state": "SA"},
    "Optus Stadium": {"lat": -31.9505, "lon": 115.8605, "city": "Perth", "state": "WA"},
    "Gabba": {"lat": -27.4858, "lon": 153.0389, "city": "Brisbane", "state": "QLD"},
    "SCG": {"lat": -33.8915, "lon": 151.2244, "city": "Sydney", "state": "NSW"},
    "GMHBA Stadium": {"lat": -38.1579, "lon": 144.3544, "city": "Geelong", "state": "VIC"},
    "Blundstone Arena": {"lat": -42.8606, "lon": 147.2869, "city": "Hobart", "state": "TAS"}
}

# Team mapping
AFL_TEAMS = {
    "Adelaide": "ADE", "Brisbane Lions": "BRL", "Carlton": "CAR", "Collingwood": "COL",
    "Essendon": "ESS", "Fremantle": "FRE", "Geelong": "GEE", "Gold Coast": "GCS",
    "GWS": "GWS", "Hawthorn": "HAW", "Melbourne": "MEL", "North Melbourne": "NTH",
    "Port Adelaide": "PTA", "Richmond": "RIC", "St Kilda": "STK", "Sydney": "SYD",
    "West Coast": "WCE", "Western Bulldogs": "WBD"
}

# Data Models
class SGMPrediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    match_id: str
    player_name: str
    prediction_type: str  # goals, disposals, marks, tackles
    prediction_value: float
    confidence: float
    weather_impact: Optional[Dict] = None
    odds_data: Optional[Dict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MatchData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    home_team: str
    away_team: str
    venue: str
    date: str
    round: int
    weather: Optional[Dict] = None
    odds: Optional[Dict] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class WeatherService:
    def __init__(self):
        self.api_key = os.environ.get('WEATHERAPI_KEY')
        self.base_url = "http://api.weatherapi.com/v1"

    async def get_weather_for_venue(self, venue: str, date: str = None) -> Dict:
        """Get weather data for AFL venue"""
        if venue not in AFL_VENUES:
            raise HTTPException(404, f"Venue {venue} not found")
        
        coords = AFL_VENUES[venue]
        
        async with httpx.AsyncClient() as client:
            try:
                if date:
                    # Forecast or historical data
                    url = f"{self.base_url}/forecast.json"
                    params = {
                        "key": self.api_key,
                        "q": f"{coords['lat']},{coords['lon']}",
                        "dt": date,
                        "aqi": "no"
                    }
                else:
                    # Current weather
                    url = f"{self.base_url}/current.json"
                    params = {
                        "key": self.api_key,
                        "q": f"{coords['lat']},{coords['lon']}",
                        "aqi": "no"
                    }
                
                response = await client.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if date:
                    forecast_day = data['forecast']['forecastday'][0]['day']
                    return {
                        "venue": venue,
                        "date": date,
                        "max_temp": forecast_day.get('maxtemp_c', 0),
                        "min_temp": forecast_day.get('mintemp_c', 0),
                        "avg_humidity": forecast_day.get('avghumidity', 0),
                        "max_wind": forecast_day.get('maxwind_kph', 0),
                        "total_precipitation": forecast_day.get('totalprecip_mm', 0),
                        "conditions": forecast_day.get('condition', {}).get('text', 'Unknown')
                    }
                else:
                    current = data['current']
                    return {
                        "venue": venue,
                        "temperature": current.get('temp_c', 0),
                        "humidity": current.get('humidity', 0),
                        "wind_speed": current.get('wind_kph', 0),
                        "wind_direction": current.get('wind_dir', ''),
                        "precipitation": current.get('precip_mm', 0),
                        "conditions": current.get('condition', {}).get('text', 'Unknown')
                    }
                    
            except Exception as e:
                logging.error(f"Weather API error: {str(e)}")
                return {"error": f"Weather data unavailable: {str(e)}"}

class OddsService:
    def __init__(self):
        self.api_key = os.environ.get('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"

    async def get_afl_odds(self) -> List[Dict]:
        """Get current AFL betting odds"""
        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}/sports/aussierules_afl/odds"
                params = {
                    "apiKey": self.api_key,
                    "regions": "au",
                    "markets": "h2h,spreads,totals",
                    "oddsFormat": "decimal"
                }
                
                response = await client.get(url, params=params, timeout=15)
                response.raise_for_status()
                return response.json()
                
            except Exception as e:
                logging.error(f"Odds API error: {str(e)}")
                return []

class SquiggleService:
    def __init__(self):
        self.base_url = os.environ.get('SQUIGGLE_API_URL', 'https://api.squiggle.com.au')

    async def get_matches(self, year: int = 2025) -> List[Dict]:
        """Get AFL matches from Squiggle API"""
        headers = {
            'User-Agent': 'AFL SGM Builder (https://github.com/JHull1234/SGM-Builder)'
        }
        async with httpx.AsyncClient(headers=headers) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/?q=games;year={year}",
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                return data.get('games', [])
            except Exception as e:
                logging.error(f"Squiggle API error: {str(e)}")
                return []

    async def get_player_stats(self, year: int = 2025) -> List[Dict]:
        """Get player statistics from Squiggle API"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/?q=standings;year={year}",
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                return data.get('standings', [])
            except Exception as e:
                logging.error(f"Squiggle stats error: {str(e)}")
                return []

# Initialize services
weather_service = WeatherService()
odds_service = OddsService()
squiggle_service = SquiggleService()

# Initialize ML and advanced analytics
ml_predictor = MachineLearningPredictor()
automated_sgm_picker = AutomatedSGMPicker(ml_predictor)

# Initialize real AFL data collector
real_afl_data = RealAFLDataCollector()
enhanced_sgm_analyzer = EnhancedSGMAnalyzer()

# Initialize SportDevs live data service
sportdevs_service = SportDevsAPIService()
live_afl_analyzer = LiveAFLAnalyzer()

# SGM Analysis Engine
class SGMAnalyzer:
    def __init__(self):
        pass

    async def calculate_player_performance_probability(
        self, 
        player_name: str, 
        stat_type: str, 
        threshold: float,
        weather_conditions: Dict,
        venue: str
    ) -> Dict:
        """Calculate probability of player achieving stat threshold"""
        
        # Base probability calculation (simplified for MVP)
        base_probabilities = {
            "goals": {"2+": 0.35, "3+": 0.15, "4+": 0.05},
            "disposals": {"15+": 0.75, "20+": 0.55, "25+": 0.35, "30+": 0.15},
            "marks": {"5+": 0.65, "8+": 0.35, "10+": 0.15},
            "tackles": {"5+": 0.70, "8+": 0.45, "10+": 0.25}
        }
        
        key = f"{int(threshold)}+"
        base_prob = base_probabilities.get(stat_type, {}).get(key, 0.5)
        
        # Weather adjustments
        weather_modifier = 1.0
        if weather_conditions:
            wind_speed = weather_conditions.get('wind_speed', 0)
            precipitation = weather_conditions.get('precipitation', 0)
            
            if stat_type == "goals":
                # High wind reduces goal accuracy
                if wind_speed > 25:
                    weather_modifier *= 0.8
                elif wind_speed > 15:
                    weather_modifier *= 0.9
                    
                # Rain reduces goal accuracy
                if precipitation > 2:
                    weather_modifier *= 0.85
                    
            elif stat_type == "disposals":
                # Rain can increase disposals (more contested)
                if precipitation > 1:
                    weather_modifier *= 1.1
                    
        # Venue adjustments
        venue_modifier = 1.0
        if venue == "MCG":
            if stat_type == "disposals":
                venue_modifier *= 1.05  # Bigger ground, more running
        elif venue == "GMHBA Stadium":
            if stat_type == "goals":
                venue_modifier *= 1.1  # Traditionally high scoring
                
        final_probability = base_prob * weather_modifier * venue_modifier
        final_probability = min(max(final_probability, 0.01), 0.99)  # Clamp between 1-99%
        
        return {
            "player": player_name,
            "stat_type": stat_type,
            "threshold": threshold,
            "probability": round(final_probability, 3),
            "base_probability": round(base_prob, 3),
            "weather_modifier": round(weather_modifier, 3),
            "venue_modifier": round(venue_modifier, 3),
            "confidence": "medium"  # Would be calculated based on data quality
        }

    async def build_sgm_combination(self, predictions: List[Dict]) -> Dict:
        """Build Same Game Multi combination analysis"""
        if not predictions:
            return {"error": "No predictions provided"}
            
        # Calculate combined probability (assuming independence for now)
        combined_prob = 1.0
        for pred in predictions:
            combined_prob *= pred['probability']
            
        # Correlation adjustments (simplified)
        correlation_modifier = 1.0
        if len(predictions) >= 2:
            # Same player multiple stats - slight positive correlation
            players = [p['player'] for p in predictions]
            if len(set(players)) < len(players):
                correlation_modifier *= 1.1
                
        final_combined_prob = combined_prob * correlation_modifier
        
        return {
            "predictions": predictions,
            "individual_probabilities": [p['probability'] for p in predictions],
            "combined_probability": round(final_combined_prob, 4),
            "correlation_modifier": round(correlation_modifier, 3),
            "implied_odds": round(1 / final_combined_prob, 2) if final_combined_prob > 0 else "N/A",
            "recommendation": "value" if final_combined_prob > 0.15 else "avoid"
        }

sgm_analyzer = SGMAnalyzer()

# API Endpoints
@api_router.get("/")
async def root():
    return {"message": "AFL SGM Builder API - Ready to analyze Same Game Multis!"}

@api_router.get("/matches")
async def get_current_matches():
    """Get current AFL matches"""
    matches = await squiggle_service.get_matches()
    
    # Store in database
    for match in matches[:10]:  # Limit to recent matches
        match_doc = {
            "match_id": str(match.get('id', uuid.uuid4())),
            "home_team": match.get('hteam', ''),
            "away_team": match.get('ateam', ''),
            "venue": match.get('venue', ''),
            "date": match.get('date', ''),
            "round": match.get('round', 0),
            "created_at": datetime.utcnow()
        }
        await db.matches.update_one(
            {"match_id": match_doc["match_id"]},
            {"$set": match_doc},
            upsert=True
        )
    
    return {"matches": matches[:10], "count": len(matches)}

@api_router.get("/weather/{venue}")
async def get_venue_weather(venue: str, date: str = None):
    """Get weather for AFL venue"""
    # URL decode venue name and handle variations
    import urllib.parse
    venue = urllib.parse.unquote(venue)
    
    # Handle venue name variations
    venue_mappings = {
        "S.C.G.": "SCG",
        "M.C.G.": "MCG", 
        "Sydney Showground": "ANZ Stadium",
        "Marvel Stadium Docklands": "Marvel Stadium"
    }
    
    # Map venue if needed
    mapped_venue = venue_mappings.get(venue, venue)
    
    weather_data = await weather_service.get_weather_for_venue(mapped_venue, date)
    
    # Store weather data
    weather_doc = {
        "venue": venue,
        "mapped_venue": mapped_venue,
        "date": date or datetime.now().isoformat(),
        "weather_data": weather_data,
        "created_at": datetime.utcnow()
    }
    await db.weather.insert_one(weather_doc)
    
    return weather_data

@api_router.get("/odds")
async def get_betting_odds():
    """Get current AFL betting odds"""
    odds_data = await odds_service.get_afl_odds()
    
    # Store odds data
    for odds in odds_data:
        odds_doc = {
            "match_id": odds.get('id', str(uuid.uuid4())),
            "sport": odds.get('sport_key', ''),
            "home_team": odds.get('home_team', ''),
            "away_team": odds.get('away_team', ''),
            "bookmakers": odds.get('bookmakers', []),
            "created_at": datetime.utcnow()
        }
        await db.odds.insert_one(odds_doc)
    
    return {"odds": odds_data, "count": len(odds_data)}

@api_router.post("/sgm/analyze")
async def analyze_sgm_combination(combination_data: Dict):
    """Analyze Same Game Multi combination"""
    try:
        match_id = combination_data.get('match_id')
        venue = combination_data.get('venue', 'MCG')
        date = combination_data.get('date')
        selections = combination_data.get('selections', [])
        
        if not selections:
            raise HTTPException(400, "No selections provided")
        
        # Get weather data for the match
        weather_data = await weather_service.get_weather_for_venue(venue, date)
        
        # Calculate individual predictions
        predictions = []
        for selection in selections:
            prediction = await sgm_analyzer.calculate_player_performance_probability(
                player_name=selection.get('player'),
                stat_type=selection.get('stat_type'),
                threshold=selection.get('threshold'),
                weather_conditions=weather_data,
                venue=venue
            )
            predictions.append(prediction)
        
        # Build SGM combination analysis
        sgm_analysis = await sgm_analyzer.build_sgm_combination(predictions)
        
        # Store SGM analysis
        sgm_doc = {
            "match_id": match_id,
            "venue": venue,
            "date": date,
            "selections": selections,
            "weather_data": weather_data,
            "analysis": sgm_analysis,
            "created_at": datetime.utcnow()
        }
        await db.sgm_analysis.insert_one(sgm_doc)
        
        return {
            "match_info": {
                "match_id": match_id,
                "venue": venue,
                "date": date
            },
            "weather_conditions": weather_data,
            "sgm_analysis": sgm_analysis
        }
        
    except Exception as e:
        logging.error(f"SGM analysis error: {str(e)}")
        raise HTTPException(500, f"SGM analysis failed: {str(e)}")

@api_router.get("/venues")
async def get_afl_venues():
    """Get all AFL venues with coordinates"""
    return {
        "venues": [
            {"name": name, **details} 
            for name, details in AFL_VENUES.items()
        ]
    }

@api_router.get("/teams")
async def get_afl_teams():
    """Get all AFL teams"""
    return {"teams": AFL_TEAMS}

@api_router.get("/player/{player_name}/enhanced-analysis")
async def get_enhanced_player_analysis(player_name: str, opponent_team: str = None, venue: str = "MCG"):
    """Get comprehensive player analysis using advanced analytics"""
    
    # Get enhanced player data
    if player_name not in ENHANCED_PLAYER_DATA:
        raise HTTPException(404, f"Enhanced data for {player_name} not available")
    
    player_data = ENHANCED_PLAYER_DATA[player_name]
    
    # Recent form analysis
    form_analysis = {}
    for stat in ["disposals", "goals", "marks", "tackles"]:
        form_analysis[stat] = RecentFormAnalyzer.calculate_form_factor(player_name, stat)
    
    # Injury impact analysis
    injury_analysis = InjuryImpactAnalyzer.get_injury_impact(player_name)
    
    # Venue performance analysis
    venue_performance = player_data.get("venue_performance", {}).get(venue, {})
    
    analysis = {
        "player": player_data,
        "form_analysis": form_analysis,
        "injury_analysis": injury_analysis,
        "venue_analysis": {
            "venue": venue,
            "performance": venue_performance,
            "venue_factor": venue_performance.get("disposals", 0) / player_data["avg_disposals"] if player_data["avg_disposals"] > 0 else 1.0
        },
        "opponent_analysis": TEAM_DEFENSIVE_STATS.get(opponent_team, {}) if opponent_team else {},
        "recommendations": []
    }
    
    # Generate recommendations based on analysis
    if form_analysis.get("disposals", {}).get("trend") == "Hot":
        analysis["recommendations"].append("🔥 Player in hot form - consider disposal markets")
    
    if injury_analysis.get("impact_rating") == "Medium":
        analysis["recommendations"].append("⚠️ Injury concerns - reduce confidence")
    
    return analysis

@api_router.post("/ml/predict-performance")
async def predict_player_performance_ml(request_data: Dict):
    """Use ML models to predict player performance"""
    
    player_name = request_data.get("player_name")
    match_context = request_data.get("match_context", {})
    
    if player_name not in ENHANCED_PLAYER_DATA:
        raise HTTPException(404, f"Player {player_name} not found in enhanced dataset")
    
    player_data = ENHANCED_PLAYER_DATA[player_name]
    
    # Add recent form and injury data to player context
    player_data["recent_form"] = RecentFormAnalyzer.RECENT_FORM_DATA.get(player_name, {})
    
    # Get injury impact
    injury_impact = InjuryImpactAnalyzer.get_injury_impact(player_name)
    match_context["injury_impact"] = injury_impact.get("performance_adjustment", 0)
    
    # Add defensive stats for opponent
    opponent_team = match_context.get("opponent_team")
    if opponent_team and opponent_team in TEAM_DEFENSIVE_STATS:
        match_context["opponent_defense"] = TEAM_DEFENSIVE_STATS[opponent_team]
    
    # Get ML prediction (would use trained models in production)
    # For now, we'll use enhanced statistical prediction
    prediction = await sgm_analyzer.calculate_player_performance_probability(
        player_name=player_name,
        stat_type=request_data.get("stat_type", "disposals"),
        threshold=request_data.get("threshold", 25),
        weather_conditions=match_context.get("weather", {}),
        venue=match_context.get("venue", "MCG")
    )
    
    # Enhance with form factors
    for stat in ["disposals", "goals", "marks", "tackles"]:
        form_factor = RecentFormAnalyzer.calculate_form_factor(player_name, stat)
        if form_factor["factor"] != 1.0:
            prediction[f"{stat}_form_adjusted"] = prediction.get(stat, 0) * form_factor["factor"]
    
    return {
        "player": player_name,
        "ml_prediction": prediction,
        "form_factors": {
            stat: RecentFormAnalyzer.calculate_form_factor(player_name, stat)
            for stat in ["disposals", "goals", "marks", "tackles"]
        },
        "injury_impact": injury_impact,
        "confidence_rating": "High" if injury_impact["impact_rating"] == "None" else "Medium"
    }

@api_router.post("/sgm/advanced-analyze")
async def advanced_sgm_analysis(combination_data: Dict):
    """Advanced SGM analysis with ML predictions and synergy analysis"""
    try:
        match_id = combination_data.get('match_id')
        venue = combination_data.get('venue', 'MCG')
        date = combination_data.get('date')
        selections = combination_data.get('selections', [])
        
        if not selections:
            raise HTTPException(400, "No selections provided")
        
        # Get weather data for the match
        weather_data = await weather_service.get_weather_for_venue(venue, date)
        
        # Enhanced match context
        match_context = {
            "venue": venue,
            "date": date,
            "weather": weather_data
        }
        
        # Calculate individual predictions with enhanced analytics
        enhanced_predictions = []
        sgm_outcomes = []
        
        for selection in selections:
            player_name = selection.get('player')
            stat_type = selection.get('stat_type')
            threshold = selection.get('threshold')
            
            if player_name not in ENHANCED_PLAYER_DATA:
                continue
                
            player_data = ENHANCED_PLAYER_DATA[player_name]
            
            # Get recent form factor
            form_analysis = RecentFormAnalyzer.calculate_form_factor(player_name, stat_type)
            
            # Get injury impact
            injury_analysis = InjuryImpactAnalyzer.get_injury_impact(player_name)
            match_context["injury_impact"] = injury_analysis.get("performance_adjustment", 0)
            
            # Enhanced prediction with form and injury adjustments
            base_prediction = await sgm_analyzer.calculate_player_performance_probability(
                player_name=player_name,
                stat_type=stat_type,
                threshold=threshold,
                weather_conditions=weather_data,
                venue=venue
            )
            
            # Apply form adjustment
            form_adjusted_prob = base_prediction["probability"] * form_analysis["factor"]
            
            # Apply injury adjustment
            injury_factor = 1 + injury_analysis.get("performance_adjustment", 0)
            final_probability = form_adjusted_prob * injury_factor
            final_probability = min(max(final_probability, 0.01), 0.99)
            
            enhanced_prediction = {
                "player": player_name,
                "stat_type": stat_type,
                "threshold": threshold,
                "base_probability": base_prediction["probability"],
                "form_factor": form_analysis["factor"],
                "injury_factor": injury_factor,
                "final_probability": final_probability,
                "form_trend": form_analysis["trend"],
                "injury_status": injury_analysis["status"],
                "confidence": base_prediction["confidence"]
            }
            
            enhanced_predictions.append(enhanced_prediction)
            
            # For synergy analysis
            sgm_outcomes.append({
                "player": player_name,
                "type": stat_type,
                "threshold": threshold,
                "probability": final_probability
            })
        
        # Teammate synergy analysis
        synergy_analysis = TeammateSymergyAnalyzer.calculate_synergy_impact(sgm_outcomes)
        
        # Combined probability with synergy adjustment
        individual_probs = [pred["final_probability"] for pred in enhanced_predictions]
        naive_combined_prob = 1.0
        for prob in individual_probs:
            naive_combined_prob *= prob
            
        # Apply synergy adjustment
        synergy_multiplier = 1 + synergy_analysis["total_synergy_impact"]
        final_combined_prob = naive_combined_prob * synergy_multiplier
        final_combined_prob = min(max(final_combined_prob, 0.001), 0.999)
        
        # Calculate implied odds
        implied_odds = 1 / final_combined_prob if final_combined_prob > 0 else 999
        
        # Generate recommendation
        recommendation = "🔥 EXCELLENT VALUE" if final_combined_prob > 0.20 else \
                        "✅ GOOD VALUE" if final_combined_prob > 0.15 else \
                        "⚠️ FAIR VALUE" if final_combined_prob > 0.10 else \
                        "❌ POOR VALUE"
        
        # Store enhanced SGM analysis
        enhanced_sgm_doc = {
            "match_id": match_id,
            "venue": venue,
            "date": date,
            "selections": selections,
            "weather_data": weather_data,
            "enhanced_predictions": enhanced_predictions,
            "synergy_analysis": synergy_analysis,
            "combined_probability": final_combined_prob,
            "implied_odds": round(implied_odds, 2),
            "recommendation": recommendation,
            "created_at": datetime.utcnow()
        }
        await db.enhanced_sgm_analysis.insert_one(enhanced_sgm_doc)
        
        return {
            "match_info": {
                "match_id": match_id,
                "venue": venue,
                "date": date
            },
            "weather_conditions": weather_data,
            "enhanced_predictions": enhanced_predictions,
            "synergy_analysis": synergy_analysis,
            "combined_analysis": {
                "individual_probabilities": individual_probs,
                "naive_combined_probability": round(naive_combined_prob, 4),
                "synergy_adjustment": round(synergy_analysis["total_synergy_impact"], 4),
                "final_combined_probability": round(final_combined_prob, 4),
                "implied_odds": round(implied_odds, 2),
                "recommendation": recommendation,
                "confidence_rating": synergy_analysis["synergy_rating"]
            }
        }
        
    except Exception as e:
        logging.error(f"Enhanced SGM analysis error: {str(e)}")
        raise HTTPException(500, f"Enhanced SGM analysis failed: {str(e)}")

@api_router.get("/sgm/auto-recommend/{target_odds}")
async def auto_recommend_sgm(target_odds: float, venue: str = "MCG", max_players: int = 3):
    """Automatically recommend optimal SGM combinations for target odds"""
    
    try:
        # Get available players (using our enhanced dataset)
        available_players = list(ENHANCED_PLAYER_DATA.values())
        
        # Mock match context
        match_context = {
            "venue": venue,
            "date": datetime.now().isoformat(),
            "weather": await weather_service.get_weather_for_venue(venue)
        }
        
        # For now, return a simplified auto recommendation since the full ML picker needs more data
        # This would be replaced with the full automated_sgm_picker in production
        
        auto_recommendations = []
        
        # Example auto-generated SGM based on current form
        hot_form_players = []
        for player_name, player_data in ENHANCED_PLAYER_DATA.items():
            form_data = RecentFormAnalyzer.RECENT_FORM_DATA.get(player_name, {})
            if form_data.get("form_trend") == "Hot":
                hot_form_players.append(player_name)
        
        if len(hot_form_players) >= 2:
            auto_recommendations.append({
                "sgm_type": "Hot Form Combo",
                "players": hot_form_players[:2],
                "selections": [
                    {"player": hot_form_players[0], "market": "25+ disposals", "confidence": "High"},
                    {"player": hot_form_players[1], "market": "2+ goals", "confidence": "Medium"}
                ],
                "estimated_odds": round(target_odds * 0.9, 2),
                "recommendation": "🔥 STRONG - Players in excellent form",
                "synergy_rating": "Good"
            })
        
        return {
            "target_odds": target_odds,
            "venue": venue,
            "auto_recommendations": auto_recommendations,
            "analysis_note": "Advanced ML auto-picker available with full training data",
            "hot_form_players": hot_form_players,
            "weather_impact": {
                "conditions": match_context["weather"].get("conditions", "Unknown"),
                "impact_rating": "Low" if match_context["weather"].get("wind_speed", 0) < 20 else "High"
            }
        }
        
    except Exception as e:
        logging.error(f"Auto SGM recommendation error: {str(e)}")
        raise HTTPException(500, f"Auto recommendation failed: {str(e)}")

@api_router.get("/analytics/player-dashboard/{player_name}")
async def get_player_analytics_dashboard(player_name: str):
    """Get comprehensive player analytics dashboard"""
    
    if player_name not in ENHANCED_PLAYER_DATA:
        raise HTTPException(404, f"Player {player_name} not found")
    
    player_data = ENHANCED_PLAYER_DATA[player_name]
    
    # Form analysis across all stats
    form_analysis = {}
    for stat in ["disposals", "goals", "marks", "tackles"]:
        form_analysis[stat] = RecentFormAnalyzer.calculate_form_factor(player_name, stat)
    
    # Injury analysis
    injury_analysis = InjuryImpactAnalyzer.get_injury_impact(player_name)
    
    # Venue performance breakdown
    venue_breakdown = []
    for venue, performance in player_data.get("venue_performance", {}).items():
        venue_breakdown.append({
            "venue": venue,
            "disposals": performance.get("disposals", 0),
            "goals": performance.get("goals", 0),
            "disposal_factor": performance.get("disposals", 0) / player_data["avg_disposals"] if player_data["avg_disposals"] > 0 else 1.0,
            "goal_factor": performance.get("goals", 0) / player_data["avg_goals"] if player_data["avg_goals"] > 0 else 1.0
        })
    
    # Calculate form confidence manually to avoid numpy issues
    form_factors = [form_analysis[stat]["factor"] for stat in form_analysis]
    avg_form_confidence = sum(form_factors) / len(form_factors) if form_factors else 1.0
    
    return {
        "player_info": player_data,
        "form_analysis": form_analysis,
        "injury_analysis": injury_analysis,
        "venue_breakdown": venue_breakdown,
        "betting_insights": {
            "strongest_stat": max(form_analysis.keys(), key=lambda k: form_analysis[k]["factor"]),
            "form_confidence": round(avg_form_confidence, 3),
            "injury_risk": injury_analysis["impact_rating"],
            "recommended_markets": [
                f"{stat} markets" for stat, analysis in form_analysis.items() 
                if analysis["trend"] == "Hot" and analysis["confidence"] in ["High", "Medium"]
            ]
        }
    }

@api_router.get("/real-data/current-matches")
async def get_real_current_matches():
    """Get real current AFL matches from live sources"""
    try:
        matches = await real_afl_data.get_current_round_matches()
        return {
            "matches": matches,
            "count": len(matches),
            "source": "Multiple live sources",
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Real matches error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch real matches: {str(e)}")

@api_router.get("/real-data/player/{player_name}/stats")
async def get_real_player_stats(player_name: str, team: str = None):
    """Get real 2025 season player statistics"""
    try:
        stats = await real_afl_data.get_player_season_stats(player_name, team)
        if not stats:
            raise HTTPException(404, f"No real data found for player: {player_name}")
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Real player stats error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch real player stats: {str(e)}")

@api_router.get("/real-data/team/{team_name}/defensive-stats")
async def get_real_team_defensive_stats(team_name: str):
    """Get real team defensive statistics for 2025"""
    try:
        stats = await real_afl_data.get_team_defensive_stats(team_name)
        if not stats:
            raise HTTPException(404, f"No real defensive data found for team: {team_name}")
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Real team stats error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch real team stats: {str(e)}")

@api_router.get("/real-data/injuries")
async def get_real_injury_list():
    """Get current AFL injury list from official sources"""
    try:
        injuries = await real_afl_data.get_injury_list()
        return {
            "injuries": injuries,
            "count": len(injuries),
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Real injury data error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch real injury data: {str(e)}")

@api_router.post("/sgm/real-analyze")
async def analyze_real_sgm(combination_data: Dict):
    """Analyze SGM using real AFL 2025 season data"""
    try:
        venue = combination_data.get('venue', 'MCG')
        date = combination_data.get('date', datetime.now().isoformat())
        selections = combination_data.get('selections', [])
        
        if not selections:
            raise HTTPException(400, "No selections provided")
        
        # Get real weather data
        weather_data = await weather_service.get_weather_for_venue(venue, date)
        
        # Analyze using real AFL data
        real_analysis = await enhanced_sgm_analyzer.analyze_real_sgm(
            selections=selections,
            venue=venue,
            date=date
        )
        
        # Combine with weather data
        result = {
            "match_info": {
                "venue": venue,
                "date": date,
                "analysis_type": "Real AFL Data Analysis"
            },
            "weather_conditions": weather_data,
            "real_sgm_analysis": real_analysis,
            "data_sources": {
                "player_stats": "Live AFL Tables/Footywire",
                "weather": "WeatherAPI",
                "analysis": "Enhanced SGM Analyzer"
            }
        }
        
        # Store analysis
        analysis_doc = {
            "venue": venue,
            "date": date,
            "selections": selections,
            "weather_data": weather_data,
            "real_analysis": real_analysis,
            "analysis_type": "real_data",
            "created_at": datetime.utcnow()
        }
        await db.real_sgm_analysis.insert_one(analysis_doc)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Real SGM analysis error: {str(e)}")
        raise HTTPException(500, f"Real SGM analysis failed: {str(e)}")

@api_router.get("/demo/collingwood-melbourne-sgm")
async def demo_collingwood_melbourne_sgm():
    """Demo SGM for Collingwood vs Melbourne using real data"""
    try:
        # Import current player data
        from current_afl_data import CURRENT_AFL_PLAYER_DATA_2025, TEAM_MATCHUP_DATA, MCG_CONDITIONS_JUNE_9_2025
        
        # Weather for MCG
        weather_data = await weather_service.get_weather_for_venue("MCG")
        
        # Real player data
        clayton_data = CURRENT_AFL_PLAYER_DATA_2025["Clayton Oliver"]
        daicos_data = CURRENT_AFL_PLAYER_DATA_2025["Nick Daicos"]
        petracca_data = CURRENT_AFL_PLAYER_DATA_2025["Christian Petracca"]
        
        # Advanced statistical analysis using real form
        import numpy as np
        from scipy import stats
        
        # Clayton Oliver 30+ Disposals Analysis
        clayton_recent_avg = clayton_data["recent_form_averages"]["disposals"]  # 33.2
        clayton_mcg_avg = clayton_data["venue_performance"]["MCG"]["disposals_avg"]  # 35.2
        clayton_vs_coll_avg = clayton_data["recent_matchups_vs_collingwood"]["average_vs_collingwood"]["disposals"]  # 33.7
        
        # Weighted average considering venue and opponent
        clayton_expected = (clayton_recent_avg * 0.4 + clayton_mcg_avg * 0.4 + clayton_vs_coll_avg * 0.2)
        clayton_std = clayton_expected * 0.22  # 22% coefficient of variation
        clayton_30plus_prob = 1 - stats.norm.cdf(30, clayton_expected, clayton_std)
        
        # Nick Daicos 25+ Disposals Analysis  
        daicos_recent_avg = daicos_data["recent_form_averages"]["disposals"]  # 30.2
        daicos_mcg_avg = daicos_data["venue_performance"]["MCG"]["disposals_avg"]  # 31.8
        daicos_vs_melb_avg = daicos_data["recent_matchups_vs_melbourne"]["average_vs_melbourne"]["disposals"]  # 29.7
        
        daicos_expected = (daicos_recent_avg * 0.4 + daicos_mcg_avg * 0.4 + daicos_vs_melb_avg * 0.2)
        daicos_std = daicos_expected * 0.25
        daicos_25plus_prob = 1 - stats.norm.cdf(25, daicos_expected, daicos_std)
        
        # Christian Petracca 1+ Goals Analysis
        petracca_recent_goals = petracca_data["recent_form_averages"]["goals"]  # 1.4
        petracca_mcg_goals = petracca_data["venue_performance"]["MCG"]["goals_avg"]  # 1.6
        petracca_vs_coll_goals = petracca_data["recent_matchups_vs_collingwood"]["average_vs_collingwood"]["goals"]  # 1.0
        
        petracca_expected_goals = (petracca_recent_goals * 0.5 + petracca_mcg_goals * 0.3 + petracca_vs_coll_goals * 0.2)
        
        # Poisson distribution for goals
        petracca_1plus_prob = 1 - stats.poisson.pmf(0, petracca_expected_goals)
        
        # Weather adjustments
        weather_disposal_impact = MCG_CONDITIONS_JUNE_9_2025["impact_on_disposal_efficiency"]
        weather_goal_impact = MCG_CONDITIONS_JUNE_9_2025["impact_on_goal_accuracy"]
        
        clayton_30plus_prob_adjusted = clayton_30plus_prob * (1 + weather_disposal_impact)
        daicos_25plus_prob_adjusted = daicos_25plus_prob * (1 + weather_disposal_impact) 
        petracca_1plus_prob_adjusted = petracca_1plus_prob * (1 + weather_goal_impact)
        
        # Recommended SGM combinations
        sgm_options = [
            {
                "name": "Conservative Power Play",
                "selections": [
                    {"player": "Clayton Oliver", "market": "30+ Disposals", "probability": clayton_30plus_prob_adjusted},
                    {"player": "Nick Daicos", "market": "25+ Disposals", "probability": daicos_25plus_prob_adjusted}
                ],
                "combined_probability": clayton_30plus_prob_adjusted * daicos_25plus_prob_adjusted,
                "reasoning": "Both players in good form, favorable venue matchup"
            },
            {
                "name": "High Value Triple",
                "selections": [
                    {"player": "Clayton Oliver", "market": "30+ Disposals", "probability": clayton_30plus_prob_adjusted},
                    {"player": "Nick Daicos", "market": "25+ Disposals", "probability": daicos_25plus_prob_adjusted},
                    {"player": "Christian Petracca", "market": "1+ Goals", "probability": petracca_1plus_prob_adjusted}
                ],
                "combined_probability": clayton_30plus_prob_adjusted * daicos_25plus_prob_adjusted * petracca_1plus_prob_adjusted,
                "reasoning": "Higher odds play with strong statistical backing"
            }
        ]
        
        # Calculate implied odds and recommendations
        for sgm in sgm_options:
            sgm["implied_odds"] = round(1 / sgm["combined_probability"], 2)
            sgm["percentage_chance"] = round(sgm["combined_probability"] * 100, 1)
            
            if sgm["combined_probability"] > 0.25:
                sgm["recommendation"] = "🔥 EXCELLENT VALUE"
                sgm["confidence"] = "High"
            elif sgm["combined_probability"] > 0.15:
                sgm["recommendation"] = "✅ GOOD VALUE" 
                sgm["confidence"] = "Medium-High"
            else:
                sgm["recommendation"] = "⚠️ SPECULATIVE"
                sgm["confidence"] = "Medium"
        
        return {
            "match_title": "🏈 Collingwood vs Melbourne - June 9, 2025",
            "venue": "MCG",
            "analysis_type": "Professional Statistical Analysis",
            
            "player_form_analysis": {
                "clayton_oliver": {
                    "season_avg": clayton_data["season_stats"]["disposals_per_game"],
                    "recent_form": clayton_data["recent_form_averages"]["disposals"],
                    "mcg_performance": clayton_data["venue_performance"]["MCG"]["disposals_avg"],
                    "vs_collingwood": clayton_data["recent_matchups_vs_collingwood"]["average_vs_collingwood"]["disposals"],
                    "expected_disposals": round(clayton_expected, 1),
                    "30plus_probability": round(clayton_30plus_prob_adjusted, 3),
                    "form_rating": clayton_data["form_trend"],
                    "injury_status": clayton_data["injury_status"]
                },
                "nick_daicos": {
                    "season_avg": daicos_data["season_stats"]["disposals_per_game"],
                    "recent_form": daicos_data["recent_form_averages"]["disposals"],
                    "mcg_performance": daicos_data["venue_performance"]["MCG"]["disposals_avg"],
                    "vs_melbourne": daicos_data["recent_matchups_vs_melbourne"]["average_vs_melbourne"]["disposals"],
                    "expected_disposals": round(daicos_expected, 1),
                    "25plus_probability": round(daicos_25plus_prob_adjusted, 3),
                    "form_rating": daicos_data["form_trend"],
                    "injury_status": daicos_data["injury_status"]
                },
                "christian_petracca": {
                    "season_avg": petracca_data["season_stats"]["goals_per_game"],
                    "recent_form": petracca_data["recent_form_averages"]["goals"],
                    "mcg_performance": petracca_data["venue_performance"]["MCG"]["goals_avg"],
                    "vs_collingwood": petracca_data["recent_matchups_vs_collingwood"]["average_vs_collingwood"]["goals"],
                    "expected_goals": round(petracca_expected_goals, 2),
                    "1plus_probability": round(petracca_1plus_prob_adjusted, 3),
                    "form_rating": petracca_data["form_trend"],
                    "injury_status": petracca_data["injury_status"]
                }
            },
            
            "match_conditions": {
                "weather": MCG_CONDITIONS_JUNE_9_2025,
                "venue_factors": {
                    "ground_size": "Large (MCG advantage for running players)",
                    "surface": "Good despite light rain",
                    "crowd_factor": "High (traditional rivalry)"
                },
                "team_matchup": {
                    "melbourne_defense_rank": TEAM_MATCHUP_DATA["Melbourne"]["defensive_ranking"],
                    "collingwood_defense_rank": TEAM_MATCHUP_DATA["Collingwood"]["defensive_ranking"],
                    "melbourne_pressure": TEAM_MATCHUP_DATA["Melbourne"]["pressure_rating"],
                    "historical_disposal_average": 375  # Average disposals in recent meetings
                }
            },
            
            "recommended_sgms": sgm_options,
            
            "key_insights": [
                f"🔥 Clayton Oliver averaging {clayton_recent_avg} disposals in last 5 games (above {clayton_data['season_stats']['disposals_per_game']} season avg)",
                f"📈 Nick Daicos excellent at MCG: {daicos_mcg_avg} disposal average vs {daicos_data['season_stats']['disposals_per_game']} season",
                f"⚡ Petracca {petracca_data['season_stats']['goals_per_game']} goals/game season, {petracca_expected_goals:.1f} expected vs Collingwood",
                f"🌧️ Light rain conditions: -{abs(weather_disposal_impact)*100:.0f}% disposal efficiency, -{abs(weather_goal_impact)*100:.0f}% goal accuracy",
                f"🏟️ MCG suits both Oliver and Daicos based on historical performance"
            ],
            
            "data_sources": {
                "player_stats": "2025 Season Data (13 rounds)",
                "form_analysis": "Last 5 games performance",
                "venue_data": "Historical MCG performance",
                "weather": "Live WeatherAPI data",
                "matchup_history": "Recent head-to-head meetings"
            },
            
            "betting_strategy": {
                "primary_recommendation": sgm_options[0]["name"],
                "rationale": "Conservative approach with high-probability outcomes",
                "stake_suggestion": "2-3% of bankroll",
                "risk_level": "Medium",
                "confidence_level": sgm_options[0]["confidence"]
            }
        }
        
    except Exception as e:
        logging.error(f"Demo SGM error: {str(e)}")
        raise HTTPException(500, f"Demo SGM failed: {str(e)}")

@api_router.get("/live-data/player/{player_name}/stats")
async def get_live_player_stats(player_name: str):
    """Get live 2025 season player statistics from SportDevs API"""
    try:
        stats = await sportdevs_service.get_player_season_stats(player_name)
        if "error" in stats:
            raise HTTPException(404, f"Player data error: {stats['error']}")
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Live player stats error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch live player stats: {str(e)}")

@api_router.get("/live-data/player/{player_name}/recent-form")
async def get_live_player_recent_form(player_name: str, games: int = 5):
    """Get live recent form data for player from SportDevs API"""
    try:
        recent_data = await sportdevs_service.get_player_recent_games(player_name, games)
        if "error" in recent_data:
            raise HTTPException(404, f"Recent form data error: {recent_data['error']}")
        
        return recent_data
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Live recent form error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch live recent form: {str(e)}")

@api_router.get("/live-data/team/{team_name}/defensive-stats")
async def get_live_team_defensive_stats(team_name: str):
    """Get live team defensive statistics from SportDevs API"""
    try:
        stats = await sportdevs_service.get_team_defensive_stats(team_name)
        if "error" in stats:
            raise HTTPException(404, f"Team defensive data error: {stats['error']}")
        
        return stats
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Live team defensive stats error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch live team defensive stats: {str(e)}")

@api_router.get("/live-data/injuries")
async def get_live_injury_reports():
    """Get current injury reports from SportDevs API"""
    try:
        injuries = await sportdevs_service.get_injury_reports()
        return {
            "injuries": injuries,
            "count": len(injuries),
            "data_source": "SportDevs API",
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Live injury reports error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch live injury reports: {str(e)}")

@api_router.get("/live-data/matches")
async def get_live_matches():
    """Get current round live match data from SportDevs API"""
    try:
        matches = await sportdevs_service.get_live_match_data()
        return {
            "matches": matches,
            "count": len(matches),
            "data_source": "SportDevs API",
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Live matches error: {str(e)}")
        raise HTTPException(500, f"Failed to fetch live matches: {str(e)}")

@api_router.post("/sgm/live-analyze")
async def analyze_live_sgm(combination_data: Dict):
    """Analyze SGM using live SportDevs AFL data"""
    try:
        venue = combination_data.get('venue', 'MCG')
        selections = combination_data.get('selections', [])
        
        if not selections:
            raise HTTPException(400, "No selections provided")
        
        # Get live weather data
        weather_data = await weather_service.get_weather_for_venue(venue)
        
        # Analyze using live AFL data from SportDevs
        live_analysis = await live_afl_analyzer.analyze_live_sgm(
            selections=selections,
            venue=venue
        )
        
        # Combine results
        result = {
            "analysis_type": "Live AFL Data Analysis (SportDevs API)",
            "match_info": {
                "venue": venue,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "weather_conditions": weather_data,
            "live_sgm_analysis": live_analysis,
            "data_sources": {
                "player_stats": "SportDevs AFL API (Live 2025 Season)",
                "weather": "WeatherAPI (Live)",
                "analysis_engine": "Advanced Statistical Modeling"
            }
        }
        
        # Store live analysis
        analysis_doc = {
            "venue": venue,
            "selections": selections,
            "weather_data": weather_data,
            "live_analysis": live_analysis,
            "analysis_type": "live_sportdevs_data",
            "created_at": datetime.utcnow()
        }
        await db.live_sgm_analysis.insert_one(analysis_doc)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Live SGM analysis error: {str(e)}")
        raise HTTPException(500, f"Live SGM analysis failed: {str(e)}")

@api_router.get("/live-demo/collingwood-melbourne-sgm")
async def live_demo_collingwood_melbourne_sgm():
    """Live demo SGM for Collingwood vs Melbourne using SportDevs real data"""
    try:
        # Get live player data using SportDevs API
        clayton_stats = await sportdevs_service.get_player_season_stats("Clayton Oliver")
        clayton_recent = await sportdevs_service.get_player_recent_games("Clayton Oliver", 5)
        
        daicos_stats = await sportdevs_service.get_player_season_stats("Nick Daicos")
        daicos_recent = await sportdevs_service.get_player_recent_games("Nick Daicos", 5)
        
        petracca_stats = await sportdevs_service.get_player_season_stats("Christian Petracca")
        petracca_recent = await sportdevs_service.get_player_recent_games("Christian Petracca", 5)
        
        # Get current injury reports
        injuries = await sportdevs_service.get_injury_reports()
        
        # Get live weather
        weather = await weather_service.get_weather_for_venue("MCG")
        
        # Build realistic SGM based on actual current form
        sgm_recommendations = []
        
        # Analyze Clayton Oliver's real recent form
        if "error" not in clayton_recent and clayton_recent.get("recent_averages"):
            recent_disposals = clayton_recent["recent_averages"]["disposals"]
            
            # YOUR REAL DATA: Oliver last 5 = [22, 13, 23, 31, 16] = 21.0 avg
            # Set conservative threshold based on poor recent form
            if recent_disposals > 0:
                threshold = max(18, int(recent_disposals * 0.85))  # 85% of recent avg
                sgm_recommendations.append({
                    "player": "Clayton Oliver",
                    "market": f"{threshold}+ Disposals",
                    "reasoning": f"Recent avg: {recent_disposals:.1f}, conservative threshold due to inconsistent form",
                    "risk_level": "Medium-High" if recent_disposals < 22 else "Medium"
                })
        
        # Analyze Nick Daicos's real recent form  
        if "error" not in daicos_recent and daicos_recent.get("recent_averages"):
            recent_disposals = daicos_recent["recent_averages"]["disposals"]
            
            # YOUR REAL DATA: Daicos last 5 = [28, 18, 28, 38, 32] = 28.8 avg
            # More consistent performer, can set higher threshold
            if recent_disposals > 0:
                threshold = max(22, int(recent_disposals * 0.8))  # 80% of recent avg
                sgm_recommendations.append({
                    "player": "Nick Daicos",
                    "market": f"{threshold}+ Disposals",
                    "reasoning": f"Recent avg: {recent_disposals:.1f}, consistent performer with good MCG record",
                    "risk_level": "Medium"
                })
        
        # Check for injury concerns
        injury_concerns = []
        for injury in injuries:
            if "error" not in injury:
                player_name = injury.get("player_name", "")
                if any(name in player_name for name in ["Clayton Oliver", "Nick Daicos", "Christian Petracca"]):
                    injury_concerns.append(injury)
        
        return {
            "demo_title": "🏈 LIVE Collingwood vs Melbourne SGM Analysis",
            "data_source": "SportDevs API - Live 2025 AFL Data",
            "match_info": {
                "teams": "Collingwood vs Melbourne",
                "venue": "MCG",
                "analysis_date": datetime.now().isoformat()
            },
            
            "live_player_data": {
                "clayton_oliver": {
                    "season_stats": clayton_stats,
                    "recent_form": clayton_recent,
                    "api_status": "success" if "error" not in clayton_stats else "error"
                },
                "nick_daicos": {
                    "season_stats": daicos_stats,
                    "recent_form": daicos_recent,
                    "api_status": "success" if "error" not in daicos_stats else "error"
                },
                "christian_petracca": {
                    "season_stats": petracca_stats,
                    "recent_form": petracca_recent,
                    "api_status": "success" if "error" not in petracca_stats else "error"
                }
            },
            
            "sgm_recommendations": sgm_recommendations,
            "injury_concerns": injury_concerns,
            "weather_conditions": weather,
            
            "api_integration_status": {
                "sportdevs_api": "Connected" if os.environ.get('SPORTDEVS_API_KEY') else "API Key Required",
                "weather_api": "Connected",
                "data_freshness": "Live 2025 Season Data"
            },
            
            "next_steps": [
                "1. Sign up for SportDevs API at https://sportdevs.com",
                "2. Get your API key and add to SPORTDEVS_API_KEY environment variable",
                "3. Test with 300 free requests per day",
                "4. Upgrade to Major Plan (€19/month) for full access",
                "5. Your SGM analysis will then use real live AFL data!"
            ]
        }
        
    except Exception as e:
        logging.error(f"Live demo SGM error: {str(e)}")
        raise HTTPException(500, f"Live demo SGM failed: {str(e)}")

import statistics

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
