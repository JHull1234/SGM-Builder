from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
import httpx
import asyncio

# Add the project root to Python path to import our modules
sys.path.append('/app')

# Import our advanced modules
try:
    from ml_sgm_picker import MachineLearningPredictor, AutomatedSGMPicker, AdvancedStatistics
    from advanced_analytics import RecentFormAnalyzer, TeammateSymergyAnalyzer, DefensiveMatchupAnalyzer, InjuryImpactAnalyzer
    from sportdevs_integration import SportDevsAFLAPI, AFLDataProcessor
    ML_MODULES_AVAILABLE = True
    print("✅ Advanced ML modules loaded successfully")
except ImportError as e:
    ML_MODULES_AVAILABLE = False
    print(f"⚠️ ML modules not available: {e}")

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Advanced AFL SGM Builder API", version="2.0.0")

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
class AdvancedSGMRequest(BaseModel):
    match_id: str
    target_odds: Optional[float] = 3.0
    max_players: Optional[int] = 4
    confidence_threshold: Optional[float] = 0.7
    use_ml_models: Optional[bool] = True
    include_weather: Optional[bool] = True

class PlayerPredictionRequest(BaseModel):
    player_id: str
    match_context: Dict
    stat_types: List[str] = ["disposals", "goals", "marks", "tackles"]

# Initialize services
class WeatherService:
    def __init__(self):
        self.api_key = os.environ.get('WEATHERAPI_KEY')
        self.base_url = "http://api.weatherapi.com/v1"

    async def get_weather_for_venue(self, venue: str, date: str = None) -> Dict:
        if venue not in AFL_VENUES:
            return {"error": f"Venue {venue} not found"}
        
        coords = AFL_VENUES[venue]
        
        async with httpx.AsyncClient() as client:
            try:
                if date:
                    url = f"{self.base_url}/forecast.json"
                    params = {"key": self.api_key, "q": f"{coords['lat']},{coords['lon']}", "dt": date}
                else:
                    url = f"{self.base_url}/current.json" 
                    params = {"key": self.api_key, "q": f"{coords['lat']},{coords['lon']}"}
                
                response = await client.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if date:
                    forecast_day = data['forecast']['forecastday'][0]['day']
                    return {
                        "venue": venue, "date": date,
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

# Initialize services
weather_service = WeatherService()
odds_service = OddsService()

if ML_MODULES_AVAILABLE:
    # Initialize SportDevs API and advanced modules
    sportdevs_api = SportDevsAFLAPI()
    afl_data_processor = AFLDataProcessor(sportdevs_api)
    ml_predictor = MachineLearningPredictor()
    automated_sgm_picker = AutomatedSGMPicker(ml_predictor)

# API Endpoints
@api_router.get("/")
async def root():
    return {
        "message": "Advanced AFL SGM Builder API v2.0",
        "features": {
            "sportdevs_integration": ML_MODULES_AVAILABLE,
            "ml_models": ML_MODULES_AVAILABLE,
            "weather_data": True,
            "betting_odds": True,
            "advanced_analytics": ML_MODULES_AVAILABLE
        }
    }

@api_router.get("/teams")
async def get_teams():
    """Get all AFL teams from SportDevs"""
    if not ML_MODULES_AVAILABLE:
        return {"teams": AFL_TEAMS}
    
    try:
        teams = await sportdevs_api.get_teams()
        
        # Store in database
        for team in teams:
            team_doc = {
                "team_id": team.get("id"),
                "name": team.get("name"),
                "abbreviation": team.get("abbreviation"),
                "city": team.get("city"),
                "founded": team.get("founded"),
                "last_updated": datetime.utcnow()
            }
            await db.teams.update_one(
                {"team_id": team_doc["team_id"]},
                {"$set": team_doc},
                upsert=True
            )
        
        return {"teams": teams, "count": len(teams)}
    except Exception as e:
        logging.error(f"Teams API error: {str(e)}")
        return {"error": str(e), "fallback_teams": AFL_TEAMS}

@api_router.get("/players")
async def get_players(team_id: Optional[str] = None):
    """Get AFL players from SportDevs"""
    if not ML_MODULES_AVAILABLE:
        return {"error": "SportDevs integration not available"}
    
    try:
        players = await sportdevs_api.get_players(team_id)
        
        # Store key players in database
        for player in players[:50]:  # Limit to avoid timeout
            player_doc = {
                "player_id": player.get("id"),
                "name": player.get("name"),
                "team_id": player.get("team_id"),
                "position": player.get("position"),
                "age": player.get("age"),
                "last_updated": datetime.utcnow()
            }
            await db.players.update_one(
                {"player_id": player_doc["player_id"]},
                {"$set": player_doc},
                upsert=True
            )
        
        return {"players": players, "count": len(players)}
    except Exception as e:
        logging.error(f"Players API error: {str(e)}")
        return {"error": str(e)}

@api_router.get("/player/{player_id}/enhanced")
async def get_enhanced_player_data(player_id: str):
    """Get comprehensive enhanced player data"""
    if not ML_MODULES_AVAILABLE:
        return {"error": "Advanced analytics not available"}
    
    try:
        enhanced_data = await afl_data_processor.get_enhanced_player_data(player_id)
        
        # Store in database
        await db.enhanced_players.update_one(
            {"player_id": player_id},
            {"$set": enhanced_data},
            upsert=True
        )
        
        return enhanced_data
    except Exception as e:
        logging.error(f"Enhanced player data error: {str(e)}")
        return {"error": str(e)}

@api_router.post("/predict/player")
async def predict_player_performance(request: PlayerPredictionRequest):
    """Advanced ML player performance prediction"""
    if not ML_MODULES_AVAILABLE:
        return {"error": "ML models not available"}
    
    try:
        # Get enhanced player data
        enhanced_data = await afl_data_processor.get_enhanced_player_data(request.player_id)
        
        if not enhanced_data:
            raise HTTPException(404, "Player data not found")
        
        # Get ML prediction
        ml_prediction = ml_predictor.predict_performance(enhanced_data, request.match_context)
        
        # Get recent form analysis
        form_analysis = {}
        for stat_type in request.stat_types:
            form_analysis[stat_type] = RecentFormAnalyzer.calculate_form_factor(
                enhanced_data.get("name", "Unknown"), stat_type
            )
        
        # Get injury impact
        injury_impact = InjuryImpactAnalyzer.get_injury_impact(enhanced_data.get("name", "Unknown"))
        
        combined_prediction = {
            "player_id": request.player_id,
            "player_name": enhanced_data.get("name", "Unknown"),
            "ml_predictions": ml_prediction,
            "form_analysis": form_analysis,
            "injury_impact": injury_impact,
            "enhanced_data": enhanced_data,
            "confidence_rating": "High" if ml_prediction.get("confidence", {}).get("disposals", 0) > 0.8 else "Medium",
            "prediction_timestamp": datetime.now().isoformat()
        }
        
        # Store prediction
        await db.player_predictions.insert_one(combined_prediction)
        
        return combined_prediction
        
    except Exception as e:
        logging.error(f"Player prediction error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@api_router.post("/sgm/advanced")
async def advanced_sgm_analysis(request: AdvancedSGMRequest):
    """Advanced SGM analysis with ML models and SportDevs data"""
    if not ML_MODULES_AVAILABLE:
        return {"error": "Advanced SGM analysis not available"}
    
    try:
        # Get match context from SportDevs
        match_context = await afl_data_processor.get_match_context(request.match_id)
        
        if not match_context:
            raise HTTPException(404, "Match data not found")
        
        # Get weather data if requested
        if request.include_weather:
            venue = match_context.get("venue", "MCG")
            date = match_context.get("date")
            weather_data = await weather_service.get_weather_for_venue(venue, date)
            match_context["weather"] = weather_data
        
        # Get available players for both teams
        home_team_id = match_context.get("home_team_defense", {}).get("team_id")
        away_team_id = match_context.get("away_team_defense", {}).get("team_id")
        
        available_players = []
        if home_team_id:
            home_players = await sportdevs_api.get_players(home_team_id)
            available_players.extend(home_players[:15])  # Top 15 players per team
        if away_team_id:
            away_players = await sportdevs_api.get_players(away_team_id)
            available_players.extend(away_players[:15])
        
        if not available_players:
            raise HTTPException(404, "No player data available for teams")
        
        # Process players for ML analysis
        processed_players = []
        for player in available_players:
            enhanced_data = await afl_data_processor.get_enhanced_player_data(player.get("id"))
            if enhanced_data:
                processed_players.append(enhanced_data)
        
        # Use automated SGM picker
        sgm_recommendations = await automated_sgm_picker.recommend_sgm(
            target_odds=request.target_odds,
            match_context=match_context,
            available_players=processed_players
        )
        
        # Enhance with additional analytics
        for recommendation in sgm_recommendations.get("recommendations", []):
            # Add teammate synergy analysis
            if len(recommendation.get("sgm_outcomes", [])) > 1:
                synergy_analysis = TeammateSymergyAnalyzer.calculate_synergy_impact(
                    recommendation["sgm_outcomes"]
                )
                recommendation["synergy_analysis"] = synergy_analysis
            
            # Add defensive matchup analysis for each player
            matchup_analyses = []
            for outcome in recommendation.get("sgm_outcomes", []):
                player_name = outcome.get("player")
                opponent_team = match_context.get("away_team") if player_name in [p.get("name") for p in processed_players if p.get("team") == match_context.get("home_team")] else match_context.get("home_team")
                
                if player_name and opponent_team:
                    # Find player data
                    player_data = next((p for p in processed_players if p.get("name") == player_name), None)
                    if player_data:
                        matchup_analysis = DefensiveMatchupAnalyzer.get_detailed_matchup_analysis(
                            player_data, opponent_team, match_context.get("venue", "MCG")
                        )
                        matchup_analyses.append({
                            "player": player_name,
                            "matchup": matchup_analysis
                        })
            
            recommendation["matchup_analyses"] = matchup_analyses
        
        # Store analysis
        analysis_doc = {
            "match_id": request.match_id,
            "target_odds": request.target_odds,
            "match_context": match_context,
            "sgm_recommendations": sgm_recommendations,
            "analysis_timestamp": datetime.now().isoformat(),
            "confidence_threshold": request.confidence_threshold
        }
        await db.advanced_sgm_analysis.insert_one(analysis_doc)
        
        return {
            "match_context": match_context,
            "sgm_recommendations": sgm_recommendations,
            "analysis_summary": {
                "total_recommendations": len(sgm_recommendations.get("recommendations", [])),
                "high_confidence_picks": len([r for r in sgm_recommendations.get("recommendations", []) if r.get("confidence_score", 0) > request.confidence_threshold]),
                "weather_impact": "Included" if request.include_weather else "Not included",
                "ml_models_used": request.use_ml_models
            }
        }
        
    except Exception as e:
        logging.error(f"Advanced SGM analysis error: {str(e)}")
        raise HTTPException(500, f"SGM analysis failed: {str(e)}")

@api_router.get("/fixtures")
async def get_fixtures(team_id: Optional[str] = None):
    """Get AFL fixtures from SportDevs"""
    if not ML_MODULES_AVAILABLE:
        return {"error": "SportDevs integration not available"}
    
    try:
        fixtures = await sportdevs_api.get_fixtures(team_id=team_id)
        
        # Store fixtures
        for fixture in fixtures:
            fixture_doc = {
                "match_id": fixture.get("id"),
                "home_team": fixture.get("home_team"),
                "away_team": fixture.get("away_team"),
                "venue": fixture.get("venue"),
                "date": fixture.get("date"),
                "round": fixture.get("round"),
                "season": fixture.get("season", "2025"),
                "last_updated": datetime.utcnow()
            }
            await db.fixtures.update_one(
                {"match_id": fixture_doc["match_id"]},
                {"$set": fixture_doc},
                upsert=True
            )
        
        return {"fixtures": fixtures, "count": len(fixtures)}
    except Exception as e:
        logging.error(f"Fixtures API error: {str(e)}")
        return {"error": str(e)}

@api_router.get("/injuries")
async def get_current_injuries(team_id: Optional[str] = None):
    """Get current AFL injury list"""
    if not ML_MODULES_AVAILABLE:
        return {"error": "SportDevs integration not available"}
    
    try:
        injuries = await sportdevs_api.get_injuries(team_id)
        
        # Store injury data
        for injury in injuries:
            injury_doc = {
                "player_id": injury.get("player_id"),
                "player_name": injury.get("player_name"),
                "team_id": injury.get("team_id"),
                "injury_type": injury.get("injury_type"),
                "status": injury.get("status"),
                "expected_return": injury.get("expected_return"),
                "last_updated": datetime.utcnow()
            }
            await db.injuries.update_one(
                {"player_id": injury_doc["player_id"]},
                {"$set": injury_doc},
                upsert=True
            )
        
        return {"injuries": injuries, "count": len(injuries)}
    except Exception as e:
        logging.error(f"Injuries API error: {str(e)}")
        return {"error": str(e)}

@api_router.get("/weather/{venue}")
async def get_venue_weather(venue: str, date: str = None):
    """Get weather for AFL venue"""
    weather_data = await weather_service.get_weather_for_venue(venue, date)
    
    # Store weather data
    weather_doc = {
        "venue": venue,
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
