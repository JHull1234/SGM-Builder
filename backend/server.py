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
