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

# Initialize venues collection
@app.on_event("startup")
async def startup_event():
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
        """Get player statistics - simulated for now since Squiggle has limited player data"""
        # This would integrate with more detailed player APIs in production
        sample_players = [
            {
                "id": str(uuid.uuid4()),
                "name": "Clayton Oliver",
                "team": "Melbourne",
                "position": "Midfielder",
                "avg_disposals": 32.5,
                "avg_goals": 0.8,
                "avg_marks": 4.2,
                "avg_tackles": 6.8,
                "games_played": 20
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Christian Petracca", 
                "team": "Melbourne",
                "position": "Midfielder",
                "avg_disposals": 28.3,
                "avg_goals": 1.2,
                "avg_marks": 5.1,
                "avg_tackles": 4.9,
                "games_played": 18
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Marcus Bontempelli",
                "team": "Western Bulldogs", 
                "position": "Midfielder",
                "avg_disposals": 29.7,
                "avg_goals": 1.1,
                "avg_marks": 6.3,
                "avg_tackles": 5.2,
                "games_played": 22
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Jeremy Cameron",
                "team": "Geelong",
                "position": "Forward",
                "avg_disposals": 12.4,
                "avg_goals": 2.8,
                "avg_marks": 7.9,
                "avg_tackles": 2.1,
                "games_played": 21
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Tom Hawkins",
                "team": "Geelong", 
                "position": "Forward",
                "avg_disposals": 10.2,
                "avg_goals": 2.3,
                "avg_marks": 8.5,
                "avg_tackles": 1.8,
                "games_played": 19
            }
        ]
        return sample_players

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
        """Calculate correlation between multiple outcomes in SGM"""
        # Simplified correlation calculation
        # In production, this would use historical data and advanced statistics
        base_score = 0.7
        
        # Reduce correlation for outcomes from same player (less independent)
        same_player_outcomes = {}
        for outcome in outcomes:
            player = outcome.get('player')
            if player:
                same_player_outcomes[player] = same_player_outcomes.get(player, 0) + 1
        
        # Penalty for multiple outcomes from same player
        for player, count in same_player_outcomes.items():
            if count > 1:
                base_score -= (count - 1) * 0.1
        
        return max(0.1, min(1.0, base_score))
    
    @staticmethod
    def calculate_weather_impact(weather: Dict, outcomes: List[Dict]) -> Dict:
        """Calculate how weather impacts SGM outcomes"""
        impact = {
            "total_impact": 0.0,
            "wind_impact": 0.0,
            "rain_impact": 0.0,
            "temperature_impact": 0.0
        }
        
        # Wind impact on goals and marks
        if weather["wind_speed"] > 25:
            for outcome in outcomes:
                if "goals" in outcome.get("type", "").lower():
                    impact["wind_impact"] -= 0.15
                if "marks" in outcome.get("type", "").lower():
                    impact["wind_impact"] -= 0.1
        
        # Rain impact on disposals and ball handling
        if weather["precipitation"] > 1:
            for outcome in outcomes:
                if "disposals" in outcome.get("type", "").lower():
                    impact["rain_impact"] -= 0.1
        
        # Temperature impact on endurance-based stats
        if weather["temperature"] > 30 or weather["temperature"] < 10:
            for outcome in outcomes:
                if "disposals" in outcome.get("type", "").lower():
                    impact["temperature_impact"] -= 0.05
        
        impact["total_impact"] = sum([
            impact["wind_impact"],
            impact["rain_impact"], 
            impact["temperature_impact"]
        ])
        
        return impact
    
    @staticmethod
    def calculate_value_rating(predicted_prob: float, market_odds: float) -> float:
        """Calculate value rating for SGM bet"""
        if market_odds <= 0:
            return 0.0
        
        implied_prob = 1 / market_odds
        value = (predicted_prob - implied_prob) / implied_prob
        return max(-1.0, min(1.0, value))

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
    """Analyze Same Game Multi combination"""
    try:
        match_id = request.get("match_id")
        outcomes = request.get("outcomes", [])
        venue = request.get("venue")
        
        if not outcomes:
            raise HTTPException(400, "No outcomes provided")
        
        # Get weather data for venue
        weather_data = await WeatherService.get_venue_weather(venue)
        
        # Calculate correlation score
        correlation_score = SGMAnalytics.calculate_correlation_score(outcomes)
        
        # Calculate weather impact
        weather_impact = SGMAnalytics.calculate_weather_impact(weather_data, outcomes)
        
        # Calculate value rating (simplified - would use real market odds in production)
        mock_market_odds = 3.50  # Mock odds for demonstration
        predicted_probability = correlation_score * (1 + weather_impact["total_impact"])
        value_rating = SGMAnalytics.calculate_value_rating(predicted_probability, mock_market_odds)
        
        # Recommended stake (basic Kelly Criterion)
        recommended_stake = max(0, min(0.1, value_rating * 0.25)) if value_rating > 0 else 0
        
        result = {
            "match_id": match_id,
            "outcomes": outcomes,
            "analysis": {
                "correlation_score": round(correlation_score, 3),
                "weather_impact": weather_impact,
                "predicted_probability": round(predicted_probability, 3),
                "value_rating": round(value_rating, 3),
                "recommended_stake": round(recommended_stake, 3),
                "confidence": "High" if correlation_score > 0.6 else "Medium" if correlation_score > 0.4 else "Low"
            },
            "weather_conditions": weather_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store analysis in database
        analysis_collection = db["sgm_analyses"]
        await analysis_collection.insert_one(result)
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"SGM analysis error: {str(e)}")

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

@app.get("/api/venues")
async def get_all_venues():
    """Get all AFL venues"""
    return [{"name": name, **data} for name, data in AFL_VENUES.items()]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
