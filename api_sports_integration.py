# API Sports AFL Integration - Live 2025 Season Data
# Professional AFL statistics for real SGM analysis

import httpx
import asyncio
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import os

class APISportsAFLService:
    """Professional AFL data integration with API Sports"""
    
    def __init__(self):
        self.api_key = os.environ.get('APISPORTS_API_KEY')
        self.base_url = "https://api-sports.io"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'api-sports.io',
            'Content-Type': 'application/json'
        }
        self.session = httpx.AsyncClient(timeout=30.0)
        
        if not self.api_key:
            logging.warning("APISPORTS_API_KEY not found in environment variables")
    
    async def test_api_connectivity(self) -> Dict:
        """Test API connectivity with correct API-Sports.io AFL endpoints"""
        results = {}
        
        # Test main AFL endpoints from documentation
        endpoints_to_test = [
            "/timezone",
            "/seasons", 
            "/leagues",
            "/teams",
            "/players",
            "/games"
        ]
        
        for endpoint in endpoints_to_test:
            try:
                response = await self.session.get(
                    f"{self.base_url}{endpoint}",
                    headers=self.headers
                )
                results[endpoint] = {
                    "status_code": response.status_code,
                    "response": response.text[:300] if response.text else "No response",
                    "success": response.status_code == 200
                }
            except Exception as e:
                results[endpoint] = {"error": str(e)}
        
        return results
    
    async def get_leagues_and_seasons(self) -> Dict:
        """Get available AFL leagues and seasons"""
        try:
            # Get seasons first
            seasons_response = await self.session.get(
                f"{self.base_url}/seasons",
                headers=self.headers
            )
            
            # Get leagues
            leagues_response = await self.session.get(
                f"{self.base_url}/leagues",
                headers=self.headers
            )
            
            return {
                "success": True,
                "seasons": seasons_response.json() if seasons_response.status_code == 200 else None,
                "leagues": leagues_response.json() if leagues_response.status_code == 200 else None,
                "seasons_status": seasons_response.status_code,
                "leagues_status": leagues_response.status_code,
                "data_source": "API-Sports.io AFL"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def get_teams(self, season: int = 2025) -> Dict:
        """Get AFL teams for the season"""
        try:
            response = await self.session.get(
                f"{self.base_url}/teams",
                headers=self.headers,
                params={"season": season}
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "teams": data,
                    "season": season,
                    "data_source": "API-Sports.io AFL"
                }
            else:
                return {
                    "error": f"API returned status {response.status_code}",
                    "response": response.text[:200]
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    async def get_players(self, team_id: Optional[int] = None, season: int = 2025) -> Dict:
        """Get AFL players"""
        try:
            params = {"season": season}
            if team_id:
                params["team"] = team_id
            
            response = await self.session.get(
                f"{self.base_url}/players",
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "players": data,
                    "season": season,
                    "team_id": team_id,
                    "data_source": "API-Sports.io AFL"
                }
            else:
                return {
                    "error": f"API returned status {response.status_code}",
                    "response": response.text[:200]
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    async def search_player_by_name(self, player_name: str) -> Dict:
        """Search for a specific player by name"""
        try:
            response = await self.session.get(
                f"{self.base_url}/players",
                headers=self.headers,
                params={"search": player_name}
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "search_results": data,
                    "player_name": player_name,
                    "data_source": "API-Sports.io AFL"
                }
            else:
                return {
                    "error": f"API returned status {response.status_code}",
                    "response": response.text[:200]
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    async def get_matches(self, season: int = 2025, team_id: Optional[int] = None) -> Dict:
        """Get AFL matches"""
        try:
            params = {"season": season}
            if team_id:
                params["team"] = team_id
            
            response = await self.session.get(
                f"{self.base_url}/games",
                headers=self.headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "matches": data,
                    "season": season,
                    "data_source": "API-Sports.io AFL"
                }
            else:
                return {
                    "error": f"API returned status {response.status_code}",
                    "response": response.text[:200]
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    async def comprehensive_afl_test(self) -> Dict:
        """Comprehensive test of all AFL endpoints"""
        test_results = {
            "api_key_status": "Connected" if self.api_key else "Missing",
            "connectivity_tests": {},
            "endpoint_tests": {}
        }
        
        # Test basic connectivity
        test_results["connectivity_tests"] = await self.test_api_connectivity()
        
        # Test specific AFL endpoints
        test_results["endpoint_tests"]["leagues"] = await self.get_leagues_and_seasons()
        test_results["endpoint_tests"]["teams"] = await self.get_teams()
        test_results["endpoint_tests"]["players"] = await self.get_players()
        test_results["endpoint_tests"]["matches"] = await self.get_matches()
        
        # Test player search
        test_results["endpoint_tests"]["clayton_oliver_search"] = await self.search_player_by_name("Clayton Oliver")
        test_results["endpoint_tests"]["nick_daicos_search"] = await self.search_player_by_name("Nick Daicos")
        
        return test_results
    
    async def close(self):
        """Close the HTTP session"""
        await self.session.aclose()

class APISportsAFLAnalyzer:
    """SGM Analysis using API Sports AFL data"""
    
    def __init__(self):
        self.api_service = APISportsAFLService()
    
    async def analyze_player_sgm(self, player_name: str, stat_type: str, threshold: float) -> Dict:
        """Analyze individual player performance for SGM"""
        try:
            # Search for player
            player_search = await self.api_service.search_player_by_name(player_name)
            
            if "error" not in player_search and player_search.get("success"):
                search_results = player_search.get("search_results", {})
                
                # Try to extract player ID and get statistics
                if isinstance(search_results, dict) and "response" in search_results:
                    players = search_results["response"]
                    if players and len(players) > 0:
                        player_id = players[0].get("id")
                        if player_id:
                            stats = await self.api_service.get_player_statistics(player_id)
                            
                            return {
                                "player_name": player_name,
                                "player_id": player_id,
                                "player_search": player_search,
                                "statistics": stats,
                                "analysis_timestamp": datetime.now().isoformat()
                            }
            
            return {
                "player_name": player_name,
                "status": "Player found but limited statistics access",
                "search_result": player_search,
                "note": "May require different API subscription or endpoint"
            }
            
        except Exception as e:
            return {"error": str(e), "player_name": player_name}
    
    async def close(self):
        """Close API service"""
        await self.api_service.close()
