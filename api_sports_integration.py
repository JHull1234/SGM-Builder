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
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.session = httpx.AsyncClient(timeout=30.0)
        
        if not self.api_key:
            logging.warning("APISPORTS_API_KEY not found in environment variables")
    
    async def test_api_connectivity(self) -> Dict:
        """Test API connectivity and find working endpoints"""
        results = {}
        
        for base_url in self.base_urls:
            try:
                # Test basic connectivity
                response = await self.session.get(
                    f"{base_url}/status",
                    headers=self.headers
                )
                results[base_url] = {
                    "status_code": response.status_code,
                    "response": response.text[:200] if response.text else "No response"
                }
            except Exception as e:
                results[base_url] = {"error": str(e)}
        
        # Try alternative endpoints
        alternative_endpoints = [
            "https://api.api-sports.io/aussie-rules",
            "https://api-sports.io/v1/aussie-rules", 
            "https://australian-football.api-sports.io/v1"
        ]
        
        for endpoint in alternative_endpoints:
            try:
                response = await self.session.get(
                    f"{endpoint}/seasons",
                    headers=self.headers
                )
                results[endpoint] = {
                    "status_code": response.status_code,
                    "response": response.text[:200] if response.text else "No response"
                }
            except Exception as e:
                results[endpoint] = {"error": str(e)}
        
        return results
    
    async def get_leagues_and_seasons(self) -> Dict:
        """Get available leagues and seasons"""
        # Try different possible endpoints for AFL data
        possible_endpoints = [
            "/leagues",
            "/seasons", 
            "/competitions",
            "/tournaments"
        ]
        
        for base_url in self.base_urls[:2]:  # Test top 2 URLs
            for endpoint in possible_endpoints:
                try:
                    response = await self.session.get(
                        f"{base_url}{endpoint}",
                        headers=self.headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "success": True,
                            "endpoint": f"{base_url}{endpoint}",
                            "data": data,
                            "data_source": "API Sports"
                        }
                        
                except Exception as e:
                    continue
        
        return {"error": "No working leagues/seasons endpoint found"}
    
    async def get_teams(self, season: int = 2025) -> Dict:
        """Get AFL teams for the season"""
        possible_endpoints = [
            f"/teams?season={season}",
            f"/teams?league=afl&season={season}",
            f"/teams/{season}",
            "/teams"
        ]
        
        for base_url in self.base_urls[:2]:
            for endpoint in possible_endpoints:
                try:
                    response = await self.session.get(
                        f"{base_url}{endpoint}",
                        headers=self.headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "success": True,
                            "endpoint": f"{base_url}{endpoint}",
                            "teams": data,
                            "data_source": "API Sports"
                        }
                        
                except Exception as e:
                    continue
        
        return {"error": "No working teams endpoint found"}
    
    async def get_players(self, team_id: Optional[int] = None, season: int = 2025) -> Dict:
        """Get AFL players"""
        possible_endpoints = []
        
        if team_id:
            possible_endpoints.extend([
                f"/players?team={team_id}&season={season}",
                f"/players?team={team_id}",
                f"/teams/{team_id}/players",
                f"/squads?team={team_id}&season={season}"
            ])
        else:
            possible_endpoints.extend([
                f"/players?season={season}",
                "/players",
                f"/squads?season={season}"
            ])
        
        for base_url in self.base_urls[:2]:
            for endpoint in possible_endpoints:
                try:
                    response = await self.session.get(
                        f"{base_url}{endpoint}",
                        headers=self.headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "success": True,
                            "endpoint": f"{base_url}{endpoint}",
                            "players": data,
                            "data_source": "API Sports"
                        }
                        
                except Exception as e:
                    continue
        
        return {"error": "No working players endpoint found"}
    
    async def search_player_by_name(self, player_name: str) -> Dict:
        """Search for a specific player by name"""
        possible_endpoints = [
            f"/players?search={player_name}",
            f"/players?name={player_name}",
            f"/search/players?q={player_name}"
        ]
        
        for base_url in self.base_urls[:2]:
            for endpoint in possible_endpoints:
                try:
                    response = await self.session.get(
                        f"{base_url}{endpoint}",
                        headers=self.headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "success": True,
                            "endpoint": f"{base_url}{endpoint}",
                            "search_results": data,
                            "player_name": player_name,
                            "data_source": "API Sports"
                        }
                        
                except Exception as e:
                    continue
        
        return {"error": f"No player data found for {player_name}"}
    
    async def get_player_statistics(self, player_id: int, season: int = 2025) -> Dict:
        """Get detailed player statistics"""
        possible_endpoints = [
            f"/players/{player_id}/statistics?season={season}",
            f"/players/{player_id}/stats?season={season}",
            f"/statistics/players/{player_id}?season={season}",
            f"/players/{player_id}?season={season}"
        ]
        
        for base_url in self.base_urls[:2]:
            for endpoint in possible_endpoints:
                try:
                    response = await self.session.get(
                        f"{base_url}{endpoint}",
                        headers=self.headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "success": True,
                            "endpoint": f"{base_url}{endpoint}",
                            "statistics": data,
                            "player_id": player_id,
                            "season": season,
                            "data_source": "API Sports"
                        }
                        
                except Exception as e:
                    continue
        
        return {"error": f"No statistics found for player ID {player_id}"}
    
    async def get_matches(self, season: int = 2025, team_id: Optional[int] = None) -> Dict:
        """Get AFL matches"""
        possible_endpoints = []
        
        if team_id:
            possible_endpoints.extend([
                f"/fixtures?team={team_id}&season={season}",
                f"/matches?team={team_id}&season={season}",
                f"/games?team={team_id}&season={season}"
            ])
        else:
            possible_endpoints.extend([
                f"/fixtures?season={season}",
                f"/matches?season={season}",
                f"/games?season={season}",
                "/fixtures/live",
                "/matches/live"
            ])
        
        for base_url in self.base_urls[:2]:
            for endpoint in possible_endpoints:
                try:
                    response = await self.session.get(
                        f"{base_url}{endpoint}",
                        headers=self.headers
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "success": True,
                            "endpoint": f"{base_url}{endpoint}",
                            "matches": data,
                            "season": season,
                            "data_source": "API Sports"
                        }
                        
                except Exception as e:
                    continue
        
        return {"error": "No working matches endpoint found"}
    
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
