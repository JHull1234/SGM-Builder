"""
SportDevs AFL API Integration
Professional grade AFL data integration for SGM betting analysis
"""

import os
import asyncio
import httpx
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

class SportDevsAFLAPI:
    """Professional AFL data provider via SportDevs API"""
    
    def __init__(self):
        self.api_key = os.environ.get('SPORTDEVS_API_KEY')
        self.base_url = "https://aussie-rules.sportdevs.com"  # Correct SportDevs URL
        self.headers = {
            'Accept': 'application/json',
            'User-Agent': 'AFL-SGM-Builder/1.0'
        }
        # Note: SportDevs may not require Bearer token for public endpoints
        
    async def get_teams(self) -> List[Dict]:
        """Get all AFL teams data from SportDevs"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/teams",
                    headers=self.headers,
                    timeout=15
                )
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, list) else []
            except Exception as e:
                logging.error(f"SportDevs teams API error: {str(e)}")
                return []
    
    async def get_team_statistics(self, team_id: str, season: str = "2025") -> Dict:
        """Get detailed team statistics"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/aussie-rules/teams/{team_id}/statistics",
                    headers=self.headers,
                    params={"season": season},
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"SportDevs team stats error: {str(e)}")
                return {}
    
    async def get_players(self, team_id: Optional[str] = None) -> List[Dict]:
        """Get player data, optionally filtered by team"""
        async with httpx.AsyncClient() as client:
            try:
                params = {}
                if team_id:
                    params['team_id'] = team_id
                    
                response = await client.get(
                    f"{self.base_url}/aussie-rules/players",
                    headers=self.headers,
                    params=params,
                    timeout=15
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"SportDevs players API error: {str(e)}")
                return []
    
    async def get_player_statistics(self, player_id: str, season: str = "2025") -> Dict:
        """Get comprehensive player statistics"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/aussie-rules/players/{player_id}/statistics",
                    headers=self.headers,
                    params={"season": season},
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"SportDevs player stats error: {str(e)}")
                return {}
    
    async def get_fixtures(self, season: str = "2025", team_id: Optional[str] = None) -> List[Dict]:
        """Get match fixtures from Squiggle API (more reliable for AFL)"""
        async with httpx.AsyncClient() as client:
            try:
                headers = {'User-Agent': 'AFL-SGM-Builder/1.0'}
                url = f"https://api.squiggle.com.au/?q=games;year={season}"
                if team_id:
                    url += f";team={team_id}"
                    
                response = await client.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()
                return data.get('games', [])
            except Exception as e:
                logging.error(f"Fixtures API error: {str(e)}")
                return []
    
    async def get_current_round_fixtures(self) -> List[Dict]:
        """Get current round fixtures with live data"""
        async with httpx.AsyncClient() as client:
            try:
                headers = {'User-Agent': 'AFL-SGM-Builder/1.0'}
                # Get current round matches
                response = await client.get(
                    "https://api.squiggle.com.au/?q=games;year=2025;round=latest",
                    headers=headers,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                return data.get('games', [])
            except Exception as e:
                logging.error(f"Current round fixtures error: {str(e)}")
                return []
    
    async def get_live_standings(self) -> List[Dict]:
        """Get current 2025 AFL standings"""
        async with httpx.AsyncClient() as client:
            try:
                headers = {'User-Agent': 'AFL-SGM-Builder/1.0'}
                response = await client.get(
                    "https://api.squiggle.com.au/?q=standings;year=2025",
                    headers=headers,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                return data.get('standings', [])
            except Exception as e:
                logging.error(f"Live standings error: {str(e)}")
                return []
    
    async def get_match_statistics(self, match_id: str) -> Dict:
        """Get detailed match statistics"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/aussie-rules/matches/{match_id}/statistics",
                    headers=self.headers,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"SportDevs match stats error: {str(e)}")
                return {}
    
    async def get_player_recent_form(self, player_id: str, games_count: int = 5) -> List[Dict]:
        """Get player's recent game statistics"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/aussie-rules/players/{player_id}/recent-matches",
                    headers=self.headers,
                    params={"limit": games_count},
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"SportDevs recent form error: {str(e)}")
                return []
    
    async def get_head_to_head(self, team1_id: str, team2_id: str, limit: int = 10) -> List[Dict]:
        """Get head-to-head match history between teams"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/aussie-rules/head-to-head",
                    headers=self.headers,
                    params={
                        "team1_id": team1_id,
                        "team2_id": team2_id,
                        "limit": limit
                    },
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"SportDevs H2H error: {str(e)}")
                return []
    
    async def get_injuries(self, team_id: Optional[str] = None) -> List[Dict]:
        """Get current injury list"""
        async with httpx.AsyncClient() as client:
            try:
                params = {}
                if team_id:
                    params['team_id'] = team_id
                    
                response = await client.get(
                    f"{self.base_url}/aussie-rules/injuries",
                    headers=self.headers,
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"SportDevs injuries error: {str(e)}")
                return []
    
    async def get_standings(self, season: str = "2025") -> List[Dict]:
        """Get current league standings"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/aussie-rules/standings",
                    headers=self.headers,
                    params={"season": season},
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logging.error(f"SportDevs standings error: {str(e)}")
                return []

class AFLDataProcessor:
    """Process and normalize AFL data from SportDevs for SGM analysis"""
    
    def __init__(self, sportdevs_api: SportDevsAFLAPI):
        self.api = sportdevs_api
    
    async def get_enhanced_player_data(self, player_id: str) -> Dict:
        """Get comprehensive enhanced player data for ML models"""
        try:
            # Get basic player stats
            player_stats = await self.api.get_player_statistics(player_id)
            
            # Get recent form
            recent_form = await self.api.get_player_recent_form(player_id)
            
            # Process and enhance data
            enhanced_data = {
                "player_id": player_id,
                "name": player_stats.get("name", "Unknown"),
                "team": player_stats.get("team", "Unknown"),
                "position": player_stats.get("position", "Unknown"),
                
                # Season averages
                "avg_disposals": player_stats.get("avg_disposals", 0),
                "avg_goals": player_stats.get("avg_goals", 0),
                "avg_marks": player_stats.get("avg_marks", 0),
                "avg_tackles": player_stats.get("avg_tackles", 0),
                "avg_kicks": player_stats.get("avg_kicks", 0),
                "avg_handballs": player_stats.get("avg_handballs", 0),
                "games_played": player_stats.get("games_played", 0),
                
                # Recent form analysis
                "recent_form": self._process_recent_form(recent_form),
                
                # Performance consistency
                "consistency_rating": self._calculate_consistency(recent_form),
                
                # Venue performance (if available)
                "venue_performance": self._extract_venue_performance(player_stats),
                
                # Last updated
                "last_updated": datetime.now().isoformat()
            }
            
            return enhanced_data
            
        except Exception as e:
            logging.error(f"Enhanced player data error: {str(e)}")
            return {}
    
    def _process_recent_form(self, recent_games: List[Dict]) -> Dict:
        """Process recent form data"""
        if not recent_games:
            return {"last_5_games": [], "trend": "Unknown"}
        
        # Extract key stats from recent games
        processed_games = []
        for game in recent_games[-5:]:  # Last 5 games
            processed_games.append({
                "disposals": game.get("disposals", 0),
                "goals": game.get("goals", 0),
                "marks": game.get("marks", 0),
                "tackles": game.get("tackles", 0),
                "date": game.get("date", ""),
                "opponent": game.get("opponent", ""),
                "venue": game.get("venue", "")
            })
        
        # Calculate trend
        if len(processed_games) >= 3:
            recent_avg = sum(g["disposals"] for g in processed_games[-3:]) / 3
            earlier_avg = sum(g["disposals"] for g in processed_games[:-3]) / len(processed_games[:-3]) if len(processed_games) > 3 else recent_avg
            
            if recent_avg > earlier_avg * 1.1:
                trend = "Hot"
            elif recent_avg < earlier_avg * 0.9:
                trend = "Cold" 
            else:
                trend = "Average"
        else:
            trend = "Unknown"
        
        return {
            "last_5_games": processed_games,
            "trend": trend,
            "recent_average_disposals": sum(g["disposals"] for g in processed_games) / len(processed_games) if processed_games else 0
        }
    
    def _calculate_consistency(self, recent_games: List[Dict]) -> float:
        """Calculate player consistency rating"""
        if len(recent_games) < 3:
            return 0.5
        
        disposals = [game.get("disposals", 0) for game in recent_games]
        
        if not disposals or sum(disposals) == 0:
            return 0.5
        
        avg = sum(disposals) / len(disposals)
        variance = sum((x - avg) ** 2 for x in disposals) / len(disposals)
        std_dev = variance ** 0.5
        
        # Consistency score (lower std dev = higher consistency)
        consistency = max(0, 1 - (std_dev / avg)) if avg > 0 else 0.5
        
        return round(consistency, 3)
    
    def _extract_venue_performance(self, player_stats: Dict) -> Dict:
        """Extract venue-specific performance if available"""
        venue_data = player_stats.get("venue_breakdown", {})
        
        if not venue_data:
            return {}
        
        processed_venues = {}
        for venue, stats in venue_data.items():
            processed_venues[venue] = {
                "disposals": stats.get("avg_disposals", 0),
                "goals": stats.get("avg_goals", 0),
                "games": stats.get("games_played", 0)
            }
        
        return processed_venues
    
    async def get_team_defensive_stats(self, team_id: str) -> Dict:
        """Get team's defensive statistics for opposition analysis"""
        try:
            team_stats = await self.api.get_team_statistics(team_id)
            
            return {
                "team_id": team_id,
                "team_name": team_stats.get("name", "Unknown"),
                "midfielder_disposals_allowed": team_stats.get("avg_opposition_disposals", 125),
                "forward_goals_allowed": team_stats.get("avg_goals_conceded", 8.5),
                "tackles_per_game": team_stats.get("avg_tackles", 65),
                "pressure_rating": team_stats.get("pressure_factor", 1.0),
                "defensive_efficiency": team_stats.get("defensive_efficiency", 0.5),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Team defensive stats error: {str(e)}")
            return {}
    
    async def get_match_context(self, match_id: str) -> Dict:
        """Get comprehensive match context for predictions"""
        try:
            match_stats = await self.api.get_match_statistics(match_id)
            
            # Extract teams
            home_team_id = match_stats.get("home_team_id")
            away_team_id = match_stats.get("away_team_id")
            
            # Get team defensive stats
            home_defense = await self.get_team_defensive_stats(home_team_id) if home_team_id else {}
            away_defense = await self.get_team_defensive_stats(away_team_id) if away_team_id else {}
            
            # Get head-to-head
            h2h_data = await self.api.get_head_to_head(home_team_id, away_team_id) if home_team_id and away_team_id else []
            
            return {
                "match_id": match_id,
                "home_team": match_stats.get("home_team", "Unknown"),
                "away_team": match_stats.get("away_team", "Unknown"),
                "venue": match_stats.get("venue", "Unknown"),
                "date": match_stats.get("date", ""),
                "round": match_stats.get("round", 0),
                "home_team_defense": home_defense,
                "away_team_defense": away_defense,
                "head_to_head": h2h_data[:5],  # Last 5 H2H matches
                "match_importance": self._assess_match_importance(match_stats),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Match context error: {str(e)}")
            return {}
    
    def _assess_match_importance(self, match_stats: Dict) -> str:
        """Assess the importance of the match"""
        round_num = match_stats.get("round", 0)
        
        if round_num >= 22:  # Finals
            return "Finals"
        elif round_num >= 18:  # Late season
            return "High"
        elif round_num <= 3:  # Early season
            return "Medium"
        else:
            return "Standard"
