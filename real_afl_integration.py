"""
REAL-TIME AFL Data Integration
NO FAKE DATA - Only live, current AFL statistics from real sources
"""

import aiohttp
import asyncio
import json
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

class RealAFLDataIntegrator:
    def __init__(self):
        self.sources = {
            "footywire": "https://www.footywire.com",
            "afl_tables": "https://afltables.com/afl/stats/2025.html",
            "afl_official": "https://www.afl.com.au"
        }
        
    async def fetch_real_player_stats(self) -> dict:
        """Fetch REAL current AFL player statistics"""
        
        async with aiohttp.ClientSession() as session:
            try:
                # Try Footywire first - they have current stats
                url = "https://www.footywire.com/afl/footy/ft_player_rankings?type=DT&year=2025"
                
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Parse real player statistics from Footywire
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find the player statistics table
                        players_data = []
                        
                        # Look for table with player stats
                        tables = soup.find_all('table')
                        
                        for table in tables:
                            rows = table.find_all('tr')
                            
                            for row in rows[1:]:  # Skip header
                                cells = row.find_all('td')
                                
                                if len(cells) >= 6:  # Ensure enough columns
                                    try:
                                        player_name = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                                        team = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                                        games = cells[3].get_text(strip=True) if len(cells) > 3 else "0"
                                        avg_points = cells[4].get_text(strip=True) if len(cells) > 4 else "0"
                                        
                                        if player_name and team:
                                            players_data.append({
                                                "name": player_name,
                                                "team": team,
                                                "games_played": int(games) if games.isdigit() else 0,
                                                "avg_fantasy_points": float(avg_points) if avg_points.replace('.', '').isdigit() else 0.0
                                            })
                                            
                                    except (ValueError, IndexError):
                                        continue
                        
                        return {
                            "status": "success",
                            "source": "Real Footywire data",
                            "players": players_data[:50],  # Top 50 players
                            "timestamp": datetime.now().isoformat()
                        }
                        
            except Exception as e:
                print(f"Footywire fetch error: {str(e)}")
                
            # Fallback to AFL Tables if Footywire fails
            try:
                url = "https://afltables.com/afl/stats/2025.html"
                
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Parse AFL Tables data
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        players_data = []
                        
                        # Look for current season statistics
                        tables = soup.find_all('table')
                        
                        for table in tables:
                            rows = table.find_all('tr')
                            
                            for row in rows:
                                cells = row.find_all('td')
                                
                                if len(cells) >= 4:
                                    try:
                                        player_text = cells[0].get_text(strip=True)
                                        
                                        # Extract player name and team from AFL Tables format
                                        if '(' in player_text and ')' in player_text:
                                            name_part = player_text.split('(')[0].strip()
                                            team_part = player_text.split('(')[1].split(')')[0].strip()
                                            
                                            players_data.append({
                                                "name": name_part,
                                                "team": team_part,
                                                "source": "AFL Tables"
                                            })
                                            
                                    except (ValueError, IndexError):
                                        continue
                        
                        return {
                            "status": "success", 
                            "source": "Real AFL Tables data",
                            "players": players_data[:30],
                            "timestamp": datetime.now().isoformat()
                        }
                        
            except Exception as e:
                print(f"AFL Tables fetch error: {str(e)}")
                
        return {
            "status": "error",
            "message": "Could not fetch real AFL data from any source"
        }
    
    async def fetch_specific_player_stats(self, player_name: str) -> dict:
        """Fetch detailed stats for a specific player from real sources"""
        
        async with aiohttp.ClientSession() as session:
            try:
                # Search for player on Footywire
                search_url = f"https://www.footywire.com/afl/footy/ft_player_search?searchtext={player_name.replace(' ', '+')}"
                
                async with session.get(search_url, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find player profile link
                        links = soup.find_all('a', href=True)
                        
                        for link in links:
                            if 'ft_player_profile' in link['href'] and player_name.lower() in link.get_text().lower():
                                
                                # Get player's detailed page
                                player_url = f"https://www.footywire.com{link['href']}"
                                
                                async with session.get(player_url, timeout=30) as player_response:
                                    if player_response.status == 200:
                                        player_html = await player_response.text()
                                        player_soup = BeautifulSoup(player_html, 'html.parser')
                                        
                                        # Extract current season statistics
                                        stats = self._parse_player_stats_from_page(player_soup)
                                        
                                        return {
                                            "player_name": player_name,
                                            "source": "Real Footywire profile",
                                            "stats": stats,
                                            "url": player_url,
                                            "timestamp": datetime.now().isoformat()
                                        }
                                        
            except Exception as e:
                print(f"Player search error: {str(e)}")
                
        return {
            "error": f"Could not find real statistics for {player_name}"
        }
    
    def _parse_player_stats_from_page(self, soup) -> dict:
        """Parse real player statistics from their profile page"""
        
        stats = {
            "games_played": 0,
            "avg_disposals": 0.0,
            "avg_goals": 0.0,
            "avg_marks": 0.0,
            "avg_tackles": 0.0,
            "total_disposals": 0,
            "total_goals": 0
        }
        
        try:
            # Look for statistics tables
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                # Look for 2025 season row
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    
                    if len(cells) >= 6:
                        year_cell = cells[0].get_text(strip=True)
                        
                        if '2025' in year_cell:
                            try:
                                stats["games_played"] = int(cells[1].get_text(strip=True)) if len(cells) > 1 else 0
                                
                                # Parse other stats if available
                                for i, stat in enumerate(['disposals', 'goals', 'marks', 'tackles']):
                                    if len(cells) > i + 4:
                                        value = cells[i + 4].get_text(strip=True)
                                        stats[f"total_{stat}"] = int(float(value)) if value.replace('.', '').isdigit() else 0
                                        
                                        # Calculate average
                                        if stats["games_played"] > 0:
                                            stats[f"avg_{stat}"] = stats[f"total_{stat}"] / stats["games_played"]
                                            
                            except (ValueError, IndexError):
                                continue
                                
        except Exception as e:
            print(f"Stats parsing error: {str(e)}")
            
        return stats
    
    async def get_current_round_matches(self) -> dict:
        """Get real current round AFL matches"""
        
        async with aiohttp.ClientSession() as session:
            try:
                # Try AFL.com.au fixtures
                url = "https://www.afl.com.au/fixture"
                
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        matches = []
                        
                        # Look for fixture information
                        match_divs = soup.find_all('div', class_=re.compile('match|fixture|game'))
                        
                        for div in match_divs[:10]:  # Limit to recent matches
                            match_text = div.get_text(strip=True)
                            
                            if 'vs' in match_text.lower() or 'v' in match_text:
                                matches.append({
                                    "match_info": match_text,
                                    "source": "Real AFL.com.au"
                                })
                        
                        return {
                            "status": "success",
                            "source": "Real AFL.com.au fixtures",
                            "matches": matches,
                            "timestamp": datetime.now().isoformat()
                        }
                        
            except Exception as e:
                print(f"Fixtures fetch error: {str(e)}")
                
        return {
            "status": "error",
            "message": "Could not fetch real fixture data"
        }

# Create the real data integrator
real_afl_data = RealAFLDataIntegrator()

if __name__ == "__main__":
    # Test real data fetching
    async def test_real_data():
        print("🔄 Fetching REAL AFL data...")
        
        # Get real player statistics
        players_data = await real_afl_data.fetch_real_player_stats()
        print(f"✅ Real players data: {players_data['status']}")
        
        if players_data['status'] == 'success':
            print(f"📊 Found {len(players_data['players'])} real players")
            for player in players_data['players'][:5]:
                print(f"   {player}")
        
        # Get specific player
        clayton_data = await real_afl_data.fetch_specific_player_stats("Clayton Oliver")
        print(f"🎯 Clayton Oliver real data: {clayton_data}")
        
    asyncio.run(test_real_data())