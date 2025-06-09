# Real AFL Data Integration - 2025 Season Live Data
# Comprehensive data collection from multiple reliable sources

import httpx
import asyncio
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re
import logging

class RealAFLDataCollector:
    """Collect real AFL 2025 season data from multiple sources"""
    
    def __init__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        self.base_urls = {
            "afl_tables": "https://afltables.com",
            "footywire": "https://www.footywire.com",
            "afl_official": "https://api.afl.com.au"
        }
        
    async def get_current_round_matches(self) -> List[Dict]:
        """Get current round matches with live data"""
        try:
            # AFL Tables current round
            url = f"{self.base_urls['afl_tables']}/afl/seas/2025.html"
            response = await self.session.get(url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                matches = await self._parse_afl_tables_matches(soup)
                return matches
            else:
                logging.error(f"Failed to fetch AFL Tables data: {response.status_code}")
                return []
                
        except Exception as e:
            logging.error(f"Error fetching current round matches: {str(e)}")
            return []
    
    async def _parse_afl_tables_matches(self, soup) -> List[Dict]:
        """Parse AFL Tables HTML for current matches"""
        matches = []
        
        # Find the latest round results table
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 6:  # Match row should have multiple cells
                    try:
                        # Extract match data
                        date_cell = cells[0].text.strip() if cells[0] else ""
                        teams_cell = cells[1].text.strip() if cells[1] else ""
                        venue_cell = cells[2].text.strip() if cells[2] else ""
                        
                        if " v " in teams_cell:
                            home_team, away_team = teams_cell.split(" v ")
                            matches.append({
                                "date": date_cell,
                                "home_team": home_team.strip(),
                                "away_team": away_team.strip(),
                                "venue": venue_cell,
                                "source": "afl_tables"
                            })
                    except Exception as e:
                        continue
                        
        return matches[-20:] if matches else []  # Return latest 20 matches
    
    async def get_player_season_stats(self, player_name: str, team: str = None) -> Dict:
        """Get real player statistics for 2025 season"""
        try:
            # Try Footywire first
            stats = await self._get_footywire_player_stats(player_name, team)
            if stats:
                return stats
                
            # Fallback to AFL Tables
            return await self._get_afl_tables_player_stats(player_name, team)
            
        except Exception as e:
            logging.error(f"Error fetching player stats for {player_name}: {str(e)}")
            return {}
    
    async def _get_footywire_player_stats(self, player_name: str, team: str = None) -> Dict:
        """Get player stats from Footywire (most current)"""
        try:
            # Footywire player search
            search_name = player_name.replace(" ", "-").lower()
            url = f"{self.base_urls['footywire']}/afl/footy/ft_player_profile?playerid={search_name}"
            
            response = await self.session.get(url)
            if response.status_code != 200:
                return {}
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse 2025 season stats
            stats_table = soup.find('table', {'class': 'playerstats'})
            if not stats_table:
                return {}
                
            stats = {
                "player_name": player_name,
                "team": team,
                "source": "footywire",
                "season": 2025,
                "games_played": 0,
                "disposals": [],
                "goals": [],
                "marks": [],
                "tackles": [],
                "season_averages": {}
            }
            
            # Extract game-by-game stats
            rows = stats_table.find_all('tr')[1:]  # Skip header
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 10:
                    try:
                        game_stats = {
                            "round": cells[0].text.strip(),
                            "opponent": cells[1].text.strip(),
                            "disposals": int(cells[4].text.strip() or 0),
                            "goals": int(cells[5].text.strip() or 0),
                            "marks": int(cells[6].text.strip() or 0),
                            "tackles": int(cells[7].text.strip() or 0)
                        }
                        
                        stats["disposals"].append(game_stats["disposals"])
                        stats["goals"].append(game_stats["goals"])
                        stats["marks"].append(game_stats["marks"])
                        stats["tackles"].append(game_stats["tackles"])
                        stats["games_played"] += 1
                        
                    except (ValueError, IndexError):
                        continue
            
            # Calculate averages
            if stats["games_played"] > 0:
                stats["season_averages"] = {
                    "disposals": round(sum(stats["disposals"]) / stats["games_played"], 1),
                    "goals": round(sum(stats["goals"]) / stats["games_played"], 1),
                    "marks": round(sum(stats["marks"]) / stats["games_played"], 1),
                    "tackles": round(sum(stats["tackles"]) / stats["games_played"], 1)
                }
                
                # Recent form (last 5 games)
                recent_games = min(5, stats["games_played"])
                stats["recent_form"] = {
                    "disposals": round(sum(stats["disposals"][-recent_games:]) / recent_games, 1),
                    "goals": round(sum(stats["goals"][-recent_games:]) / recent_games, 1),
                    "marks": round(sum(stats["marks"][-recent_games:]) / recent_games, 1),
                    "tackles": round(sum(stats["tackles"][-recent_games:]) / recent_games, 1)
                }
                
                # Form factor calculation
                stats["form_factors"] = {}
                for stat in ["disposals", "goals", "marks", "tackles"]:
                    if stats["season_averages"][stat] > 0:
                        factor = stats["recent_form"][stat] / stats["season_averages"][stat]
                        stats["form_factors"][stat] = round(factor, 3)
                        
                        # Classify form trend
                        if factor > 1.15:
                            trend = "Hot"
                        elif factor < 0.85:
                            trend = "Cold"
                        else:
                            trend = "Average"
                        stats["form_factors"][f"{stat}_trend"] = trend
            
            return stats
            
        except Exception as e:
            logging.error(f"Footywire player stats error for {player_name}: {str(e)}")
            return {}
    
    async def _get_afl_tables_player_stats(self, player_name: str, team: str = None) -> Dict:
        """Fallback to AFL Tables for player stats"""
        try:
            # AFL Tables 2025 player stats
            url = f"{self.base_urls['afl_tables']}/afl/stats/players/2025.html"
            response = await self.session.get(url)
            
            if response.status_code != 200:
                return {}
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find player in stats table
            stats_table = soup.find('table')
            if not stats_table:
                return {}
                
            rows = stats_table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) > 5:
                    name_cell = cells[0].text.strip()
                    if player_name.lower() in name_cell.lower():
                        try:
                            return {
                                "player_name": player_name,
                                "team": cells[1].text.strip() if len(cells) > 1 else team,
                                "source": "afl_tables",
                                "season": 2025,
                                "games_played": int(cells[2].text.strip() or 0),
                                "season_averages": {
                                    "disposals": float(cells[3].text.strip() or 0),
                                    "goals": float(cells[4].text.strip() or 0),
                                    "marks": float(cells[5].text.strip() or 0),
                                    "tackles": float(cells[6].text.strip() or 0) if len(cells) > 6 else 0
                                }
                            }
                        except (ValueError, IndexError):
                            continue
            
            return {}
            
        except Exception as e:
            logging.error(f"AFL Tables player stats error for {player_name}: {str(e)}")
            return {}
    
    async def get_team_defensive_stats(self, team_name: str) -> Dict:
        """Get real team defensive statistics for 2025"""
        try:
            url = f"{self.base_urls['afl_tables']}/afl/teams/{team_name.lower()}/2025.html"
            response = await self.session.get(url)
            
            if response.status_code != 200:
                return {}
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract team defensive stats
            stats = {
                "team": team_name,
                "season": 2025,
                "source": "afl_tables",
                "defensive_stats": {
                    "points_against_per_game": 0,
                    "disposals_allowed_per_game": 0,
                    "goals_against_per_game": 0,
                    "tackles_per_game": 0
                }
            }
            
            # Parse team stats tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 5:
                        try:
                            # Look for defensive metrics
                            metric = cells[0].text.strip().lower()
                            value = float(cells[1].text.strip() or 0)
                            
                            if "points against" in metric:
                                stats["defensive_stats"]["points_against_per_game"] = value
                            elif "disposals against" in metric:
                                stats["defensive_stats"]["disposals_allowed_per_game"] = value
                                
                        except (ValueError, IndexError):
                            continue
            
            return stats
            
        except Exception as e:
            logging.error(f"Team stats error for {team_name}: {str(e)}")
            return {}
    
    async def get_injury_list(self) -> List[Dict]:
        """Get current AFL injury list from multiple sources"""
        try:
            injuries = []
            
            # Try AFL official injury list
            url = "https://www.afl.com.au/news/injury-list"
            response = await self.session.get(url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Parse injury information
                injury_items = soup.find_all('div', {'class': 'injury-item'})
                for item in injury_items:
                    try:
                        player_name = item.find('h3').text.strip()
                        team = item.find('span', {'class': 'team'}).text.strip()
                        injury_type = item.find('span', {'class': 'injury'}).text.strip()
                        status = item.find('span', {'class': 'status'}).text.strip()
                        
                        injuries.append({
                            "player_name": player_name,
                            "team": team,
                            "injury_type": injury_type,
                            "status": status,
                            "source": "afl_official",
                            "updated": datetime.now().isoformat()
                        })
                    except Exception:
                        continue
            
            return injuries
            
        except Exception as e:
            logging.error(f"Injury list error: {str(e)}")
            return []
    
    async def close(self):
        """Close the HTTP session"""
        await self.session.aclose()

# Integration with existing SGM system
class EnhancedSGMAnalyzer:
    """Enhanced SGM Analyzer with real AFL data"""
    
    def __init__(self):
        self.afl_data = RealAFLDataCollector()
    
    async def analyze_real_sgm(self, selections: List[Dict], venue: str, date: str) -> Dict:
        """Analyze SGM using real AFL data"""
        try:
            enhanced_predictions = []
            
            for selection in selections:
                player_name = selection["player"]
                stat_type = selection["stat_type"]
                threshold = selection["threshold"]
                
                # Get real player data
                real_stats = await self.afl_data.get_player_season_stats(player_name)
                
                if real_stats and "season_averages" in real_stats:
                    # Calculate probability using real data
                    season_avg = real_stats["season_averages"].get(stat_type, 0)
                    recent_avg = real_stats.get("recent_form", {}).get(stat_type, season_avg)
                    
                    # Use normal distribution for probability calculation
                    import numpy as np
                    from scipy import stats
                    
                    # Estimate standard deviation (conservative approach)
                    std_dev = season_avg * 0.3  # 30% of average as std dev
                    
                    # Calculate probability of exceeding threshold
                    if std_dev > 0:
                        z_score = (threshold - recent_avg) / std_dev
                        probability = 1 - stats.norm.cdf(z_score)
                        probability = max(0.05, min(0.95, probability))  # Clamp
                    else:
                        probability = 0.5  # Default if no data
                    
                    # Form factor
                    form_factor = recent_avg / season_avg if season_avg > 0 else 1.0
                    
                    enhanced_predictions.append({
                        "player": player_name,
                        "stat_type": stat_type,
                        "threshold": threshold,
                        "probability": round(probability, 3),
                        "season_average": season_avg,
                        "recent_form_average": recent_avg,
                        "form_factor": round(form_factor, 3),
                        "form_trend": real_stats.get("form_factors", {}).get(f"{stat_type}_trend", "Unknown"),
                        "games_played": real_stats.get("games_played", 0),
                        "data_source": real_stats.get("source", "unknown"),
                        "confidence": "High" if real_stats.get("games_played", 0) > 10 else "Medium"
                    })
                else:
                    # Fallback if no real data available
                    enhanced_predictions.append({
                        "player": player_name,
                        "stat_type": stat_type,
                        "threshold": threshold,
                        "probability": 0.5,
                        "data_source": "fallback",
                        "confidence": "Low",
                        "error": "No real data available"
                    })
            
            # Calculate combined probability
            individual_probs = [pred["probability"] for pred in enhanced_predictions]
            combined_prob = 1.0
            for prob in individual_probs:
                combined_prob *= prob
            
            # Apply correlation adjustments (simplified)
            correlation_factor = 0.95 if len(enhanced_predictions) > 1 else 1.0
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
                    "data_quality": "High" if all(p.get("confidence") == "High" for p in enhanced_predictions) else "Medium",
                    "recommendation": self._get_recommendation(final_combined_prob, implied_odds)
                },
                "venue": venue,
                "date": date,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Real SGM analysis error: {str(e)}")
            return {"error": str(e)}
    
    def _get_recommendation(self, probability: float, implied_odds: float) -> str:
        """Generate betting recommendation"""
        if probability > 0.25:
            return f"🔥 EXCELLENT VALUE - High probability ({probability*100:.1f}%) at ${implied_odds:.2f}"
        elif probability > 0.15:
            return f"✅ GOOD VALUE - Decent probability ({probability*100:.1f}%) at ${implied_odds:.2f}"
        elif probability > 0.10:
            return f"⚠️ FAIR VALUE - Moderate probability ({probability*100:.1f}%) at ${implied_odds:.2f}"
        else:
            return f"❌ POOR VALUE - Low probability ({probability*100:.1f}%) at ${implied_odds:.2f}"
    
    async def close(self):
        """Close data collector"""
        await self.afl_data.close()
