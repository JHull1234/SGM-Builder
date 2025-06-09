"""
AFL Statistics Integration - Phase 1: Free Data Sources
Build comprehensive player statistics database using free sources
"""

import pandas as pd
import requests
import sqlite3
from datetime import datetime
import os

class AFLStatsIntegrator:
    def __init__(self):
        self.db_path = "/app/afl_stats.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for AFL statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Player statistics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS player_stats (
                player_id TEXT,
                name TEXT,
                team TEXT,
                position TEXT,
                round_number INTEGER,
                disposals INTEGER,
                goals INTEGER,
                marks INTEGER,
                tackles INTEGER,
                fantasy_points INTEGER,
                date TEXT,
                opponent TEXT,
                venue TEXT,
                PRIMARY KEY (player_id, round_number)
            )
        ''')
        
        # Player averages table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS player_averages (
                player_id TEXT PRIMARY KEY,
                name TEXT,
                team TEXT,
                position TEXT,
                games_played INTEGER,
                avg_disposals REAL,
                avg_goals REAL,
                avg_marks REAL,
                avg_tackles REAL,
                disposal_20_plus_rate REAL,
                disposal_25_plus_rate REAL,
                goal_1_plus_rate REAL,
                goal_2_plus_rate REAL,
                last_updated TEXT
            )
        ''')
        
        # Team selections table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS team_selections (
                match_id TEXT,
                team TEXT,
                player_id TEXT,
                player_name TEXT,
                position TEXT,
                selected BOOLEAN,
                played BOOLEAN,
                date TEXT,
                opponent TEXT,
                PRIMARY KEY (match_id, player_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def download_dfs_australia_data(self):
        """Download AFL statistics from DFS Australia"""
        # Note: This would need to be implemented with actual DFS Australia CSV URLs
        # They update after each round with comprehensive player statistics
        pass
    
    def calculate_player_probabilities(self, player_id: str, stat_type: str, threshold: int) -> float:
        """Calculate real probability based on historical performance"""
        conn = sqlite3.connect(self.db_path)
        
        # Get player's game log
        query = f"""
            SELECT {stat_type} 
            FROM player_stats 
            WHERE player_id = ? 
            ORDER BY round_number DESC 
            LIMIT 10
        """
        
        results = conn.execute(query, (player_id,)).fetchall()
        conn.close()
        
        if not results:
            return 0.5  # Default if no data
        
        # Calculate actual success rate
        successful_games = sum(1 for game in results if game[0] >= threshold)
        total_games = len(results)
        
        return successful_games / total_games if total_games > 0 else 0.5
    
    def get_position_based_averages(self, position: str) -> dict:
        """Get realistic averages by position"""
        position_averages = {
            "Midfielder": {
                "avg_disposals": 24.5,
                "avg_goals": 0.8,
                "avg_marks": 4.2,
                "disposal_20_plus_rate": 0.75,
                "disposal_25_plus_rate": 0.45
            },
            "Forward": {
                "avg_disposals": 14.2,
                "avg_goals": 1.8,
                "avg_marks": 5.1,
                "goal_1_plus_rate": 0.65,
                "goal_2_plus_rate": 0.35
            },
            "Defender": {
                "avg_disposals": 18.6,
                "avg_goals": 0.2,
                "avg_marks": 6.3,
                "disposal_20_plus_rate": 0.45,
                "disposal_25_plus_rate": 0.15
            },
            "Ruckman": {
                "avg_disposals": 14.8,
                "avg_goals": 0.9,
                "avg_marks": 3.2,
                "disposal_20_plus_rate": 0.25,
                "disposal_25_plus_rate": 0.08  # Very rare for ruckmen
            }
        }
        
        return position_averages.get(position, position_averages["Midfielder"])

# Example usage for proper SGM calculation
def build_realistic_sgm(player_data: list, weather_conditions: dict, venue: str):
    """Build SGM with REAL statistical backing"""
    
    integrator = AFLStatsIntegrator()
    predictions = []
    
    for player in player_data:
        # Get real probability based on historical performance
        prob = integrator.calculate_player_probabilities(
            player['id'], 
            player['stat_type'], 
            player['threshold']
        )
        
        # Apply weather and venue modifiers (small, realistic adjustments)
        weather_modifier = 1.0
        if weather_conditions.get('precipitation', 0) > 5:  # Heavy rain
            if player['stat_type'] == 'goals':
                weather_modifier *= 0.85  # Goals harder in rain
            elif player['stat_type'] == 'disposals':
                weather_modifier *= 1.05  # More contested possessions
        
        venue_modifier = 1.0
        if venue == "MCG" and player['stat_type'] == 'disposals':
            venue_modifier *= 1.03  # Bigger ground, slightly more running
        
        final_prob = prob * weather_modifier * venue_modifier
        final_prob = max(0.05, min(0.95, final_prob))  # Clamp between 5-95%
        
        predictions.append({
            'player': player['name'],
            'selection': f"{player['threshold']}+ {player['stat_type']}",
            'probability': final_prob,
            'historical_rate': prob,
            'weather_impact': weather_modifier - 1,
            'venue_impact': venue_modifier - 1
        })
    
    return predictions

if __name__ == "__main__":
    # Initialize the statistics integration
    integrator = AFLStatsIntegrator()
    print("AFL Statistics Database initialized")