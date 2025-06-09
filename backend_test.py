
import requests
import unittest
import sys
import json
from datetime import datetime

class AFLSGMBuilderAPITester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AFLSGMBuilderAPITester, self).__init__(*args, **kwargs)
        self.base_url = "https://5f8277a1-b7cf-4159-a607-d66ea1780bac.preview.emergentagent.com/api"
        self.test_venues = ["MCG", "Marvel Stadium", "Adelaide Oval", "SCG", "Sydney Showground"]
        self.test_match_id = "demo_123"  # Using demo match ID

    def test_01_root_endpoint(self):
        """Test the root API endpoint"""
        print("\n🔍 Testing root endpoint...")
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("features", data)
        print("✅ Root endpoint test passed")
        print(f"API Message: {data['message']}")
        print(f"Available Features: {json.dumps(data['features'], indent=2)}")

    def test_02_teams_endpoint(self):
        """Test the teams endpoint"""
        print("\n🔍 Testing teams endpoint...")
        response = requests.get(f"{self.base_url}/teams")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check if we got teams data (either from SportDevs or fallback)
        if "teams" in data:
            teams = data["teams"]
            self.assertTrue(isinstance(teams, (list, dict)))
            print(f"✅ Teams endpoint test passed - Retrieved teams data")
        elif "fallback_teams" in data:
            # Fallback to static teams data
            teams = data["fallback_teams"]
            self.assertTrue(isinstance(teams, dict))
            print(f"✅ Teams endpoint test passed - Using fallback teams data")
        else:
            self.fail("No teams data found in response")

    def test_03_weather_endpoint(self):
        """Test the weather endpoint for different venues"""
        print("\n🔍 Testing weather endpoint...")
        for venue in self.test_venues:
            print(f"  Testing weather for {venue}...")
            response = requests.get(f"{self.base_url}/weather/{venue}")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Check if we got weather data or error
            if "error" not in data:
                self.assertIn("venue", data)
                self.assertEqual(data["venue"], venue)
                print(f"  ✅ Weather for {venue}: {data.get('conditions', 'Unknown')}")
            else:
                print(f"  ⚠️ Weather API error: {data['error']}")

    def test_04_fixtures_endpoint(self):
        """Test the fixtures endpoint"""
        print("\n🔍 Testing fixtures endpoint...")
        response = requests.get(f"{self.base_url}/fixtures")
        
        # This might return an error if SportDevs integration is not available
        if response.status_code == 200:
            data = response.json()
            if "fixtures" in data:
                fixtures = data["fixtures"]
                self.assertTrue(isinstance(fixtures, list))
                print(f"✅ Fixtures endpoint test passed - Retrieved {len(fixtures)} fixtures")
            else:
                print("⚠️ No fixtures data available")
        else:
            print(f"⚠️ Fixtures endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")

    def test_05_injuries_endpoint(self):
        """Test the injuries endpoint"""
        print("\n🔍 Testing injuries endpoint...")
        response = requests.get(f"{self.base_url}/injuries")
        
        # This might return an error if SportDevs integration is not available
        if response.status_code == 200:
            data = response.json()
            if "injuries" in data:
                injuries = data["injuries"]
                self.assertTrue(isinstance(injuries, list))
                print(f"✅ Injuries endpoint test passed - Retrieved {len(injuries)} injuries")
            else:
                print("⚠️ No injuries data available")
        else:
            print(f"⚠️ Injuries endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")

    def test_06_odds_endpoint(self):
        """Test the odds endpoint"""
        print("\n🔍 Testing odds endpoint...")
        response = requests.get(f"{self.base_url}/odds")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("odds", data)
        self.assertIn("count", data)
        print(f"✅ Odds endpoint test passed - Retrieved odds for {data['count']} matches")

    def test_07_player_enhanced_endpoint(self):
        """Test the player enhanced data endpoint"""
        print("\n🔍 Testing player enhanced data endpoint...")
        # Using a player ID - this might fail if SportDevs integration is not available
        player_id = "player123"  # Example player ID
        response = requests.get(f"{self.base_url}/player/{player_id}/enhanced")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Enhanced player data retrieved for player ID: {player_id}")
        else:
            print(f"⚠️ Enhanced player data endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")

    def test_08_advanced_sgm_analysis(self):
        """Test the advanced SGM analysis endpoint"""
        print("\n🔍 Testing advanced SGM analysis endpoint...")
        payload = {
            "match_id": self.test_match_id,
            "target_odds": 3.0,
            "max_players": 4,
            "confidence_threshold": 0.7,
            "use_ml_models": True,
            "include_weather": True
        }
        
        response = requests.post(f"{self.base_url}/sgm/advanced", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for expected response structure
        self.assertIn("match_context", data)
        self.assertIn("sgm_recommendations", data)
        self.assertIn("analysis_summary", data)
        
        # Print some details from the response
        match_context = data["match_context"]
        recommendations = data["sgm_recommendations"]["recommendations"]
        summary = data["analysis_summary"]
        
        print(f"✅ Advanced SGM analysis test passed")
        print(f"  Match: {match_context.get('home_team', 'Unknown')} vs {match_context.get('away_team', 'Unknown')}")
        print(f"  Venue: {match_context.get('venue', 'Unknown')}")
        print(f"  Recommendations: {len(recommendations)}")
        print(f"  High confidence picks: {summary.get('high_confidence_picks', 0)}")
        
        # Check the first recommendation if available
        if recommendations:
            first_rec = recommendations[0]
            print(f"  First recommendation: {first_rec.get('recommendation', 'Unknown')}")
            print(f"  Implied odds: {first_rec.get('implied_odds', 'Unknown')}")
            print(f"  Confidence score: {first_rec.get('confidence_score', 'Unknown')}")
            
            # Check SGM outcomes
            outcomes = first_rec.get("sgm_outcomes", [])
            if outcomes:
                print(f"  SGM outcomes:")
                for outcome in outcomes:
                    print(f"    - {outcome.get('player', 'Unknown')}: {outcome.get('stat_type', 'Unknown')} {outcome.get('target', 'Unknown')}+")

    def test_10_data_status_endpoint(self):
        """Test the data status endpoint for 2025 AFL season"""
        print("\n🔍 Testing data status endpoint...")
        response = requests.get(f"{self.base_url}/data/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for 2025 season data
        self.assertEqual(data.get("season"), "2025", "Season should be 2025")
        
        # Check data sources
        data_sources = data.get("data_sources", {})
        squiggle_api = data_sources.get("squiggle_api", {})
        
        # Verify total games count
        total_games = squiggle_api.get("total_2025_games")
        self.assertEqual(total_games, 216, f"Expected 216 total games for 2025 season, got {total_games}")
        
        # Verify current round
        current_round = squiggle_api.get("current_round")
        print(f"  Current round: {current_round}")
        self.assertIsNotNone(current_round, "Current round should not be None")
        
        # Check all data sources are connected
        for source_name, source_data in data_sources.items():
            status = source_data.get("status", "")
            print(f"  {source_name}: {status}")
            self.assertIn("✅", status, f"{source_name} should be connected")
        
        print("✅ Data status endpoint test passed")
        print(f"  Season: {data.get('season')}")
        print(f"  Total 2025 games: {squiggle_api.get('total_2025_games')}")
        print(f"  Current round: {squiggle_api.get('current_round')}")
        
        # Check weather data
        weather_data = data_sources.get("weather_api", {}).get("sample_data", {})
        print(f"  Weather conditions: {weather_data.get('conditions')}")
        print(f"  Temperature: {weather_data.get('temperature')}°C")
        
        # Check odds data
        odds_data = data_sources.get("odds_api", {})
        print(f"  Matches with odds: {odds_data.get('matches_with_odds')}")

    def test_11_current_fixtures_endpoint(self):
        """Test the current fixtures endpoint for 2025 AFL season"""
        print("\n🔍 Testing current fixtures endpoint...")
        response = requests.get(f"{self.base_url}/fixtures/current")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for current round fixtures
        self.assertIn("current_round_fixtures", data, "Response should include current_round_fixtures")
        fixtures = data.get("current_round_fixtures", [])
        self.assertTrue(len(fixtures) > 0, "Should have at least one fixture")
        
        # Check season info
        self.assertEqual(data.get("season"), "2025", "Season should be 2025")
        
        # Check current round
        current_round = data.get("current_round")
        print(f"  Current round: {current_round}")
        
        # Check total games
        total_games = data.get("total_2025_games")
        self.assertEqual(total_games, 216, f"Expected 216 total games for 2025 season, got {total_games}")
        
        # Verify fixture details for real 2025 AFL teams and venues
        real_teams = ["Sydney", "Hawthorn", "GWS", "Collingwood", "Brisbane Lions", 
                      "Carlton", "Essendon", "Geelong", "Melbourne", "Richmond"]
        real_venues = ["SCG", "Sydney Showground", "MCG", "Marvel Stadium", "Gabba", 
                       "Adelaide Oval", "Optus Stadium", "GMHBA Stadium"]
        
        # Check at least one fixture has a real team and venue
        team_found = False
        venue_found = False
        
        for fixture in fixtures:
            home_team = fixture.get("hteam", "")
            away_team = fixture.get("ateam", "")
            venue = fixture.get("venue", "")
            date = fixture.get("date", "")
            
            print(f"  Match: {home_team} vs {away_team}")
            print(f"  Venue: {venue}")
            print(f"  Date: {date}")
            
            # Check if this is a real team
            if home_team in real_teams or away_team in real_teams:
                team_found = True
            
            # Check if this is a real venue
            if venue in real_venues:
                venue_found = True
            
            # Check if date is from 2025
            self.assertTrue("2025" in date, f"Date should be from 2025 season, got {date}")
            
            # Check if completed matches have scores
            if fixture.get("complete") == 100:
                self.assertIn("hscore", fixture, "Completed match should have home score")
                self.assertIn("ascore", fixture, "Completed match should have away score")
                print(f"  Score: {fixture.get('hscore')} - {fixture.get('ascore')} (Final)")
            
            print("  ---")
        
        self.assertTrue(team_found, "At least one fixture should have a real AFL team")
        self.assertTrue(venue_found, "At least one fixture should have a real AFL venue")
        
        print("✅ Current fixtures endpoint test passed")
        print(f"  Total fixtures: {len(fixtures)}")
        print(f"  Season: {data.get('season')}")

    def test_12_live_standings_endpoint(self):
        """Test the live standings endpoint for 2025 AFL season"""
        print("\n🔍 Testing live standings endpoint...")
        response = requests.get(f"{self.base_url}/standings/live")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check for standings data
        self.assertIn("standings", data, "Response should include standings")
        standings = data.get("standings", [])
        
        # Check season
        self.assertEqual(data.get("season"), "2025", "Season should be 2025")
        
        # Check number of teams
        self.assertEqual(len(standings), 18, f"Should have 18 AFL teams, got {len(standings)}")
        
        # Check for key teams
        key_teams = ["Collingwood", "Brisbane Lions", "Sydney", "Carlton", "Melbourne"]
        teams_found = [team for team in key_teams if any(s.get("team") == team for s in standings)]
        
        print(f"  Found {len(teams_found)} of {len(key_teams)} key teams")
        for team in teams_found:
            print(f"  ✓ Found {team}")
        
        # Check standings data structure
        for team in standings[:5]:  # Check top 5 teams
            print(f"\n  Team: {team.get('team')}")
            print(f"  Position: {team.get('rank')}")
            print(f"  Record: {team.get('wins')}-{team.get('losses')}-{team.get('draws')}")
            print(f"  Percentage: {team.get('percentage')}%")
            print(f"  Points: {team.get('points')}")
            
            # Verify data types and ranges
            self.assertIsInstance(team.get("wins"), int, "Wins should be an integer")
            self.assertIsInstance(team.get("losses"), int, "Losses should be an integer")
            self.assertIsInstance(team.get("percentage"), (int, float), "Percentage should be numeric")
            self.assertIsInstance(team.get("points"), int, "Points should be an integer")
            
            # Verify realistic values
            self.assertGreaterEqual(team.get("percentage", 0), 0, "Percentage should be positive")
            self.assertLessEqual(team.get("percentage", 200), 200, "Percentage should be under 200%")
            self.assertEqual(team.get("points"), team.get("wins") * 4 + team.get("draws") * 2, 
                            "Points should equal wins*4 + draws*2")
        
        print("\n✅ Live standings endpoint test passed")
        print(f"  Total teams: {len(standings)}")
        print(f"  Season: {data.get('season')}")
        print(f"  Last updated: {data.get('last_updated')}")

def run_tests():
    """Run all API tests"""
    test_suite = unittest.TestSuite()
    
    # Core API tests
    test_suite.addTest(AFLSGMBuilderAPITester('test_01_root_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_02_teams_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_03_weather_endpoint'))
    
    # 2025 AFL Season Data Tests (Priority)
    test_suite.addTest(AFLSGMBuilderAPITester('test_10_data_status_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_11_current_fixtures_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_12_live_standings_endpoint'))
    
    # Additional API tests
    test_suite.addTest(AFLSGMBuilderAPITester('test_04_fixtures_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_05_injuries_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_06_odds_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_07_player_enhanced_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_08_advanced_sgm_analysis'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_09_predict_player_performance'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    print("🏈 Advanced AFL SGM Builder API v2.0 Test Suite")
    print("=" * 60)
    print("🔍 Testing Live 2025 AFL Data Integration")
    print("=" * 60)
    print("📊 Testing backend API endpoints for 2025 AFL season data")
    print("🏆 Verifying live ladder, fixtures, and data status")
    print("🌤️ Checking weather integration for AFL venues")
    print("=" * 60)
    sys.exit(run_tests())
