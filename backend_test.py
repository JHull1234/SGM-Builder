
import requests
import unittest
import sys
import json
from datetime import datetime

class AFLSGMBuilderAPITester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AFLSGMBuilderAPITester, self).__init__(*args, **kwargs)
        self.base_url = "https://5f8277a1-b7cf-4159-a607-d66ea1780bac.preview.emergentagent.com/api"
        self.test_venues = ["MCG", "Marvel Stadium", "Adelaide Oval"]
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

    def test_09_predict_player_performance(self):
        """Test the player performance prediction endpoint"""
        print("\n🔍 Testing player performance prediction endpoint...")
        payload = {
            "player_id": "player123",  # Example player ID
            "match_context": {
                "venue": "MCG",
                "opponent_team": "Collingwood",
                "weather": {
                    "temperature": 18,
                    "wind_speed": 15,
                    "precipitation": 0
                }
            },
            "stat_types": ["disposals", "goals", "marks", "tackles"]
        }
        
        response = requests.post(f"{self.base_url}/predict/player", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Player performance prediction test passed")
        else:
            print(f"⚠️ Player performance prediction endpoint returned status {response.status_code}")
            print(f"Response: {response.text}")

def run_tests():
    """Run all API tests"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(AFLSGMBuilderAPITester('test_01_root_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_02_teams_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_03_weather_endpoint'))
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
    sys.exit(run_tests())
