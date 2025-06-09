
import requests
import unittest
import sys
import json
from datetime import datetime

class AFLSGMBuilderAPITester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AFLSGMBuilderAPITester, self).__init__(*args, **kwargs)
        self.base_url = "https://5f8277a1-b7cf-4159-a607-d66ea1780bac.preview.emergentagent.com/api"
        self.test_players = ["Clayton Oliver", "Christian Petracca", "Marcus Bontempelli", "Jeremy Cameron"]
        self.test_venues = ["MCG", "Marvel Stadium", "Adelaide Oval"]

    def test_01_root_endpoint(self):
        """Test the root API endpoint"""
        print("\n🔍 Testing root endpoint...")
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        print("✅ Root endpoint test passed")

    def test_02_matches_endpoint(self):
        """Test the matches endpoint"""
        print("\n🔍 Testing matches endpoint...")
        response = requests.get(f"{self.base_url}/matches")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("matches", data)
        self.assertIn("count", data)
        print(f"✅ Matches endpoint test passed - Retrieved {data['count']} matches")

    def test_03_venues_endpoint(self):
        """Test the venues endpoint"""
        print("\n🔍 Testing venues endpoint...")
        response = requests.get(f"{self.base_url}/venues")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("venues", data)
        venues = data["venues"]
        self.assertTrue(len(venues) > 0)
        print(f"✅ Venues endpoint test passed - Retrieved {len(venues)} venues")

    def test_04_teams_endpoint(self):
        """Test the teams endpoint"""
        print("\n🔍 Testing teams endpoint...")
        response = requests.get(f"{self.base_url}/teams")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("teams", data)
        teams = data["teams"]
        self.assertTrue(len(teams) > 0)
        print(f"✅ Teams endpoint test passed - Retrieved {len(teams)} teams")

    def test_05_weather_endpoint(self):
        """Test the weather endpoint for different venues"""
        print("\n🔍 Testing weather endpoint...")
        for venue in self.test_venues:
            print(f"  Testing weather for {venue}...")
            response = requests.get(f"{self.base_url}/weather/{venue}")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("venue", data)
            self.assertEqual(data["venue"], venue)
            print(f"  ✅ Weather for {venue}: {data.get('conditions', 'Unknown')}")

    def test_06_odds_endpoint(self):
        """Test the odds endpoint"""
        print("\n🔍 Testing odds endpoint...")
        response = requests.get(f"{self.base_url}/odds")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("odds", data)
        self.assertIn("count", data)
        print(f"✅ Odds endpoint test passed - Retrieved odds for {data['count']} matches")

    def test_07_enhanced_player_analysis(self):
        """Test the enhanced player analysis endpoint"""
        print("\n🔍 Testing enhanced player analysis endpoint...")
        for player in self.test_players:
            print(f"  Testing analysis for {player}...")
            response = requests.get(f"{self.base_url}/player/{player}/enhanced-analysis")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("player", data)
            self.assertIn("form_analysis", data)
            self.assertIn("injury_analysis", data)
            print(f"  ✅ Enhanced analysis for {player} retrieved")

    def test_08_player_dashboard(self):
        """Test the player dashboard endpoint"""
        print("\n🔍 Testing player dashboard endpoint...")
        for player in self.test_players:
            print(f"  Testing dashboard for {player}...")
            response = requests.get(f"{self.base_url}/analytics/player-dashboard/{player}")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("player_info", data)
            self.assertIn("form_analysis", data)
            self.assertIn("betting_insights", data)
            print(f"  ✅ Dashboard for {player} retrieved")

    def test_09_sgm_analyze(self):
        """Test the SGM analyze endpoint"""
        print("\n🔍 Testing SGM analyze endpoint...")
        payload = {
            "match_id": "12345",
            "venue": "MCG",
            "date": datetime.now().isoformat(),
            "selections": [
                {
                    "player": "Clayton Oliver",
                    "stat_type": "disposals",
                    "threshold": 25
                },
                {
                    "player": "Christian Petracca",
                    "stat_type": "goals",
                    "threshold": 2
                }
            ]
        }
        response = requests.post(f"{self.base_url}/sgm/analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("match_info", data)
        self.assertIn("weather_conditions", data)
        self.assertIn("sgm_analysis", data)
        print("✅ SGM analyze endpoint test passed")

    def test_10_advanced_sgm_analyze(self):
        """Test the advanced SGM analyze endpoint"""
        print("\n🔍 Testing advanced SGM analyze endpoint...")
        payload = {
            "match_id": "12345",
            "venue": "MCG",
            "date": datetime.now().isoformat(),
            "selections": [
                {
                    "player": "Clayton Oliver",
                    "stat_type": "disposals",
                    "threshold": 25
                },
                {
                    "player": "Christian Petracca",
                    "stat_type": "goals",
                    "threshold": 2
                }
            ]
        }
        response = requests.post(f"{self.base_url}/sgm/advanced-analyze", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("match_info", data)
        self.assertIn("weather_conditions", data)
        self.assertIn("enhanced_predictions", data)
        self.assertIn("synergy_analysis", data)
        self.assertIn("combined_analysis", data)
        print("✅ Advanced SGM analyze endpoint test passed")

    def test_11_auto_recommend_sgm(self):
        """Test the auto recommend SGM endpoint"""
        print("\n🔍 Testing auto recommend SGM endpoint...")
        target_odds = 5.0
        response = requests.get(f"{self.base_url}/sgm/auto-recommend/{target_odds}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("target_odds", data)
        self.assertIn("auto_recommendations", data)
        print("✅ Auto recommend SGM endpoint test passed")

    def test_12_ml_predict_performance(self):
        """Test the ML predict performance endpoint"""
        print("\n🔍 Testing ML predict performance endpoint...")
        payload = {
            "player_name": "Clayton Oliver",
            "stat_type": "disposals",
            "threshold": 25,
            "match_context": {
                "venue": "MCG",
                "opponent_team": "Collingwood",
                "weather": {
                    "temperature": 18,
                    "wind_speed": 15,
                    "precipitation": 0
                }
            }
        }
        response = requests.post(f"{self.base_url}/ml/predict-performance", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("player", data)
        self.assertIn("ml_prediction", data)
        self.assertIn("form_factors", data)
        print("✅ ML predict performance endpoint test passed")

def run_tests():
    """Run all API tests"""
    test_suite = unittest.TestSuite()
    test_suite.addTest(AFLSGMBuilderAPITester('test_01_root_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_02_matches_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_03_venues_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_04_teams_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_05_weather_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_06_odds_endpoint'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_07_enhanced_player_analysis'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_08_player_dashboard'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_09_sgm_analyze'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_10_advanced_sgm_analyze'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_11_auto_recommend_sgm'))
    test_suite.addTest(AFLSGMBuilderAPITester('test_12_ml_predict_performance'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    print("🏈 AFL SGM Builder API Test Suite")
    print("=" * 50)
    sys.exit(run_tests())
