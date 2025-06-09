# API Key Detection and Testing
# Test various sports API providers to identify the correct service

import httpx
import asyncio

async def test_api_key_providers():
    """Test the API key against multiple sports data providers"""
    
    api_key = "0cd3a2f69535f9834e2dc7d695f20525"
    
    # Test different providers and their authentication methods
    test_configs = [
        {
            "name": "API Sports (api-sports.io)",
            "base_url": "https://api.api-sports.io",
            "headers": {"x-apisports-key": api_key},
            "endpoints": ["/status", "/football/status", "/basketball/status"]
        },
        {
            "name": "RapidAPI Sports",
            "base_url": "https://api-football.p.rapidapi.com/v3",
            "headers": {"x-rapidapi-key": api_key, "x-rapidapi-host": "api-football.p.rapidapi.com"},
            "endpoints": ["/status", "/timezone"]
        },
        {
            "name": "SportsData.io",
            "base_url": "https://api.sportsdata.io",
            "headers": {"Ocp-Apim-Subscription-Key": api_key},
            "endpoints": ["/v3/nfl/scores/json/AreAnyGamesInProgress", "/api/afl/status"]
        },
        {
            "name": "ESPN API",
            "base_url": "https://site.api.espn.com/apis/site/v2",
            "headers": {"Authorization": f"Bearer {api_key}"},
            "endpoints": ["/sports/football/afl/news", "/sports"]
        },
        {
            "name": "The Sports DB",
            "base_url": "https://www.thesportsdb.com/api/v1/json",
            "headers": {"api_key": api_key},
            "endpoints": [f"/{api_key}/search_all_teams.php?l=Australian%20Football%20League"]
        },
        {
            "name": "SportRadar",
            "base_url": "https://api.sportradar.us",
            "headers": {"api_key": api_key},
            "endpoints": [f"/aussie-rules/trial/v2/en/sport_events.json?api_key={api_key}"]
        }
    ]
    
    results = {}
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for config in test_configs:
            provider_results = []
            
            for endpoint in config["endpoints"]:
                try:
                    response = await client.get(
                        f"{config['base_url']}{endpoint}",
                        headers=config["headers"]
                    )
                    
                    provider_results.append({
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "response_snippet": response.text[:200] if response.text else "No response",
                        "success": response.status_code == 200
                    })
                    
                except Exception as e:
                    provider_results.append({
                        "endpoint": endpoint,
                        "error": str(e)
                    })
            
            results[config["name"]] = provider_results
    
    return results

# Run the test
import json
test_results = asyncio.run(test_api_key_providers())
print(json.dumps(test_results, indent=2))