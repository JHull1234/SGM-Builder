import React, { useState, useEffect } from 'react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [matches, setMatches] = useState([]);
  const [players, setPlayers] = useState([]);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [sgmOutcomes, setSgmOutcomes] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [weather, setWeather] = useState(null);
  const [odds, setOdds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      const [matchesRes, playersRes, oddsRes, historyRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/matches`),
        fetch(`${BACKEND_URL}/api/players`),
        fetch(`${BACKEND_URL}/api/odds`),
        fetch(`${BACKEND_URL}/api/sgm/history`)
      ]);

      const matchesData = await matchesRes.json();
      const playersData = await playersRes.json();
      const oddsData = await oddsRes.json();
      const historyData = await historyRes.json();

      setMatches(matchesData);
      setPlayers(playersData);
      setOdds(oddsData);
      setHistory(historyData);
    } catch (error) {
      console.error('Error fetching initial data:', error);
    }
  };

  const addOutcome = () => {
    setSgmOutcomes([...sgmOutcomes, {
      id: Date.now(),
      type: 'disposals',
      player: '',
      value: '',
      operator: 'over'
    }]);
  };

  const updateOutcome = (id, field, value) => {
    setSgmOutcomes(sgmOutcomes.map(outcome => 
      outcome.id === id ? { ...outcome, [field]: value } : outcome
    ));
  };

  const removeOutcome = (id) => {
    setSgmOutcomes(sgmOutcomes.filter(outcome => outcome.id !== id));
  };

  const analyzeSGM = async () => {
    if (!selectedMatch || sgmOutcomes.length === 0) {
      alert('Please select a match and add at least one outcome');
      return;
    }

    setLoading(true);
    try {
      // Get weather data first
      const weatherRes = await fetch(`${BACKEND_URL}/api/weather/${selectedMatch.venue}`);
      const weatherData = await weatherRes.json();
      setWeather(weatherData);

      // Analyze SGM
      const sgmRes = await fetch(`${BACKEND_URL}/api/sgm/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          match_id: selectedMatch.match_id,
          outcomes: sgmOutcomes,
          venue: selectedMatch.venue
        })
      });

      const analysisData = await sgmRes.json();
      setAnalysis(analysisData);

      // Refresh history
      const historyRes = await fetch(`${BACKEND_URL}/api/sgm/history`);
      const historyData = await historyRes.json();
      setHistory(historyData);

    } catch (error) {
      console.error('Error analyzing SGM:', error);
      alert('Error analyzing SGM. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getValueColor = (value) => {
    if (value > 0.2) return '#2ecc71'; // Green
    if (value > 0) return '#f39c12'; // Orange
    return '#e74c3c'; // Red
  };

  const getConfidenceColor = (confidence) => {
    switch (confidence) {
      case 'High': return '#2ecc71';
      case 'Medium': return '#f39c12';
      default: return '#e74c3c';
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🏈 AFL Same Game Multi Analytics</h1>
          <p className="subtitle">Advanced betting analytics for correlated outcomes</p>
        </header>

        <div className="main-content">
          {/* Match Selection */}
          <div className="section">
            <h2>📅 Select Match</h2>
            <div className="match-grid">
              {matches.map(match => (
                <div 
                  key={match.match_id}
                  className={`match-card ${selectedMatch?.match_id === match.match_id ? 'selected' : ''}`}
                  onClick={() => setSelectedMatch(match)}
                >
                  <div className="match-teams">
                    <span className="team">{match.home_team}</span>
                    <span className="vs">vs</span>
                    <span className="team">{match.away_team}</span>
                  </div>
                  <div className="match-details">
                    <div className="venue">{match.venue}</div>
                    <div className="round">Round {match.round}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* SGM Builder */}
          <div className="section">
            <h2>🎯 Build Same Game Multi</h2>
            <div className="sgm-builder">
              {sgmOutcomes.map(outcome => (
                <div key={outcome.id} className="outcome-row">
                  <select 
                    value={outcome.player}
                    onChange={(e) => updateOutcome(outcome.id, 'player', e.target.value)}
                  >
                    <option value="">Select Player</option>
                    {players.map(player => (
                      <option key={player.id} value={player.name}>
                        {player.name} ({player.team})
                      </option>
                    ))}
                  </select>

                  <select
                    value={outcome.type}
                    onChange={(e) => updateOutcome(outcome.id, 'type', e.target.value)}
                  >
                    <option value="disposals">Disposals</option>
                    <option value="goals">Goals</option>
                    <option value="marks">Marks</option>
                    <option value="tackles">Tackles</option>
                  </select>

                  <select
                    value={outcome.operator}
                    onChange={(e) => updateOutcome(outcome.id, 'operator', e.target.value)}
                  >
                    <option value="over">Over</option>
                    <option value="under">Under</option>
                  </select>

                  <input
                    type="number"
                    placeholder="Value"
                    value={outcome.value}
                    onChange={(e) => updateOutcome(outcome.id, 'value', e.target.value)}
                  />

                  <button 
                    className="remove-btn"
                    onClick={() => removeOutcome(outcome.id)}
                  >
                    ✕
                  </button>
                </div>
              ))}

              <div className="sgm-actions">
                <button className="add-outcome-btn" onClick={addOutcome}>
                  + Add Outcome
                </button>
                <button 
                  className="analyze-btn" 
                  onClick={analyzeSGM}
                  disabled={loading || !selectedMatch || sgmOutcomes.length === 0}
                >
                  {loading ? 'Analyzing...' : '🔍 Analyze SGM'}
                </button>
              </div>
            </div>
          </div>

          {/* Analysis Results */}
          {analysis && (
            <div className="section">
              <h2>📊 SGM Analysis Results</h2>
              <div className="analysis-grid">
                <div className="analysis-card">
                  <h3>🎲 Correlation Score</h3>
                  <div className="score">{analysis.analysis.correlation_score}</div>
                  <div className="description">How independent are these outcomes?</div>
                </div>

                <div className="analysis-card">
                  <h3>💰 Value Rating</h3>
                  <div 
                    className="score"
                    style={{ color: getValueColor(analysis.analysis.value_rating) }}
                  >
                    {analysis.analysis.value_rating > 0 ? '+' : ''}{analysis.analysis.value_rating}
                  </div>
                  <div className="description">
                    {analysis.analysis.value_rating > 0.2 ? 'Excellent Value!' : 
                     analysis.analysis.value_rating > 0 ? 'Potential Value' : 'Poor Value'}
                  </div>
                </div>

                <div className="analysis-card">
                  <h3>🎯 Confidence</h3>
                  <div 
                    className="score"
                    style={{ color: getConfidenceColor(analysis.analysis.confidence) }}
                  >
                    {analysis.analysis.confidence}
                  </div>
                  <div className="description">Prediction confidence level</div>
                </div>

                <div className="analysis-card">
                  <h3>💡 Recommended Stake</h3>
                  <div className="score">{(analysis.analysis.recommended_stake * 100).toFixed(1)}%</div>
                  <div className="description">Of your bankroll</div>
                </div>
              </div>

              {/* Weather Impact */}
              {weather && (
                <div className="weather-section">
                  <h3>🌤️ Weather Impact Analysis</h3>
                  <div className="weather-grid">
                    <div className="weather-item">
                      <span>🌡️ Temperature:</span>
                      <span>{weather.temperature}°C</span>
                    </div>
                    <div className="weather-item">
                      <span>💨 Wind:</span>
                      <span>{weather.wind_speed} km/h {weather.wind_direction}</span>
                    </div>
                    <div className="weather-item">
                      <span>🌧️ Precipitation:</span>
                      <span>{weather.precipitation} mm</span>
                    </div>
                    <div className="weather-item">
                      <span>☁️ Conditions:</span>
                      <span>{weather.conditions}</span>
                    </div>
                  </div>
                  
                  <div className="weather-impact">
                    <h4>Impact on SGM:</h4>
                    <div className="impact-item">
                      <span>Wind Impact:</span>
                      <span style={{ color: analysis.analysis.weather_impact.wind_impact < 0 ? '#e74c3c' : '#2ecc71' }}>
                        {analysis.analysis.weather_impact.wind_impact.toFixed(3)}
                      </span>
                    </div>
                    <div className="impact-item">
                      <span>Rain Impact:</span>
                      <span style={{ color: analysis.analysis.weather_impact.rain_impact < 0 ? '#e74c3c' : '#2ecc71' }}>
                        {analysis.analysis.weather_impact.rain_impact.toFixed(3)}
                      </span>
                    </div>
                    <div className="impact-item">
                      <span>Total Impact:</span>
                      <span style={{ color: analysis.analysis.weather_impact.total_impact < 0 ? '#e74c3c' : '#2ecc71' }}>
                        {analysis.analysis.weather_impact.total_impact.toFixed(3)}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Historical Analysis */}
          {history.length > 0 && (
            <div className="section">
              <h2>📈 Recent SGM Analyses</h2>
              <div className="history-grid">
                {history.slice(0, 5).map((item, index) => (
                  <div key={index} className="history-card">
                    <div className="history-header">
                      <span className="match-info">
                        {item.outcomes.length} outcome SGM
                      </span>
                      <span 
                        className="value-badge"
                        style={{ backgroundColor: getValueColor(item.analysis.value_rating) }}
                      >
                        {item.analysis.value_rating > 0 ? '+' : ''}{item.analysis.value_rating.toFixed(2)}
                      </span>
                    </div>
                    <div className="history-details">
                      <div>Correlation: {item.analysis.correlation_score}</div>
                      <div>Confidence: {item.analysis.confidence}</div>
                      <div>Stake: {(item.analysis.recommended_stake * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Market Odds Comparison */}
          {odds.length > 0 && (
            <div className="section">
              <h2>💹 Market Odds</h2>
              <div className="odds-grid">
                {odds.slice(0, 3).map((match, index) => (
                  <div key={index} className="odds-card">
                    <h4>{match.home_team} vs {match.away_team}</h4>
                    {match.bookmakers && match.bookmakers[0] && (
                      <div className="bookmaker-odds">
                        <h5>{match.bookmakers[0].title}</h5>
                        {match.bookmakers[0].markets && match.bookmakers[0].markets[0] && (
                          <div className="market-odds">
                            {match.bookmakers[0].markets[0].outcomes.map((outcome, idx) => (
                              <div key={idx} className="outcome-odds">
                                <span>{outcome.name}</span>
                                <span className="odds-value">${outcome.price}</span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
