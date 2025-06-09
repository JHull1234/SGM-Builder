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
  
  // AI SGM Picker state
  const [targetOdds, setTargetOdds] = useState(3.50);
  const [aiRecommendations, setAiRecommendations] = useState([]);
  const [aiLoading, setAiLoading] = useState(false);
  const [forumIntel, setForumIntel] = useState(null);

  useEffect(() => {
    fetchInitialData();
    fetchForumIntelligence();
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

  const fetchForumIntelligence = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/forum/intelligence`);
      const data = await response.json();
      setForumIntel(data);
    } catch (error) {
      console.error('Error fetching forum intelligence:', error);
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

  const getAiRecommendations = async () => {
    setAiLoading(true);
    try {
      const response = await fetch(`${BACKEND_URL}/api/ai/recommend-sgm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          target_odds: targetOdds,
          match_context: {
            venue: "MCG",
            weather: {
              temperature: 20,
              wind_speed: 10,
              humidity: 60,
              precipitation: 0
            },
            opponent_team: "Brisbane"
          }
        })
      });

      const data = await response.json();
      setAiRecommendations(data.ai_recommendations || []);
    } catch (error) {
      console.error('Error getting AI recommendations:', error);
      alert('Error getting AI recommendations. Please try again.');
    } finally {
      setAiLoading(false);
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

  // Helper function to safely access analysis data
  const getAnalysisValue = (path, defaultValue = 0) => {
    if (!analysis) return defaultValue;
    
    // Try new format first (advanced_analysis)
    if (analysis.advanced_analysis) {
      const pathParts = path.split('.');
      let value = analysis.advanced_analysis;
      for (const part of pathParts) {
        value = value?.[part];
      }
      if (value !== undefined) return value;
    }
    
    // Fallback to old format (analysis)
    if (analysis.analysis) {
      const pathParts = path.split('.');
      let value = analysis.analysis;
      for (const part of pathParts) {
        value = value?.[part];
      }
      if (value !== undefined) return value;
    }
    
    return defaultValue;
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>🤖 AI-Powered AFL SGM Analytics</h1>
          <p className="subtitle">Professional-grade betting analytics with machine learning</p>
        </header>

        <div className="main-content">
          {/* AI SGM Picker */}
          <div className="section">
            <h2>🎯 AI SGM Picker</h2>
            <div className="ai-picker">
              <div className="target-odds-input">
                <label>Target Odds:</label>
                <input
                  type="number"
                  step="0.10"
                  min="1.10"
                  max="20.00"
                  value={targetOdds}
                  onChange={(e) => setTargetOdds(parseFloat(e.target.value))}
                />
                <button 
                  className="ai-recommend-btn"
                  onClick={getAiRecommendations}
                  disabled={aiLoading}
                >
                  {aiLoading ? '🤖 Analyzing...' : '🎯 Get AI Recommendations'}
                </button>
              </div>

              {aiRecommendations.length > 0 && (
                <div className="ai-recommendations">
                  <h3>🔥 AI Recommendations for {targetOdds} Odds</h3>
                  {aiRecommendations.map((rec, index) => (
                    <div key={index} className="ai-rec-card">
                      <div className="rec-header">
                        <span className="rec-title">Option {index + 1}</span>
                        <span 
                          className="value-badge"
                          style={{ backgroundColor: getValueColor(rec.value_rating) }}
                        >
                          {rec.value_rating > 0 ? '+' : ''}{(rec.value_rating * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="rec-outcomes">
                        {rec.sgm_outcomes.map((outcome, idx) => (
                          <div key={idx} className="outcome-item">
                            <span className="player-name">{outcome.player}</span>
                            <span className="stat-line">{outcome.target}+ {outcome.stat_type}</span>
                            <span className="probability">({(outcome.implied_probability * 100).toFixed(1)}%)</span>
                          </div>
                        ))}
                      </div>
                      
                      <div className="rec-analysis">
                        <div className="analysis-item">
                          <span>Fair Odds:</span>
                          <span className="odds-value">{rec.implied_odds}</span>
                        </div>
                        <div className="analysis-item">
                          <span>Confidence:</span>
                          <span style={{ color: getConfidenceColor(rec.confidence_breakdown?.overall_rating || 'Medium') }}>
                            {rec.confidence_breakdown?.overall_rating || 'Medium'}
                          </span>
                        </div>
                      </div>
                      
                      <div className="rec-recommendation">
                        {rec.recommendation}
                      </div>
                      
                      {rec.ai_insights && rec.ai_insights.length > 0 && (
                        <div className="ai-insights">
                          {rec.ai_insights.map((insight, idx) => (
                            <div key={idx} className="insight-item">{insight}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Forum Intelligence */}
          {forumIntel && (
            <div className="section">
              <h2>📱 Forum Intelligence</h2>
              <div className="forum-intel">
                {forumIntel.sharp_plays && forumIntel.sharp_plays.length > 0 && (
                  <div className="intel-section">
                    <h3>🔍 Sharp Plays</h3>
                    {forumIntel.sharp_plays.map((play, index) => (
                      <div key={index} className="intel-card">
                        <div className="intel-content">{play.content}</div>
                        <div className="intel-meta">
                          <span>by {play.author}</span>
                          <span>👍 {play.upvotes}</span>
                          <span>Confidence: {(play.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                
                {forumIntel.injury_intel && forumIntel.injury_intel.length > 0 && (
                  <div className="intel-section">
                    <h3>🏥 Injury Intelligence</h3>
                    {forumIntel.injury_intel.map((intel, index) => (
                      <div key={index} className="intel-card injury-intel">
                        <div className="intel-content">{intel.content}</div>
                        <div className="intel-meta">
                          <span>by {intel.author}</span>
                          <span className={`severity ${intel.severity?.toLowerCase()}`}>
                            {intel.severity} Risk
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Match Selection */}
          <div className="section">
            <h2>📅 Select Match</h2>
            <div className="match-grid">
              {matches.length === 0 ? (
                <div className="no-matches">
                  <p>🏈 No upcoming matches available</p>
                  <p style={{fontSize: '0.9em', color: '#666'}}>
                    Note: Currently showing off-season. During AFL season, live matches will appear here.
                  </p>
                </div>
              ) : (
                matches.map(match => (
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
                ))
              )}
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
                  disabled={loading || sgmOutcomes.length === 0}
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
                  <div className="score">{getAnalysisValue('correlation_analysis.final_score', getAnalysisValue('correlation_score')).toFixed(3)}</div>
                  <div className="description">How independent are these outcomes?</div>
                </div>

                <div className="analysis-card">
                  <h3>💰 Value Rating</h3>
                  <div 
                    className="score"
                    style={{ color: getValueColor(getAnalysisValue('value_rating')) }}
                  >
                    {getAnalysisValue('value_rating') > 0 ? '+' : ''}{getAnalysisValue('value_rating').toFixed(3)}
                  </div>
                  <div className="description">
                    {getAnalysisValue('value_rating') > 0.2 ? 'Excellent Value!' : 
                     getAnalysisValue('value_rating') > 0 ? 'Potential Value' : 'Poor Value'}
                  </div>
                </div>

                <div className="analysis-card">
                  <h3>🎯 Confidence</h3>
                  <div 
                    className="score"
                    style={{ color: getConfidenceColor(getAnalysisValue('confidence', 'Medium')) }}
                  >
                    {getAnalysisValue('confidence', 'Medium')}
                  </div>
                  <div className="description">Prediction confidence level</div>
                </div>

                <div className="analysis-card">
                  <h3>💡 Recommended Stake</h3>
                  <div className="score">{(getAnalysisValue('recommended_stake') * 100).toFixed(1)}%</div>
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
                  
                  {getAnalysisValue('weather_impact') && (
                    <div className="weather-impact">
                      <h4>Impact on SGM:</h4>
                      <div className="impact-item">
                        <span>Wind Impact:</span>
                        <span style={{ color: getAnalysisValue('weather_impact.wind_impact') < 0 ? '#e74c3c' : '#2ecc71' }}>
                          {getAnalysisValue('weather_impact.wind_impact', 0).toFixed(3)}
                        </span>
                      </div>
                      <div className="impact-item">
                        <span>Rain Impact:</span>
                        <span style={{ color: getAnalysisValue('weather_impact.rain_impact') < 0 ? '#e74c3c' : '#2ecc71' }}>
                          {getAnalysisValue('weather_impact.rain_impact', 0).toFixed(3)}
                        </span>
                      </div>
                      <div className="impact-item">
                        <span>Total Impact:</span>
                        <span style={{ color: getAnalysisValue('weather_impact.total_impact') < 0 ? '#e74c3c' : '#2ecc71' }}>
                          {getAnalysisValue('weather_impact.total_impact', 0).toFixed(3)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Historical Analysis */}
          {history.length > 0 && (
            <div className="section">
              <h2>📈 Recent SGM Analyses</h2>
              <div className="history-grid">
                {history.slice(0, 5).map((item, index) => {
                  const analysis = item.advanced_analysis || item.analysis || {};
                  return (
                    <div key={index} className="history-card">
                      <div className="history-header">
                        <span className="match-info">
                          {item.outcomes?.length || 0} outcome SGM
                        </span>
                        <span 
                          className="value-badge"
                          style={{ backgroundColor: getValueColor(analysis.value_rating || 0) }}
                        >
                          {(analysis.value_rating || 0) > 0 ? '+' : ''}{(analysis.value_rating || 0).toFixed(2)}
                        </span>
                      </div>
                      <div className="history-details">
                        <div>Correlation: {(analysis.correlation_score || analysis.correlation_analysis?.final_score || 0).toFixed(2)}</div>
                        <div>Confidence: {analysis.confidence || 'Medium'}</div>
                        <div>Stake: {((analysis.recommended_stake || 0) * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                  );
                })}
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
