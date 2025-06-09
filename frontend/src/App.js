import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// SGM Builder Component
const SGMBuilder = () => {
  const [matches, setMatches] = useState([]);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [venues, setVenues] = useState([]);
  const [teams, setTeams] = useState({});
  const [weather, setWeather] = useState(null);
  const [selections, setSelections] = useState([]);
  const [sgmAnalysis, setSgmAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch initial data
  useEffect(() => {
    fetchMatches();
    fetchVenues();
    fetchTeams();
  }, []);

  const fetchMatches = async () => {
    try {
      const response = await axios.get(`${API}/matches`);
      setMatches(response.data.matches || []);
    } catch (error) {
      console.error("Error fetching matches:", error);
    }
  };

  const fetchVenues = async () => {
    try {
      const response = await axios.get(`${API}/venues`);
      setVenues(response.data.venues || []);
    } catch (error) {
      console.error("Error fetching venues:", error);
    }
  };

  const fetchTeams = async () => {
    try {
      const response = await axios.get(`${API}/teams`);
      setTeams(response.data.teams || {});
    } catch (error) {
      console.error("Error fetching teams:", error);
    }
  };

  const fetchWeather = async (venue, date) => {
    try {
      const response = await axios.get(`${API}/weather/${venue}`, {
        params: date ? { date } : {}
      });
      setWeather(response.data);
    } catch (error) {
      console.error("Error fetching weather:", error);
    }
  };

  const handleMatchSelect = (match) => {
    setSelectedMatch(match);
    if (match.venue) {
      fetchWeather(match.venue, match.date);
    }
    setSelections([]);
    setSgmAnalysis(null);
  };

  const addSelection = () => {
    setSelections([
      ...selections,
      {
        id: Date.now(),
        player: "",
        stat_type: "disposals",
        threshold: 20
      }
    ]);
  };

  const updateSelection = (id, field, value) => {
    setSelections(selections.map(sel => 
      sel.id === id ? { ...sel, [field]: value } : sel
    ));
  };

  const removeSelection = (id) => {
    setSelections(selections.filter(sel => sel.id !== id));
  };

  const analyzeSGM = async () => {
    if (!selectedMatch || selections.length === 0) {
      alert("Please select a match and add at least one selection");
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/sgm/analyze`, {
        match_id: selectedMatch.id,
        venue: selectedMatch.venue,
        date: selectedMatch.date,
        selections: selections.map(sel => ({
          player: sel.player,
          stat_type: sel.stat_type,
          threshold: parseFloat(sel.threshold)
        }))
      });
      
      setSgmAnalysis(response.data);
    } catch (error) {
      console.error("Error analyzing SGM:", error);
      alert("Error analyzing SGM. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const getRecommendationColor = (recommendation) => {
    switch (recommendation) {
      case "value": return "#22c55e";
      case "avoid": return "#ef4444";
      default: return "#f59e0b";
    }
  };

  const formatProbability = (prob) => {
    return `${(prob * 100).toFixed(1)}%`;
  };

  return (
    <div className="sgm-builder">
      <header className="sgm-header">
        <h1>🏈 AFL SGM Builder</h1>
        <p>Build and analyze Same Game Multi bets with weather & statistical insights</p>
      </header>

      {/* Match Selection */}
      <div className="section">
        <h2>📅 Select Match</h2>
        <div className="matches-grid">
          {matches.slice(0, 6).map((match, index) => (
            <div 
              key={index}
              className={`match-card ${selectedMatch?.id === match.id ? 'selected' : ''}`}
              onClick={() => handleMatchSelect(match)}
            >
              <div className="match-teams">
                <span className="team">{match.hteam}</span>
                <span className="vs">vs</span>
                <span className="team">{match.ateam}</span>
              </div>
              <div className="match-details">
                <div className="venue">{match.venue}</div>
                <div className="round">Round {match.round}</div>
                <div className="date">{match.date}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Weather Conditions */}
      {weather && (
        <div className="section">
          <h2>🌤️ Weather Conditions</h2>
          <div className="weather-grid">
            <div className="weather-item">
              <span className="label">Temperature:</span>
              <span className="value">
                {weather.temperature !== undefined ? `${weather.temperature}°C` : 
                 `${weather.min_temp}°C - ${weather.max_temp}°C`}
              </span>
            </div>
            <div className="weather-item">
              <span className="label">Wind:</span>
              <span className="value">
                {weather.wind_speed || weather.max_wind} km/h
              </span>
            </div>
            <div className="weather-item">
              <span className="label">Humidity:</span>
              <span className="value">
                {weather.humidity || weather.avg_humidity}%
              </span>
            </div>
            <div className="weather-item">
              <span className="label">Conditions:</span>
              <span className="value">{weather.conditions}</span>
            </div>
          </div>
        </div>
      )}

      {/* SGM Selections */}
      <div className="section">
        <h2>🎯 SGM Selections</h2>
        {selections.map((selection) => (
          <div key={selection.id} className="selection-row">
            <input
              type="text"
              placeholder="Player Name"
              value={selection.player}
              onChange={(e) => updateSelection(selection.id, 'player', e.target.value)}
              className="player-input"
            />
            <select
              value={selection.stat_type}
              onChange={(e) => updateSelection(selection.id, 'stat_type', e.target.value)}
              className="stat-select"
            >
              <option value="disposals">Disposals</option>
              <option value="goals">Goals</option>
              <option value="marks">Marks</option>
              <option value="tackles">Tackles</option>
            </select>
            <input
              type="number"
              placeholder="Threshold"
              value={selection.threshold}
              onChange={(e) => updateSelection(selection.id, 'threshold', e.target.value)}
              className="threshold-input"
              min="1"
              step="1"
            />
            <button
              onClick={() => removeSelection(selection.id)}
              className="remove-btn"
            >
              ❌
            </button>
          </div>
        ))}
        
        <div className="selection-actions">
          <button onClick={addSelection} className="add-selection-btn">
            ➕ Add Selection
          </button>
          
          {selections.length > 0 && (
            <button 
              onClick={analyzeSGM} 
              className="analyze-btn"
              disabled={loading}
            >
              {loading ? "🔄 Analyzing..." : "🔍 Analyze SGM"}
            </button>
          )}
        </div>
      </div>

      {/* SGM Analysis Results */}
      {sgmAnalysis && (
        <div className="section">
          <h2>📊 SGM Analysis Results</h2>
          
          <div className="analysis-summary">
            <div className="summary-item">
              <span className="label">Combined Probability:</span>
              <span className="value">
                {formatProbability(sgmAnalysis.sgm_analysis.combined_probability)}
              </span>
            </div>
            <div className="summary-item">
              <span className="label">Implied Odds:</span>
              <span className="value">${sgmAnalysis.sgm_analysis.implied_odds}</span>
            </div>
            <div className="summary-item">
              <span className="label">Recommendation:</span>
              <span 
                className="value recommendation"
                style={{ color: getRecommendationColor(sgmAnalysis.sgm_analysis.recommendation) }}
              >
                {sgmAnalysis.sgm_analysis.recommendation.toUpperCase()}
              </span>
            </div>
          </div>

          <div className="individual-predictions">
            <h3>Individual Predictions</h3>
            {sgmAnalysis.sgm_analysis.predictions.map((pred, index) => (
              <div key={index} className="prediction-item">
                <div className="prediction-header">
                  <span className="player-name">{pred.player}</span>
                  <span className="stat-description">
                    {pred.threshold}+ {pred.stat_type}
                  </span>
                </div>
                <div className="prediction-details">
                  <div className="detail">
                    <span className="detail-label">Probability:</span>
                    <span className="detail-value">{formatProbability(pred.probability)}</span>
                  </div>
                  <div className="detail">
                    <span className="detail-label">Weather Impact:</span>
                    <span className="detail-value">
                      {pred.weather_modifier > 1 ? '⬆️' : pred.weather_modifier < 1 ? '⬇️' : '➡️'}
                      {((pred.weather_modifier - 1) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="detail">
                    <span className="detail-label">Venue Impact:</span>
                    <span className="detail-value">
                      {pred.venue_modifier > 1 ? '⬆️' : pred.venue_modifier < 1 ? '⬇️' : '➡️'}
                      {((pred.venue_modifier - 1) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

function App() {
  return (
    <div className="App">
      <SGMBuilder />
    </div>
  );
}

export default App;
