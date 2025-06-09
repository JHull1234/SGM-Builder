# AFL AI Model Improvements Based on Melbourne vs Collingwood Results
# Learning from 0/3 prediction failure

LESSONS_LEARNED = {
    "recent_form_analysis": {
        "issue": "Mock data showed Oliver as 'hot' when he's actually been quiet",
        "fix": "Need real recent game data, not mock historical trends",
        "impact": "High - wrong form assessment leads to wrong predictions"
    },
    
    "teammate_synergy": {
        "issue": "Overweighted positive correlation between Oliver/Petracca",
        "fix": "Reduce teammate bonuses from +0.05 to +0.01",
        "impact": "Medium - synergy less important than individual variance"
    },
    
    "target_line_setting": {
        "issue": "Lines too optimistic (Daicos 25.5 vs actual 19)",
        "fix": "Add more conservative buffer and higher variance modeling",
        "impact": "High - need more realistic target expectations"
    },
    
    "individual_variance": {
        "issue": "Didn't account for high game-to-game variance (Petracca 0 goals)",
        "fix": "Increase standard deviations in probability calculations",
        "impact": "Critical - individual games have massive variance"
    },
    
    "weather_impact": {
        "issue": "May have underestimated cold weather impact on skill execution",
        "fix": "Increase penalties for temperatures below 12°C",
        "impact": "Low-Medium - weather factors may be more significant"
    }
}

IMPROVED_CORRELATION_MODEL = {
    "same_player_penalty": 0.20,  # Increased from 0.15
    "teammate_bonus": 0.01,       # Reduced from 0.05
    "opponent_correlation": -0.02, # Reduced negative correlation
    "position_correlation": 0.95   # Same position still correlated but less impact
}

IMPROVED_VARIANCE_MODEL = {
    "disposals_std": 6.0,  # Increased from 4.0 - higher variance
    "goals_std": 1.5,      # Increased from 1.0 - much higher variance  
    "marks_std": 2.5,      # Increased from 1.5
    "tackles_std": 2.0     # Increased from 1.2
}

CONSERVATIVE_LINE_ADJUSTMENT = {
    "buffer_factor": 0.85,  # Set lines at 85% of prediction instead of 95%
    "confidence_threshold": 0.65,  # Require 65% confidence minimum
    "variance_multiplier": 1.3    # Account for higher than expected variance
}