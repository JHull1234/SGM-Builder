# Enhanced AFL Player Database
# This will be integrated into the main application

COMPREHENSIVE_AFL_PLAYERS = [
    # ADELAIDE CROWS
    {
        "id": "player_001",
        "name": "Rory Laird",
        "team": "Adelaide",
        "position": "Midfielder",
        "avg_disposals": 28.5,
        "avg_goals": 0.4,
        "avg_marks": 5.2,
        "avg_tackles": 4.8,
        "avg_kicks": 18.3,
        "avg_handballs": 10.2,
        "games_played": 22,
        "injury_history": ["hamstring_2024"],
        "venue_performance": {
            "Adelaide Oval": {"disposals": 31.2, "goals": 0.6},
            "MCG": {"disposals": 26.1, "goals": 0.2}
        }
    },
    {
        "id": "player_002", 
        "name": "Jordan Dawson",
        "team": "Adelaide",
        "position": "Midfielder/Defender",
        "avg_disposals": 25.8,
        "avg_goals": 0.8,
        "avg_marks": 6.1,
        "avg_tackles": 3.9,
        "avg_kicks": 17.2,
        "avg_handballs": 8.6,
        "games_played": 20,
        "injury_history": [],
        "venue_performance": {
            "Adelaide Oval": {"disposals": 28.3, "goals": 1.1},
            "MCG": {"disposals": 23.4, "goals": 0.5}
        }
    },
    
    # BRISBANE LIONS
    {
        "id": "player_003",
        "name": "Lachie Neale",
        "team": "Brisbane",
        "position": "Midfielder",
        "avg_disposals": 32.1,
        "avg_goals": 1.2,
        "avg_marks": 4.8,
        "avg_tackles": 6.2,
        "avg_kicks": 19.8,
        "avg_handballs": 12.3,
        "games_played": 21,
        "injury_history": ["ankle_2023"],
        "venue_performance": {
            "Gabba": {"disposals": 35.4, "goals": 1.6},
            "MCG": {"disposals": 29.8, "goals": 0.9}
        }
    },
    {
        "id": "player_004",
        "name": "Hugh McCluggage",
        "team": "Brisbane", 
        "position": "Midfielder",
        "avg_disposals": 26.4,
        "avg_goals": 0.9,
        "avg_marks": 5.5,
        "avg_tackles": 4.1,
        "avg_kicks": 16.7,
        "avg_handballs": 9.7,
        "games_played": 22,
        "injury_history": [],
        "venue_performance": {
            "Gabba": {"disposals": 29.1, "goals": 1.2},
            "MCG": {"disposals": 24.2, "goals": 0.7}
        }
    },
    
    # CARLTON BLUES
    {
        "id": "player_005",
        "name": "Patrick Cripps",
        "team": "Carlton",
        "position": "Midfielder",
        "avg_disposals": 30.8,
        "avg_goals": 1.1,
        "avg_marks": 6.2,
        "avg_tackles": 5.9,
        "avg_kicks": 18.9,
        "avg_handballs": 11.9,
        "games_played": 23,
        "injury_history": ["back_2022"],
        "venue_performance": {
            "MCG": {"disposals": 33.2, "goals": 1.4},
            "Marvel Stadium": {"disposals": 29.1, "goals": 0.9}
        }
    },
    {
        "id": "player_006",
        "name": "Sam Walsh",
        "team": "Carlton",
        "position": "Midfielder",
        "avg_disposals": 28.9,
        "avg_goals": 0.6,
        "avg_marks": 4.9,
        "avg_tackles": 5.8,
        "avg_kicks": 17.2,
        "avg_handballs": 11.7,
        "games_played": 24,
        "injury_history": [],
        "venue_performance": {
            "MCG": {"disposals": 31.1, "goals": 0.8},
            "Marvel Stadium": {"disposals": 27.3, "goals": 0.4}
        }
    },
    
    # COLLINGWOOD MAGPIES
    {
        "id": "player_007",
        "name": "Nick Daicos",
        "team": "Collingwood",
        "position": "Midfielder",
        "avg_disposals": 31.5,
        "avg_goals": 0.8,
        "avg_marks": 5.8,
        "avg_tackles": 4.2,
        "avg_kicks": 20.1,
        "avg_handballs": 11.4,
        "games_played": 22,
        "injury_history": [],
        "venue_performance": {
            "MCG": {"disposals": 34.8, "goals": 1.1},
            "Marvel Stadium": {"disposals": 28.9, "goals": 0.6}
        }
    },
    {
        "id": "player_008",
        "name": "Scott Pendlebury",
        "team": "Collingwood",
        "position": "Midfielder",
        "avg_disposals": 25.6,
        "avg_goals": 0.7,
        "avg_marks": 6.1,
        "avg_tackles": 3.4,
        "avg_kicks": 16.8,
        "avg_handballs": 8.8,
        "games_played": 20,
        "injury_history": ["calf_2024"],
        "venue_performance": {
            "MCG": {"disposals": 27.9, "goals": 0.9},
            "Marvel Stadium": {"disposals": 23.8, "goals": 0.5}
        }
    },
    
    # ESSENDON BOMBERS
    {
        "id": "player_009",
        "name": "Zach Merrett",
        "team": "Essendon", 
        "position": "Midfielder",
        "avg_disposals": 29.7,
        "avg_goals": 0.9,
        "avg_marks": 5.4,
        "avg_tackles": 5.1,
        "avg_kicks": 18.6,
        "avg_handballs": 11.1,
        "games_played": 21,
        "injury_history": [],
        "venue_performance": {
            "MCG": {"disposals": 32.1, "goals": 1.2},
            "Marvel Stadium": {"disposals": 27.8, "goals": 0.7}
        }
    },
    {
        "id": "player_010",
        "name": "Darcy Parish",
        "team": "Essendon",
        "position": "Midfielder", 
        "avg_disposals": 27.8,
        "avg_goals": 0.5,
        "avg_marks": 4.6,
        "avg_tackles": 4.9,
        "avg_kicks": 16.2,
        "avg_handballs": 11.6,
        "games_played": 23,
        "injury_history": ["hamstring_2023"],
        "venue_performance": {
            "MCG": {"disposals": 30.2, "goals": 0.7},
            "Marvel Stadium": {"disposals": 26.1, "goals": 0.4}
        }
    },
    
    # FREMANTLE DOCKERS
    {
        "id": "player_011",
        "name": "Caleb Serong",
        "team": "Fremantle",
        "position": "Midfielder",
        "avg_disposals": 28.9,
        "avg_goals": 0.6,
        "avg_marks": 4.8,
        "avg_tackles": 6.8,
        "avg_kicks": 17.1,
        "avg_handballs": 11.8,
        "games_played": 22,
        "injury_history": [],
        "venue_performance": {
            "Optus Stadium": {"disposals": 32.4, "goals": 0.9},
            "MCG": {"disposals": 26.2, "goals": 0.4}
        }
    },
    {
        "id": "player_012",
        "name": "Andrew Brayshaw",
        "team": "Fremantle",
        "position": "Midfielder",
        "avg_disposals": 26.8,
        "avg_goals": 0.8,
        "avg_marks": 5.2,
        "avg_tackles": 5.4,
        "avg_kicks": 16.9,
        "avg_handballs": 9.9,
        "games_played": 20,
        "injury_history": ["concussion_2022"],
        "venue_performance": {
            "Optus Stadium": {"disposals": 29.6, "goals": 1.1},
            "MCG": {"disposals": 24.8, "goals": 0.6}
        }
    },
    
    # GEELONG CATS
    {
        "id": "player_013",
        "name": "Jeremy Cameron",
        "team": "Geelong",
        "position": "Forward",
        "avg_disposals": 12.4,
        "avg_goals": 2.8,
        "avg_marks": 7.9,
        "avg_tackles": 2.1,
        "avg_kicks": 9.8,
        "avg_handballs": 2.6,
        "games_played": 21,
        "injury_history": ["hamstring_2024"],
        "venue_performance": {
            "GMHBA Stadium": {"disposals": 14.1, "goals": 3.4},
            "MCG": {"disposals": 11.2, "goals": 2.3}
        }
    },
    {
        "id": "player_014",
        "name": "Tom Hawkins",
        "team": "Geelong",
        "position": "Forward",
        "avg_disposals": 10.2,
        "avg_goals": 2.3,
        "avg_marks": 8.5,
        "avg_tackles": 1.8,
        "avg_kicks": 7.9,
        "avg_handballs": 2.3,
        "games_played": 19,
        "injury_history": ["foot_2023"],
        "venue_performance": {
            "GMHBA Stadium": {"disposals": 11.8, "goals": 2.8},
            "MCG": {"disposals": 9.1, "goals": 1.9}
        }
    },
    
    # GOLD COAST SUNS
    {
        "id": "player_015",
        "name": "Touk Miller",
        "team": "Gold Coast",
        "position": "Midfielder",
        "avg_disposals": 31.2,
        "avg_goals": 0.7,
        "avg_marks": 5.1,
        "avg_tackles": 6.9,
        "avg_kicks": 18.4,
        "avg_handballs": 12.8,
        "games_played": 24,
        "injury_history": [],
        "venue_performance": {
            "Carrara Stadium": {"disposals": 34.6, "goals": 1.0},
            "MCG": {"disposals": 28.8, "goals": 0.5}
        }
    },
    
    # GWS GIANTS
    {
        "id": "player_016",
        "name": "Josh Kelly",
        "team": "GWS",
        "position": "Midfielder",
        "avg_disposals": 27.5,
        "avg_goals": 1.1,
        "avg_marks": 5.8,
        "avg_tackles": 4.2,
        "avg_kicks": 17.9,
        "avg_handballs": 9.6,
        "games_played": 18,
        "injury_history": ["calf_2024", "back_2023"],
        "venue_performance": {
            "ANZ Stadium": {"disposals": 30.2, "goals": 1.4},
            "MCG": {"disposals": 25.4, "goals": 0.9}
        }
    },
    
    # HAWTHORN HAWKS
    {
        "id": "player_017",
        "name": "James Sicily",
        "team": "Hawthorn",
        "position": "Defender",
        "avg_disposals": 22.1,
        "avg_goals": 0.8,
        "avg_marks": 9.2,
        "avg_tackles": 3.6,
        "avg_kicks": 16.8,
        "avg_handballs": 5.3,
        "games_played": 22,
        "injury_history": ["ACL_2022"],
        "venue_performance": {
            "MCG": {"disposals": 24.3, "goals": 1.0},
            "Marvel Stadium": {"disposals": 20.4, "goals": 0.6}
        }
    },
    
    # MELBOURNE DEMONS
    {
        "id": "player_018",
        "name": "Clayton Oliver",
        "team": "Melbourne",
        "position": "Midfielder",
        "avg_disposals": 32.5,
        "avg_goals": 0.8,
        "avg_marks": 4.2,
        "avg_tackles": 6.8,
        "avg_kicks": 17.9,
        "avg_handballs": 14.6,
        "games_played": 20,
        "injury_history": ["seizure_2024"],
        "venue_performance": {
            "MCG": {"disposals": 36.8, "goals": 1.1},
            "Marvel Stadium": {"disposals": 29.2, "goals": 0.6}
        }
    },
    {
        "id": "player_019",
        "name": "Christian Petracca",
        "team": "Melbourne", 
        "position": "Midfielder",
        "avg_disposals": 28.3,
        "avg_goals": 1.2,
        "avg_marks": 5.1,
        "avg_tackles": 4.9,
        "avg_kicks": 16.8,
        "avg_handballs": 11.5,
        "games_played": 18,
        "injury_history": ["ribs_2024"],
        "venue_performance": {
            "MCG": {"disposals": 31.6, "goals": 1.6},
            "Marvel Stadium": {"disposals": 25.8, "goals": 0.9}
        }
    },
    
    # NORTH MELBOURNE KANGAROOS
    {
        "id": "player_020",
        "name": "Luke Davies-Uniacke",
        "team": "North Melbourne",
        "position": "Midfielder",
        "avg_disposals": 25.8,
        "avg_goals": 0.9,
        "avg_marks": 4.9,
        "avg_tackles": 5.6,
        "avg_kicks": 15.2,
        "avg_handballs": 10.6,
        "games_played": 21,
        "injury_history": [],
        "venue_performance": {
            "Marvel Stadium": {"disposals": 28.1, "goals": 1.2},
            "MCG": {"disposals": 24.2, "goals": 0.7}
        }
    },
    
    # PORT ADELAIDE POWER
    {
        "id": "player_021",
        "name": "Zak Butters",
        "team": "Port Adelaide",
        "position": "Midfielder",
        "avg_disposals": 26.9,
        "avg_goals": 1.3,
        "avg_marks": 5.4,
        "avg_tackles": 4.8,
        "avg_kicks": 16.1,
        "avg_handballs": 10.8,
        "games_played": 23,
        "injury_history": [],
        "venue_performance": {
            "Adelaide Oval": {"disposals": 29.8, "goals": 1.7},
            "MCG": {"disposals": 24.6, "goals": 1.0}
        }
    },
    
    # RICHMOND TIGERS
    {
        "id": "player_022",
        "name": "Tim Taranto",
        "team": "Richmond",
        "position": "Midfielder",
        "avg_disposals": 28.6,
        "avg_goals": 0.7,
        "avg_marks": 5.2,
        "avg_tackles": 6.1,
        "avg_kicks": 16.9,
        "avg_handballs": 11.7,
        "games_played": 22,
        "injury_history": [],
        "venue_performance": {
            "MCG": {"disposals": 31.4, "goals": 0.9},
            "Marvel Stadium": {"disposals": 26.8, "goals": 0.5}
        }
    },
    
    # ST KILDA SAINTS
    {
        "id": "player_023",
        "name": "Jack Steele",
        "team": "St Kilda",
        "position": "Midfielder",
        "avg_disposals": 27.2,
        "avg_goals": 0.8,
        "avg_marks": 5.6,
        "avg_tackles": 7.4,
        "avg_kicks": 16.1,
        "avg_handballs": 11.1,
        "games_played": 19,
        "injury_history": ["knee_2024"],
        "venue_performance": {
            "Marvel Stadium": {"disposals": 29.8, "goals": 1.1},
            "MCG": {"disposals": 25.4, "goals": 0.6}
        }
    },
    
    # SYDNEY SWANS
    {
        "id": "player_024",
        "name": "Isaac Heeney",
        "team": "Sydney",
        "position": "Forward/Midfielder",
        "avg_disposals": 21.4,
        "avg_goals": 2.1,
        "avg_marks": 6.8,
        "avg_tackles": 3.9,
        "avg_kicks": 14.2,
        "avg_handballs": 7.2,
        "games_played": 22,
        "injury_history": [],
        "venue_performance": {
            "SCG": {"disposals": 24.1, "goals": 2.6},
            "MCG": {"disposals": 19.2, "goals": 1.7}
        }
    },
    {
        "id": "player_025",
        "name": "Errol Gulden",
        "team": "Sydney",
        "position": "Midfielder",
        "avg_disposals": 25.9,
        "avg_goals": 0.9,
        "avg_marks": 4.8,
        "avg_tackles": 5.2,
        "avg_kicks": 15.7,
        "avg_handballs": 10.2,
        "games_played": 24,
        "injury_history": [],
        "venue_performance": {
            "SCG": {"disposals": 28.6, "goals": 1.2},
            "MCG": {"disposals": 23.8, "goals": 0.7}
        }
    },
    
    # WEST COAST EAGLES
    {
        "id": "player_026",
        "name": "Tim Kelly",
        "team": "West Coast",
        "position": "Midfielder",
        "avg_disposals": 24.8,
        "avg_goals": 1.0,
        "avg_marks": 5.1,
        "avg_tackles": 4.6,
        "avg_kicks": 15.2,
        "avg_handballs": 9.6,
        "games_played": 20,
        "injury_history": ["calf_2023"],
        "venue_performance": {
            "Optus Stadium": {"disposals": 27.9, "goals": 1.3},
            "MCG": {"disposals": 22.4, "goals": 0.8}
        }
    },
    
    # WESTERN BULLDOGS
    {
        "id": "player_027",
        "name": "Marcus Bontempelli",
        "team": "Western Bulldogs",
        "position": "Midfielder",
        "avg_disposals": 29.7,
        "avg_goals": 1.1,
        "avg_marks": 6.3,
        "avg_tackles": 5.2,
        "avg_kicks": 18.9,
        "avg_handballs": 10.8,
        "games_played": 22,
        "injury_history": [],
        "venue_performance": {
            "Marvel Stadium": {"disposals": 32.4, "goals": 1.4},
            "MCG": {"disposals": 27.6, "goals": 0.9}
        }
    },
    {
        "id": "player_028",
        "name": "Adam Treloar",
        "team": "Western Bulldogs",
        "position": "Midfielder", 
        "avg_disposals": 26.8,
        "avg_goals": 0.6,
        "avg_marks": 4.9,
        "avg_tackles": 5.8,
        "avg_kicks": 15.9,
        "avg_handballs": 10.9,
        "games_played": 21,
        "injury_history": ["back_2023"],
        "venue_performance": {
            "Marvel Stadium": {"disposals": 29.1, "goals": 0.8},
            "MCG": {"disposals": 25.2, "goals": 0.5}
        }
    }
]

# Team defensive statistics for matchup analysis
TEAM_DEFENSIVE_STATS = {
    "Adelaide": {
        "disposals_allowed_per_game": 385.2,
        "goals_allowed_per_game": 14.8,
        "marks_allowed_per_game": 78.4,
        "tackles_per_game": 65.2,
        "midfielder_disposals_allowed": 125.8,
        "forward_goals_allowed": 8.9
    },
    "Brisbane": {
        "disposals_allowed_per_game": 378.9,
        "goals_allowed_per_game": 13.6,
        "marks_allowed_per_game": 81.2,
        "tackles_per_game": 68.1,
        "midfielder_disposals_allowed": 121.4,
        "forward_goals_allowed": 7.8
    },
    "Carlton": {
        "disposals_allowed_per_game": 372.1,
        "goals_allowed_per_game": 12.9,
        "marks_allowed_per_game": 76.8,
        "tackles_per_game": 71.4,
        "midfielder_disposals_allowed": 118.6,
        "forward_goals_allowed": 7.2
    },
    "Collingwood": {
        "disposals_allowed_per_game": 381.7,
        "goals_allowed_per_game": 13.8,
        "marks_allowed_per_game": 79.3,
        "tackles_per_game": 69.7,
        "midfielder_disposals_allowed": 123.2,
        "forward_goals_allowed": 8.1
    },
    "Essendon": {
        "disposals_allowed_per_game": 392.4,
        "goals_allowed_per_game": 15.2,
        "marks_allowed_per_game": 82.1,
        "tackles_per_game": 62.8,
        "midfielder_disposals_allowed": 129.7,
        "forward_goals_allowed": 9.6
    },
    "Fremantle": {
        "disposals_allowed_per_game": 365.8,
        "goals_allowed_per_game": 11.4,
        "marks_allowed_per_game": 74.2,
        "tackles_per_game": 74.9,
        "midfielder_disposals_allowed": 115.3,
        "forward_goals_allowed": 6.4
    },
    "Geelong": {
        "disposals_allowed_per_game": 376.5,
        "goals_allowed_per_game": 13.1,
        "marks_allowed_per_game": 77.6,
        "tackles_per_game": 67.3,
        "midfielder_disposals_allowed": 119.8,
        "forward_goals_allowed": 7.5
    },
    "Gold Coast": {
        "disposals_allowed_per_game": 398.2,
        "goals_allowed_per_game": 16.8,
        "marks_allowed_per_game": 84.7,
        "tackles_per_game": 59.4,
        "midfielder_disposals_allowed": 134.1,
        "forward_goals_allowed": 10.9
    },
    "GWS": {
        "disposals_allowed_per_game": 384.6,
        "goals_allowed_per_game": 14.3,
        "marks_allowed_per_game": 80.5,
        "tackles_per_game": 66.8,
        "midfielder_disposals_allowed": 126.4,
        "forward_goals_allowed": 8.7
    },
    "Hawthorn": {
        "disposals_allowed_per_game": 388.9,
        "goals_allowed_per_game": 15.6,
        "marks_allowed_per_game": 81.8,
        "tackles_per_game": 63.2,
        "midfielder_disposals_allowed": 128.3,
        "forward_goals_allowed": 9.2
    },
    "Melbourne": {
        "disposals_allowed_per_game": 369.3,
        "goals_allowed_per_game": 12.2,
        "marks_allowed_per_game": 75.9,
        "tackles_per_game": 72.6,
        "midfielder_disposals_allowed": 116.7,
        "forward_goals_allowed": 6.8
    },
    "North Melbourne": {
        "disposals_allowed_per_game": 402.7,
        "goals_allowed_per_game": 18.1,
        "marks_allowed_per_game": 87.3,
        "tackles_per_game": 56.9,
        "midfielder_disposals_allowed": 138.4,
        "forward_goals_allowed": 12.3
    },
    "Port Adelaide": {
        "disposals_allowed_per_game": 374.8,
        "goals_allowed_per_game": 13.4,
        "marks_allowed_per_game": 78.1,
        "tackles_per_game": 69.5,
        "midfielder_disposals_allowed": 120.6,
        "forward_goals_allowed": 7.9
    },
    "Richmond": {
        "disposals_allowed_per_game": 395.1,
        "goals_allowed_per_game": 16.2,
        "marks_allowed_per_game": 83.4,
        "tackles_per_game": 61.7,
        "midfielder_disposals_allowed": 131.8,
        "forward_goals_allowed": 10.1
    },
    "St Kilda": {
        "disposals_allowed_per_game": 387.6,
        "goals_allowed_per_game": 14.9,
        "marks_allowed_per_game": 80.9,
        "tackles_per_game": 64.3,
        "midfielder_disposals_allowed": 127.1,
        "forward_goals_allowed": 8.8
    },
    "Sydney": {
        "disposals_allowed_per_game": 368.7,
        "goals_allowed_per_game": 11.8,
        "marks_allowed_per_game": 74.6,
        "tackles_per_game": 73.8,
        "midfielder_disposals_allowed": 114.9,
        "forward_goals_allowed": 6.1
    },
    "West Coast": {
        "disposals_allowed_per_game": 405.3,
        "goals_allowed_per_game": 19.4,
        "marks_allowed_per_game": 89.1,
        "tackles_per_game": 54.2,
        "midfielder_disposals_allowed": 141.7,
        "forward_goals_allowed": 13.6
    },
    "Western Bulldogs": {
        "disposals_allowed_per_game": 379.4,
        "goals_allowed_per_game": 14.1,
        "marks_allowed_per_game": 79.7,
        "tackles_per_game": 67.9,
        "midfielder_disposals_allowed": 122.3,
        "forward_goals_allowed": 8.4
    }
}