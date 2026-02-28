import pandas as pd
import json
import csv
import os
import math
import random
from typing import Dict, Any


def load_match_data(csv_path: str) -> pd.DataFrame:
    """Load match data CSV into a pandas DataFrame.

    Args:
        csv_path: Path to the CSV file.
    Returns:
        DataFrame with the CSV contents.
    """
    return pd.read_csv(csv_path)


def load_champion_map(csv_path: str = "datasets/champions.csv") -> Dict[int, str]:
    """Load champion ID to Name mapping from CSV."""
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Using IDs.")
        return mapping
        
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mapping[int(row['id'])] = row['name']
            except (ValueError, KeyError):
                continue
    return mapping


# Load mapping once at module level to avoid reloading every time
CHAMPION_MAP = load_champion_map()


def get_champion_name(champ_id: Any) -> str:
    """Convert champion ID to Name using global map."""
    try:
        cid = int(champ_id)
        return CHAMPION_MAP.get(cid, str(champ_id))
    except (ValueError, TypeError):
        return str(champ_id)


def format_draft_state(row: pd.Series, style: str = "full") -> str:
    """Convert a DataFrame row into a textual draft description.

    The `style` parameter determines which part of the draft is included.
    Supported styles:
        - "full": all bans and picks in order.
        - "first_ban": only the first ban.
        - "first_pick": only the first pick.
        - "last_ban": only the last ban.
        - "last_pick": only the last pick.
        - "mixed": a random re‑ordering (for diversity).
    """
    # Extract bans (Team 0 then Team 1 for standard 10 bans structure, usually alternating)
    # The columns are team0Ban0ChampionId ... team0Ban4ChampionId and team1Ban0ChampionId ...
    ban_cols = [
        f"team{t}Ban{b}ChampionId" 
        for t in range(2) 
        for b in range(5)
    ]
    
    # Extract picks (Participant 0 to 9)
    pick_cols = [f"participant{i}ChampionId" for i in range(10)]
    
    bans = []
    for col in ban_cols:
        if col in row and pd.notna(row[col]):
            bans.append(get_champion_name(row[col]))
            
    picks = []
    for col in pick_cols:
        if col in row and pd.notna(row[col]):
            picks.append(get_champion_name(row[col]))

    if style == "random":
        # Define the strict 17 steps order from prompts.py
        # ID, Type, Description, Column(s)
        steps = [
            (1, "ban", "Étape 1 : Ban de l’équipe Bleue", ["team0Ban0ChampionId"]),
            (2, "ban", "Étape 2 : Ban de l’équipe Rouge", ["team1Ban0ChampionId"]),
            (3, "ban", "Étape 3 : Ban de l’équipe Bleue", ["team0Ban1ChampionId"]),
            (4, "ban", "Étape 4 : Ban de l’équipe Rouge", ["team1Ban1ChampionId"]),
            (5, "ban", "Étape 5 : Ban de l’équipe Bleue", ["team0Ban2ChampionId"]),
            (6, "ban", "Étape 6 : Ban de l’équipe Rouge", ["team1Ban2ChampionId"]),
            (7, "pick", "Étape 7 : Pick de l’invocateur 1 de l’équipe Bleue", ["participant0ChampionId"]),
            (8, "pick", "Étape 8 : Picks simultanés des invocateurs 1 et 2 de l’équipe Rouge", ["participant5ChampionId", "participant6ChampionId"]),
            (9, "pick", "Étape 9 : Picks des invocateurs 2 et 3 de l’équipe Bleue", ["participant1ChampionId", "participant2ChampionId"]),
            (10, "pick", "Étape 10 : Pick de l’invocateur 3 de l’équipe Rouge", ["participant7ChampionId"]),
            (11, "ban", "Étape 11 : Ban de l'équipe Bleue", ["team0Ban3ChampionId"]),
            (12, "ban", "Étape 12 : Ban de l'équipe Rouge", ["team1Ban3ChampionId"]),
            (13, "ban", "Étape 13 : Ban de l'équipe Bleue", ["team0Ban4ChampionId"]),
            (14, "ban", "Étape 14 : Ban de l'équipe Rouge", ["team1Ban4ChampionId"]),
            (15, "pick", "Étape 15 : Pick de l’invocateur 4 de l’équipe Bleue", ["participant3ChampionId"]),
            (16, "pick", "Étape 16 : Picks des invocateurs 4 et 5 de l’équipe Rouge", ["participant8ChampionId", "participant9ChampionId"]),
            (17, "pick", "Étape 17 : Pick de l’invocateur 5 de l’équipe Bleue", ["participant4ChampionId"]),
        ]

        # Pick a strictly random step
        step_id, step_type, step_desc, target_cols = random.choice(steps)
        
        # Build history/context up to this step (exclusive)
        # We need to construct the lists of bans AND picks that happened BEFORE `step_id`.
        
        current_bans_blue = []
        current_bans_red = []
        current_picks_blue = []
        current_picks_red = []
        
        # Helper to safely get champ name
        def get_name(col_name):
            if col_name in row and pd.notna(row[col_name]):
                return get_champion_name(row[col_name])
            return "Unknown"

        # Iterate through steps 1 to step_id - 1
        for s_id, s_type, s_desc, s_cols in steps:
            if s_id >= step_id:
                break
                
            # Process actions for this previous step s_id
            for col in s_cols:
                c_name = get_name(col)
                # Identify if Blue or Red based on step definition or column name
                # Bans: team0 = Blue, team1 = Red
                # Picks: participant 0-4 = Blue, 5-9 = Red
                
                if "team0" in col:
                    current_bans_blue.append(c_name)
                elif "team1" in col:
                    current_bans_red.append(c_name)
                elif "participant" in col:
                    p_idx = int(col.replace("participant", "").replace("ChampionId", ""))
                    if 0 <= p_idx <= 4:
                        current_picks_blue.append(c_name)
                    else:
                        current_picks_red.append(c_name)
        
        # Format the output string
        # "État actuel de la draft : \nBans : ... \nPicks Bleue : ... \nPicks Rouge : ... \nAction à prédire : ..."
        
        # Note: User example showed "Bans : [...]" (merged?). 
        # But prompts.py says strict order. 
        # User example:
        # "Bans : ['Darius', ...]" -> merged list of all bans so far.
        # "Picks Bleue : [...]"
        # "Picks Rouge : [...]"
        
        all_bans = current_bans_blue + current_bans_red
        # We might want to keep order of appearance? 
        # The simple append above groups by team if we iterated steps.
        # Since we iterated steps 1..N, the order in all_bans is chronological! 
        # (e.g. Blue Ban 1, Red Ban 1, Blue Ban 2...)
        
        # Use simple string representation of lists
        state_str = (
            f"État actuel de la draft :\n"
            f"Bans : {all_bans}\n"
            f"Picks Bleue : {current_picks_blue}\n"
            f"Picks Rouge : {current_picks_red}\n"
            f"Action à réaliser : {step_desc}"
        )
        
        return state_str

        
        return state_str

    if style == "full":
        # Full draft = All bans + All picks
        # We can reuse the same logic but for all steps.
        # But wait, full draft implies the draft is OVER? Or we want to predict something?
        # Usually "full" means "Context: Everything that happened". 
        # If the instruction is "Action demandée : analyse_fin_de_match" or similar?
        # User prompt in main.py was "Action demandée : full_draft". 
        # Let's assume we want to show everything.
        
        current_bans_blue = []
        current_bans_red = []
        current_picks_blue = []
        current_picks_red = []
        
        def get_name(col_name):
            if col_name in row and pd.notna(row[col_name]):
                return get_champion_name(row[col_name])
            return "Unknown"

        # Extract all bans
        ban_cols = [f"team{t}Ban{b}ChampionId" for t in range(2) for b in range(5)]
        # We need to respect the order? Or just group by team?
        # The prompt says: "Bans: [...]" -> A single list?
        # In random style I did `all_bans = current_bans_blue + current_bans_red`.
        # Taking all bans from columns.
        
        all_bans_ordered = []
        # Let's try to extract in order of steps to be consistent?
        # Steps 1-6, 11-14.
        # But simpler: just iterate all ban columns sorted by some logic?
        # The standard format seems to be just list of bans.
        # Let's stick to the list of strings format.
        
        for col in ban_cols:
             c_name = get_name(col)
             if c_name != "Unknown":
                 if "team0" in col: current_bans_blue.append(c_name)
                 else: current_bans_red.append(c_name)
        
        # Extract all picks
        pick_cols = [f"participant{i}ChampionId" for i in range(10)]
        for col in pick_cols:
             c_name = get_name(col)
             if c_name != "Unknown":
                 p_idx = int(col.replace("participant", "").replace("ChampionId", ""))
                 if 0 <= p_idx <= 4: current_picks_blue.append(c_name)
                 else: current_picks_red.append(c_name)
                 
        all_bans = current_bans_blue + current_bans_red
        
        state_str = (
            f"État actuel de la draft :\n"
            f"Bans : {all_bans}\n"
            f"Picks Bleue : {current_picks_blue}\n"
            f"Picks Rouge : {current_picks_red}\n"
            f"Action à réaliser : Analyse de cette draft complète"
        )
        return state_str
    
    # Fallback to full
    return format_draft_state(row, style="full")


def save_synthetic_row(csv_path: str, draft_state: str, response_json: Dict[str, Any], confidence: float) -> None:
    """Append a synthetic row to the CSV file.

    The CSV will have columns: `draft_state`, `response_json`, `confidence`.
    """
    row = {
        "draft_state": draft_state,
        "response_json": json.dumps(response_json, ensure_ascii=False),
        "confidence": confidence,
    }
    try:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def confidence_score(logprobs_content) -> float:
    """
    Convertit les logprobs en score 0–100
    """
    if not logprobs_content:
        return 0.0

    avg_logprob = sum(t.logprob for t in logprobs_content) / len(logprobs_content)

    # exp(logprob) = prob
    avg_prob = math.exp(avg_logprob)

    return round(avg_prob * 100, 2)
