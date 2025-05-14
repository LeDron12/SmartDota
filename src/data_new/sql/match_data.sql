SELECT
    match_id,
    radiant_win,
    start_time,
    picks_bans,
    radiant_team_id,
    dire_team_id,
    objectives,
    radiant_gold_adv,
    radiant_xp_adv,
    teamfights
FROM matches
WHERE 
    CASE 
        WHEN {match_id} IS NOT NULL THEN match_id = {match_id}
        ELSE start_time >= {start_time_start}
    END