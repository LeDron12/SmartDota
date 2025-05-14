SELECT match_id,
        radiant_win,
        start_time,
        radiant_team,
        dire_team
FROM public_matches
WHERE start_time >= {start_time_start} and start_time <= {start_time_end}
    AND lobby_type = 0  -- 0 = public match
    AND game_mode = 22  -- 22 = All Pick
    AND avg_rank_tier >= {rank_lower_bound} and avg_rank_tier <= {rank_upper_bound}
ORDER BY start_time DESC
LIMIT {matches_limit}