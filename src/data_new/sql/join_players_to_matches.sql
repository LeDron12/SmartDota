WITH pub_matches AS (
    SELECT match_id
    FROM public_matches
    WHERE start_time >= 1743800400 and start_time <= 1746392400
        AND lobby_type = 0  -- 0 = public match
        AND game_mode = 22  -- 22 = All Pick
        AND avg_rank_tier >= 60  -- approximately Divine rank and above
)
SELECT 
    pm.match_id,
    array_agg(
        json_build_object(
            'account_id', pm.account_id,
            'hero_id', pm.hero_id,
            'player_slot', pm.player_slot
        )
    ) as players
FROM player_matches pm
INNER JOIN pub_matches pub ON pm.match_id = pub.match_id
GROUP BY pm.match_id