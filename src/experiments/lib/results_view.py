import requests
import json

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

from typing import List, Dict, Any


def make_single_prediction(model: Any, # Cbt or Sklearn model
                           match_id: int
                           ) -> Dict[str, str]:

    match_data = requests.get('https://api.opendota.com/api/matches/' + str(match_id)).json()

    pb = match_data['picks_bans']
    pb = sorted(filter(lambda x: x['is_pick'], pb), key=lambda x: x['team'])
    assert len(pb) == 10

    radiant_picks = [e['hero_id'] for e in pb[:5]]
    dire_picks = [e['hero_id'] for e in pb[5:]]

    with open('/Users/ankamenskiy/SmartDota/data/heroes.json', 'r') as f:
        existing_ids = sorted([h['id'] for h in json.loads(f.read())])

    existing_ids = \
    heroes = np.array(
                [1 if id in radiant_picks else 0 for id in existing_ids] \
                + \
                [1 if id in dire_picks else 0 for id in existing_ids], 
                dtype=np.int32).reshape(-1, 2*len(existing_ids)
            )

    for i, id in enumerate(heroes):
        if i + 1 in radiant_picks:
            assert i + 1 == id
        if i + 1 in dire_picks:
            assert i + 1 == id

    X_pred = heroes
    pred = model.predict(X_pred)
    probas = model.predict_proba(X_pred)
    probas = np.exp(probas) / np.sum(np.exp(probas)) # softmax

    return {
        'result': 'Radiant' if pred[0] == 1 else 'Dire',
        'dire': f'{probas[:, 0][0]:.2f}',
        'radiant': f'{probas[:, 1][0]:.2f}'
    }


def per_hero_metrics(radiant_heroes: List[List[int]], 
                     dire_heroes: List[List[int]], 
                     y_pred: List[int], 
                     y_true: List[int]
                     ) -> pd.DataFrame:

    def squeeze_ohe(heroes):
        return list(filter(lambda x: x != 0, [i + 1 if hero == 1 else 0 for i, hero in enumerate(heroes)]))

    with open('/Users/ankamenskiy/SmartDota/data/heroes.json', 'r') as f:
        all_heroes = json.loads(f.read())

    if len(radiant_heroes) == len(all_heroes):
        radiant_heroes = squeeze_ohe(radiant_heroes)
    if len(dire_heroes) == len(all_heroes):
        dire_heroes = squeeze_ohe(dire_heroes)

    assert len(radiant_heroes) == len(dire_heroes) == 5

    id_to_all_heroes = {}
    for hero in all_heroes:
        hero['preds'], hero['true'] = [], []
        id_to_all_heroes[hero['id']] = hero
    
    for r_h, d_h, y_p, y_t in zip(radiant_heroes, dire_heroes, y_pred, y_true):
        for h in r_h + d_h:
            id_to_all_heroes[h]['preds'].append(y_p)
            id_to_all_heroes[h]['true'].append(y_t)

    df = pd.DataFrame(columns=['hero_name', 'hero_id', 'metrics'])
    for k, v in id_to_all_heroes.items():
        row = {
            'hero_id': k,
            'hero_name': v['localized_name'],
            'metrics': classification_report(v['ture'], v['preds'], output_dict=True)
        }
        df = pd.concat([df, row], axis=0).reset_index(drop=True)

    df['sort_column'] = df['metrics'].apply(lambda x: np.mean(x[0]['f1-score'] + x[1]['f1-score'])[0])
    df.sort_values('sort_column', axis=0, ascending=True, inplace=True)

    return df
