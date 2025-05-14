import requests
import json
from typing import Dict, Any
import time
import argparse
from pathlib import Path

def load_query_and_params(params_file: str, full_path: bool) -> Dict[str, Any]:
    """Load query parameters from a JSON file."""
    prefix = "../data_new/sql" if full_path else "sql"
    with open(f"{prefix}/{params_file}_cfg.json", 'r') as f:
        params = json.load(f)
    with open(f"{prefix}/{params_file}.sql", 'r') as f:
        sql_template = f.read().strip()

    return sql_template, params

def fetch_opendota_matches(sql_name: str, override_params: Dict[str, Any] = None, full_path:bool=False) -> Dict[str, Any]:
    """Fetch matches from OpenDota API using parameterized query."""
    # Load parameters
    sql_template, params = load_query_and_params(sql_name, full_path)
    
    # Format the SQL query with parameters
    sql_query = sql_template.format(**(params if not override_params else override_params))
    # Dump the SQL query to a temporary file for debugging
    with open('tmp.sql', 'w') as f:
        f.write(sql_query)
    
    # Construct the API URL
    base_url = "https://api.opendota.com/api/explorer"
    
    # Make the request with retry logic
    max_retries = 1
    retry_delay = 1
    timeout = 120
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                base_url,
                params={'sql': sql_query},
                headers={
                    'Accept': 'application/json',
                    'User-Agent': 'SmartDota/1.0'
                },
                timeout=timeout
            )
            
            # Print the actual URL for debugging
            print(f"Request URL: {response.url}")
            
            response.raise_for_status()
            
            # Parse and return the JSON response
            result = response.json()
            
            # Check if we got valid data
            if 'rows' not in result:
                raise ValueError("Invalid response format: 'rows' field missing")
                
            return result, params
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                raise
            print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

def save_results(result: Dict[str, Any], sql_name: str, params: Dict[str, Any]):
    """Save the results to a JSON file."""
    param_str = '_'.join(f"[{k}-{v}]" for k, v in params.items())
    out_file_name = f"fetched_datasets/{sql_name}__{param_str}.json"
    with open(out_file_name, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Data saved to {out_file_name}")

def main():
    parser = argparse.ArgumentParser(description='Fetch Dota 2 matches from OpenDota API')
    parser.add_argument('sql_name', help='Name of the SQL file')
    
    args = parser.parse_args()
    
    try:
        # Fetch matches
        result, params = fetch_opendota_matches(args.sql_name)
        
        # Print the number of matches fetched
        num_matches = len(result.get('rows', []))
        print(f"Successfully fetched {num_matches} matches")
        
        # Save the results
        save_results(result, args.sql_name, params)
        
        # Print some basic statistics
        if num_matches > 0:
            print("\nSample match data:")
            print(result['rows'][0])
        
    except Exception as e:
        print(f"Error: {e}")
        if isinstance(e, requests.exceptions.RequestException) and hasattr(e.response, 'text'):
            print(f"Response content: {e.response.text}")
        exit(1)

if __name__ == "__main__":
    main() 