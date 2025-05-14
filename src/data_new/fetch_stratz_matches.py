import requests
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stratz_fetch.log'),
        logging.StreamHandler()
    ]
)

class RateLimiter:
    """
    the STRATZ API includes rate limits of 20 calls per second, 250/minute, 2,000/hour, and 10,000/day.
    """

    def __init__(self, calls_per_second: int = 20, calls_per_minute: int = 250):
        self.calls_per_second = calls_per_second
        self.calls_per_minute = calls_per_minute
        self.second_calls = []
        self.minute_calls = []
    
    def wait_if_needed(self):
        current_time = time.time()
        
        # Clean up old calls
        self.second_calls = [t for t in self.second_calls if current_time - t < 1]
        self.minute_calls = [t for t in self.minute_calls if current_time - t < 60]
        
        # Check if we need to wait
        if len(self.second_calls) >= self.calls_per_second:
            sleep_time = 1 + 1 - (current_time - self.second_calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if len(self.minute_calls) >= self.calls_per_minute:
            sleep_time = 60 + 60 - (current_time - self.minute_calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Record this call
        self.second_calls.append(current_time)
        self.minute_calls.append(current_time)

def load_graphql_query(query_name: str, path_to_gql: str) -> str:
    """Load GraphQL query from file."""
    query_path = Path(path_to_gql) / f"{query_name}.gql"
    try:
        with open(query_path, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"GraphQL query file not found: {query_path}")

def get_match_details(match_id: int, api_key: str, rate_limiter: RateLimiter, query_name: str, path_to_gql: str = 'graphQL') -> Dict[str, Any]:
    """Fetch match details from Stratz API."""
    url = f"https://api.stratz.com/graphql" #  ?key={api_key}"
    
    # Load and format the query
    query_template = load_graphql_query(query_name, path_to_gql)
    query = query_template.replace("$matchId", str(match_id))
    
    # Apply rate limiting
    rate_limiter.wait_if_needed()
    
    try:
        response = requests.post(
            url,
            json={
                "query": query,
                "operationName": "GetMatchById"
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "STRATZ_API"
            },
            timeout=10
        )

        # logging.info(response)
        # logging.info(response.text)
        response.raise_for_status()
        # logging.info(response.json())
        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching match {match_id}: {str(e)}")
        if hasattr(e.response, 'text'):
            logging.error(f"Response content: {e.response.text}")
        return None

def load_match_ids(args) -> List[int]:
    """Load match IDs from a JSON file."""
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    # logging.info(f"Taking matches from {args.offset} to {args.offset+args.limit}")
    return [match['match_id'] for match in data['rows']] # [args.offset : args.offset+args.limit]]

class MatchBuffer:
    def __init__(self, output_path: Path, buffer_size: int = 100):
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.buffer = []
        self.total_matches = 0
        self._initialize_file()
    
    def _initialize_file(self):
        """Initialize the file with empty matches array if it doesn't exist."""
        if not self.output_path.exists():
            with open(self.output_path, 'w') as f:
                json.dump({"matches": []}, f)
        else:
            # Count existing matches
            with open(self.output_path, 'r') as f:
                data = json.load(f)
                self.total_matches = len(data.get("matches", []))
    
    def add_match(self, match_data: Dict[str, Any]):
        """Add a match to the buffer and flush if buffer is full."""
        if match_data and 'data' in match_data and match_data['data'].get('match'):
            self.buffer.append(match_data['data']['match'])
            self.total_matches += 1
            
            if len(self.buffer) >= self.buffer_size:
                self.flush()
    
    def flush(self, force=False):
        """Flush the buffer to disk."""
        if not self.buffer:
            return
            
        # Read the current file content
        with open(self.output_path, 'r') as f:
            data = json.load(f)
        
        # Append new matches
        data["matches"].extend(self.buffer)
        
        # Write back to file
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Clear buffer
        self.buffer = []
        
        logging.info(f"Flushed {self.buffer_size} matches to disk. Total matches: {self.total_matches}")

def save_match_data(match_data: Dict[str, Any], output_dir: str, match_id: int):
    """Save match data to a JSON file using buffered writes."""
    output_path = Path(output_dir)
    
    # Initialize buffer if it doesn't exist
    if not hasattr(save_match_data, 'buffer'):
        save_match_data.buffer = MatchBuffer(output_path)
    
    # Add match to buffer
    save_match_data.buffer.add_match(match_data)
    
    logging.info(f"Buffered match {match_id} (Total matches: {save_match_data.buffer.total_matches})")

def main():
    parser = argparse.ArgumentParser(description='Fetch Dota 2 matches from Stratz API')
    parser.add_argument('--match_id', type=int, help='Single match ID to fetch')
    parser.add_argument('--input_file', help='Path to the JSON file containing match IDs')
    # parser.add_argument('--offset', type=int, default=0, help='Offset the number of matches to fetch')
    # parser.add_argument('--limit', type=int, default=10, help='Limit the number of matches to fetch')
    parser.add_argument('--api_key_path', required=True, help='Path to the Stratz API key file')
    parser.add_argument('--output_dir', default='fetched_datasets/stratz_matches.json',
                      help='Directory to save match data')
    parser.add_argument('--query_name', default='get_mathc_by_id',
                      help='Name of the GraphQL query file (without .gql extension)')
    parser.add_argument('--buffer_size', type=int, default=200,
                      help='Number of matches to buffer before writing to disk')
    parser.add_argument('--to_continue', type=bool, default=False, help='Continue from check point file')
    
    args = parser.parse_args()
    
    if not args.match_id and not args.input_file:
        parser.error("Either --match-id or --input-file must be provided")
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize rate limiter
    rate_limiter = RateLimiter()
    
    # Get match IDs to process
    match_ids = [args.match_id] if args.match_id else load_match_ids(args)
    logging.info(f"Processing {len(match_ids)} match IDs")
    if args.to_continue:
        logging.info(f"Loading used match IDs from {args.output_dir}")
        with open(args.output_dir, 'r') as f:
            data = json.load(f)
            used_match_ids = [match['id'] for match in data['matches']]
            logging.info(f"Found {len(used_match_ids)} used match IDs")
            match_ids = list(set(match_ids) - set(used_match_ids))
            logging.info(f"Remaining {len(match_ids)} match IDs to fetch")
    
    # Fetch matches
    successful_fetches = 0
    failed_fetches = 0
    not_public_fetches = 0
    with open(args.api_key_path, 'r') as f:
        api_key = f.read().strip()
    
    try:
        for match_id in match_ids:
            logging.info(f"Fetching match {match_id}")
            match_data, status_code = get_match_details(match_id, api_key, rate_limiter, args.query_name)
            
            if match_data and 'data' in match_data and match_data['data'].get('match'):
                save_match_data(match_data, args.output_dir, match_id)
                successful_fetches += 1
                logging.info(f"Successfully saved match {successful_fetches}: {match_id}")
            elif status_code == 200:
                save_match_data({'data': {'match': {'id': match_id, 'public': False}}}, args.output_dir, match_id)
                not_public_fetches += 1
                logging.info(f"Match {match_id} is not public")
                logging.info(f"Empty fetched {not_public_fetches}: {match_id}")
            else:
                failed_fetches += 1
                logging.error(f"Failed to fetch match {failed_fetches}: {match_id}")
        
        # Flush any remaining matches in the buffer
        if hasattr(save_match_data, 'buffer'):
            save_match_data.buffer.flush()
        
        logging.info(f"Fetching complete. Successful: {successful_fetches}, Failed: {failed_fetches}")
    
    except KeyboardInterrupt:
        logging.info("Process interrupted by user. Flushing buffer...")
        if hasattr(save_match_data, 'buffer'):
            save_match_data.buffer.flush(force=True)
        logging.info("Buffer flushed. Exiting...")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        if hasattr(save_match_data, 'buffer'):
            save_match_data.buffer.flush(force=True)
        exit(1)

if __name__ == "__main__":
    main()