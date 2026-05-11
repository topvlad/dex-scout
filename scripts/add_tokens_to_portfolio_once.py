#!/usr/bin/env python3
from app import manual_add_token_to_portfolio
TOKENS = [
"EK4cBucmRcKxwjNGEdGVhSMcTK9aPXpgKmbzyrTSpump",
"8x5VqbHA8D7NkD52uNuS5nnt3PwA8pLD34ymskeSo2Wn",
"NV2RYH954cTJ3ckFUpvfqaQXU4ARqqDH3562nFSpump",
"J3NKxxXZcnNiMjKw9hYb2K4LUxgwB6t1FtPtQVsv3KFr",
"9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump",
"G6mNZN8o16QBcTqfuEx6FzjiWa94B1XWhfyDxjDibrrr",
]

if __name__ == '__main__':
    totals={"portfolio_added":0,"portfolio_existing":0,"failed":0}
    for ca in TOKENS:
        res=manual_add_token_to_portfolio(raw_input=ca, chain='solana', note='one-off seed', also_monitoring=False)
        print(ca, res)
        for k in totals:
            totals[k]+=int(res.get(k,0) or 0)
    print(totals)
