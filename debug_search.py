from ddgs import DDGS
import json

query = "Introducing Claude Opus 4.5"

print(f"Query: {query}")

def test_search(timelimit):
    print(f"\n--- Testing timelimit='{timelimit}' ---")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                region='wt-wt',
                safesearch='moderate',
                timelimit=timelimit,
                max_results=3
            ))
            for r in results:
                print(f"- {r['title']} ({r['href']})")
                print(f"  Snippet: {r['body'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")

test_search('y') # Past year (Current default)
test_search('m') # Past month
test_search('w') # Past week
test_search('d') # Past day
test_search(None) # No limit
