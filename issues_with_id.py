import time
import requests
import json
from time import sleep
from requests.exceptions import RequestException, ConnectionError, Timeout


username = 'norma1guy'
token = ''
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}


def get_issues_with_one_assignee():
    issues_with_one_assignee = []
    page = 1
    per_page = 100  
    max_pages = 2200 
    retries = 3  
    timeout = 3  
    
    while page <= max_pages:
        try:
            
            url = f'https://api.github.com/repos/microsoft/vscode/issues?page={page}&per_page={per_page}'
            
            for attempt in range(retries):
                try:
                    response = requests.get(url, headers=headers, timeout=timeout)
                    
                    if response.status_code == 200:
                        issues = response.json()

                        if issues:
    
                            issues = [issue for issue in issues if 'pull_request' not in issue and issues['id'] <= 210000]
                            #issues = [issue for issue in issues if len(issue['assignees']) == 1]
                            issues_with_one_assignee.extend(issues)
                            print(f"Fetched page {page} with {len(issues)} issues")
                            page += 1  
                            break  
                        
                        else:
                            print(f"No more issues found at page {page}")
                            return issues_with_one_assignee 

                    elif response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers:
                        
                        reset_time = int(response.headers['X-RateLimit-Reset'])
                        sleep_duration = max(reset_time - time.time(), 0)
                        print(f"Rate limit exceeded. Sleeping for {sleep_duration / 60:.2f} minutes.")
                        sleep(sleep_duration)
                        break  
                    
                    else:
                        print(f"Failed to fetch page {page}, status code: {response.status_code}")
                        break  
                
                except (ConnectionError, Timeout) as e:
                    print(f"Request failed on page {page}: {e}. Attempt {attempt + 1}/{retries}")
                    if attempt < retries - 1:
                        sleep(2)  
                    else:
                        print(f"Max retries reached for page {page}. Skipping this page.")
                        break  

        except RequestException as e:
            print(f"Unexpected error: {e}")
            break  

    return issues_with_one_assignee

def to_json(issues, filename="issues_with_one_assignee.json"):
    with open(filename, 'w') as file:
        json.dump(issues, file, indent=4)
    print(f"Saved {len(issues)} issues to {filename}")

issues_with_one_assignee = get_issues_with_one_assignee()
to_json(issues_with_one_assignee)
