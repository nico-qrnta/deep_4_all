import urllib.request
import json
import csv
import os

url = "https://ddragon.leagueoflegends.com/cdn/15.1.1/data/en_US/champion.json"
print(f"Downloading {url}...")
try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    
    champions = []
    for key, val in data['data'].items():
        # 'key' in the dictionary is the string ID (e.g. "266")
        # 'id' is the internal name (e.g. "Aatrox")
        # 'name' is the display name (e.g. "Aatrox")
        champions.append({'id': val['key'], 'name': val['name']})
    
    # Sort by numeric ID
    champions.sort(key=lambda x: int(x['id']))
    
    output_path = "datasets/champions.csv"
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name'])
        writer.writeheader()
        writer.writerows(champions)
        
    print(f"Successfully saved {len(champions)} champions to {output_path}")

except Exception as e:
    print(f"Error: {e}")
