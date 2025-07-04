import json
import os

# Your new batch (paste or load this from another file if you prefer)
new_batch = [
  {
    "business_description": "A blockchain platform that verifies the authenticity and provenance of luxury handbags and watches for resale markets."
  },
  {
    "business_description": "A mental health app that uses biometric data from smartwatches to detect early signs of anxiety and suggest coping techniques."
  },
  {
    "business_description": "An online marketplace for renting musical instruments to students with flexible monthly payment plans."
  },
  {
    "business_description": "A carbon capture service that partners with factories to install equipment that converts emissions into building materials."
  },
  {
    "business_description": "A virtual classroom platform that specializes in teaching sign language through interactive video lessons with deaf instructors."
  },
  {
    "business_description": "An AI-powered recruiting tool that removes bias from job postings and matches candidates based on skills rather than demographics."
  },
  {
    "business_description": "A specialized logistics service for transporting live fish and aquatic plants to aquarium stores and hobbyists."
  },
  {
    "business_description": "A peer-to-peer lending platform that connects retirees with steady income streams to young entrepreneurs needing startup capital."
  },
  {
    "business_description": "A social media app designed specifically for book lovers to share reading recommendations and organize virtual book clubs."
  },
  {
    "business_description": "A telehealth platform that provides remote physical therapy sessions using motion-tracking technology and exercise equipment rentals."
  }
]
# File path
dataset_path = "data_injection.jsonl"

# Step 1: Load existing descriptions
existing_descriptions = set()
if os.path.exists(dataset_path):
    with open(dataset_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                desc = obj.get("business_description", "").strip().lower()
                existing_descriptions.add(desc)
            except json.JSONDecodeError:
                pass

# Step 2: Filter and append only unique entries
new_entries_written = 0
with open(dataset_path, "a") as f:
    for entry in new_batch:
        desc = entry["business_description"].strip().lower()
        if desc not in existing_descriptions:
            f.write(json.dumps(entry) + "\n")
            existing_descriptions.add(desc)
            new_entries_written += 1
            print(f"✅ ADDED: {entry['business_description']}")
        else:
            print(f"❌ SKIPPED (duplicate): {entry['business_description']}")

print(f"\n🎯 Total new entries added: {new_entries_written}")