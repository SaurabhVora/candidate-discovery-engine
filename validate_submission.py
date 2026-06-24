import csv
import sys
import os

def validate_csv(filepath):
    print(f"Validating {filepath}...")
    
    if not os.path.exists(filepath):
        print("❌ Error: File does not exist.")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        
        # 1. Check Header
        expected_header = ['candidate_id', 'rank', 'score', 'reasoning']
        if header != expected_header:
            print(f"❌ Error: Invalid header. Expected {expected_header}, got {header}")
            return False
        print("✅ Header is correct.")
        
        # 2. Check Rows
        rows = list(reader)
        if len(rows) != 100:
            print(f"❌ Error: Expected exactly 100 candidates, but got {len(rows)} rows.")
            return False
        print("✅ Correct number of rows (100).")
        
        seen_ids = set()
        seen_ranks = set()
        
        for i, row in enumerate(rows):
            if len(row) != 4:
                print(f"❌ Error on row {i+2}: Expected 4 columns, got {len(row)}")
                return False
                
            cid, rank, score, reasoning = row
            
            # 3. Check Candidate ID
            if not cid.startswith("CAND_"):
                print(f"❌ Error on row {i+2}: Invalid candidate_id format: {cid}")
                return False
            if cid in seen_ids:
                print(f"❌ Error on row {i+2}: Duplicate candidate_id: {cid}")
                return False
            seen_ids.add(cid)
            
            # 4. Check Rank
            try:
                rank_int = int(rank)
            except ValueError:
                print(f"❌ Error on row {i+2}: Rank must be an integer, got {rank}")
                return False
            
            if rank_int < 1 or rank_int > 100:
                print(f"❌ Error on row {i+2}: Rank must be between 1 and 100, got {rank_int}")
                return False
            if rank_int in seen_ranks:
                print(f"❌ Error on row {i+2}: Duplicate rank: {rank_int}")
                return False
            seen_ranks.add(rank_int)
            
            # 5. Check Score
            try:
                float(score)
            except ValueError:
                print(f"❌ Error on row {i+2}: Score must be a float, got {score}")
                return False
                
            # 6. Check Reasoning
            if not reasoning.strip():
                print(f"❌ Error on row {i+2}: Reasoning cannot be empty.")
                return False
                
        # 7. Check if ranks are sequential 1-100
        if sorted(list(seen_ranks)) != list(range(1, 101)):
            print(f"❌ Error: Ranks are not perfectly sequential from 1 to 100.")
            return False
            
    print("✅ All validation checks passed! The CSV is ready for submission.")
    return True

if __name__ == "__main__":
    validate_csv("team_submission.csv")
