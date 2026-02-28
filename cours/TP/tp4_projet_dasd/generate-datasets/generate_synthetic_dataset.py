import argparse
import json
import time
import os
import random
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datasets.helpers import load_match_data, format_draft_state, confidence_score
from prompts import SYSTEM_PROMPT, RESPONSE_FORMAT

load_dotenv()

# TP4 Section 3.3 parameters
TEMP_LOW = 0.3
TEMP_HIGH = 0.9

async def generate_entry_async(client, draft_desc, style):
    """Generate a single dataset entry with both low and high temperature responses concurrently."""
    
    instruction = f"Situation de draft ({style}): {draft_desc}"
    
    entry = {
        "input": instruction,
        "system_prompt": SYSTEM_PROMPT,
        "draft_style": style,
        "stage1_response": None,
        "stage2_response": None
    }

    async def get_response(temp, stage_name):
        try:
            # print(f"  > Launching {stage_name} (Temp {temp})...")
            start = time.time()
            resp = await client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                ],
                temperature=temp,
                max_tokens=2500,
                response_format=RESPONSE_FORMAT,
                logprobs=True,
                top_logprobs=1
            )
            # duration = time.time() - start
            # print(f"  < {stage_name} finished in {duration:.2f}s")
            
            c_score = 0.0
            if resp.choices[0].logprobs:
                c_score = confidence_score(resp.choices[0].logprobs.content)

            return {
                "temperature": temp,
                "content": resp.choices[0].message.content,
                "confidence": c_score
            }
        except Exception as e:
            print(f"Error generating {stage_name} for {style}: {e}")
            return None

    # Run tasks concurrently
    task_low = get_response(TEMP_LOW, "Stage 1")
    task_high = get_response(TEMP_HIGH, "Stage 2")
    
    results = await asyncio.gather(task_low, task_high)
    
    entry["stage1_response"] = results[0]
    entry["stage2_response"] = results[1]
    
    if entry["stage1_response"] is None or entry["stage2_response"] is None:
        return None

    return entry

async def generate_dataset_async(csv_input, json_output, styles, limit=None, start=None, end=None):
    """Generate synthetic draft dataset in JSON format using async requests."""
    print(f"Loading matches from {csv_input}...")
    df = load_match_data(csv_input)
    
    if start is not None:
        end_idx = end if end is not None else len(df)
        print(f"Slicing dataframe from {start} to {end_idx}...")
        df = df.iloc[start:end_idx]
    elif limit:
        df = df.sample(n=min(limit, len(df))).reset_index(drop=True)
    
    # Init Async Client
    api_key = os.getenv("INFOMANIAK_API_KEY")
    product_id = os.getenv("INFOMANIAK_PRODUCT_ID")
    if not api_key or not product_id:
        raise ValueError("INFOMANIAK_API_KEY or INFOMANIAK_PRODUCT_ID missing in .env")

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=f"https://api.infomaniak.com/2/ai/{product_id}/openai/v1"
    )

    dataset = []
    
    print(f"Starting generation for {len(df)} matches with styles: {styles}...")
    
    durations = []
    
    try:
        for i, (_, row) in enumerate(df.iterrows()):
            current_count = i + 1
            start_item = time.time()
            
            style = random.choice(styles)
            draft_desc = format_draft_state(row, style=style)
            if not draft_desc:
                continue
            
            # Async generation
            entry = await generate_entry_async(client, draft_desc, style)
            
            if entry:
                dataset.append(entry)

            duration = time.time() - start_item
            durations.append(duration)
            if durations:
                avg_duration = sum(durations) / len(durations)
                remaining_items = len(df) - current_count
                etr = avg_duration * remaining_items
                
                print(f"Match {current_count}/{len(df)} processed in {duration:.1f}s. ETR: {etr/60:.1f} min. Total saved: {len(dataset)}")
            
            # Incremental save every 5 items
            if len(dataset) % 5 == 0 and len(dataset) > 0:
                os.makedirs(os.path.dirname(json_output), exist_ok=True)
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(dataset, f, ensure_ascii=False, indent=2)

            # Wait 0.5s before next
            await asyncio.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user. Saving progress...")
    except Exception as e:
        print(f"\nUnexpected error: {e}. Saving progress...")
    finally:
        # Final save
        if dataset:
            os.makedirs(os.path.dirname(json_output), exist_ok=True)
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Final save: {len(dataset)} entries to {json_output}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LoL draft dataset (TP4 DASD format)")
    parser.add_argument("--input", type=str, default="datasets/matchData.csv",
                        help="Path to the original match CSV")
    parser.add_argument("--output", type=str, default="datasets/synthetic_dataset.json",
                        help="Path to save the synthetic dataset JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of source matches to process (ignored if start/end used)")
    parser.add_argument("--start", type=int, default=None,
                        help="Start index (row number) to process")
    parser.add_argument("--end", type=int, default=None,
                        help="End index (row number) to process")
    parser.add_argument("--styles", nargs="+", default=["random"],
                        help="Draft styles to generate (use 'random' for random state)")
    args = parser.parse_args()
    
    asyncio.run(generate_dataset_async(args.input, args.output, args.styles, args.limit, args.start, args.end))

if __name__ == "__main__":
    main()
