import os
import math
from collections import defaultdict
from supabase import create_client

"""
Nightly script to update orchestrator action weights based on observed rewards (delta mastery).
"""

def main():
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY not set.")
        return
    client = create_client(supabase_url, supabase_key)

    # Fetch all orchestrator actions
    actions_res = client.table("actions").select("session_id, user_id, timestamp, action_type").execute()
    action_events = actions_res.data or []

    # Fetch all concept events with delta_mastery
    concepts_res = client.table("concept_events").select("session_id, user_id, timestamp, delta_mastery").execute()
    concept_events = concepts_res.data or []

    # Map action_type to list of delta_mastery
    rewards_by_action = defaultdict(list)

    # Associate each concept event with the most recent preceding action in the same session and user
    for ce in concept_events:
        ses = ce.get("session_id")
        uid = ce.get("user_id")
        ce_ts = ce.get("timestamp")
        # Filter actions for this session/user before event time
        past_actions = [a for a in action_events \
                        if a.get("session_id")==ses and a.get("user_id")==uid and a.get("timestamp") <= ce_ts]
        if not past_actions:
            continue
        # Find latest action
        last_a = max(past_actions, key=lambda a: a.get("timestamp"))
        atype = last_a.get("action_type")
        rewards_by_action[atype].append(ce.get("delta_mastery", 0.0))

    # Compute average reward for each action type
    avg_rewards = {}
    for atype, rewards in rewards_by_action.items():
        if rewards:
            avg_rewards[atype] = sum(rewards) / len(rewards)

    if not avg_rewards:
        print("No rewards found to update weights.")
        return

    # Compute softmax weights
    exp_vals = {atype: math.exp(r) for atype, r in avg_rewards.items()}
    total_exp = sum(exp_vals.values())
    new_weights = {atype: exp_vals[atype] / total_exp for atype in exp_vals}

    # Upsert new weights into action_weights table
    for atype, weight in new_weights.items():
        client.table("action_weights").upsert({
            "action_type": atype,
            "weight": weight
        }).execute()
        print(f"Updated weight for {atype}: {weight:.4f}")

if __name__ == "__main__":
    main() 