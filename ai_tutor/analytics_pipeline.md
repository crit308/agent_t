# Analytics Pipeline

To support real-time and batch analytics on user interactions and orchestrator decisions, the AI Tutor project streams and processes logs via the following pipeline:

## 1. Supabase CDC → Kafka

Enable change-data-capture on the logging tables in Supabase:

```sql
ALTER TABLE public.concept_events REPLICATE ON CHANGE;
ALTER TABLE public.actions REPLICATE ON CHANGE;
```

Supabase will publish inserts on `public.concept_events` and `public.actions` to Kafka topics:

- `concept_events`
- `actions`

## 2. Kafka → ClickHouse

Define Kafka-engine tables in ClickHouse to ingest the CDC stream:

```sql
CREATE TABLE analytics.concept_events (
    id UInt64,
    session_id UUID,
    user_id UUID,
    concept String,
    outcome String,
    timestamp DateTime64(3),
    delta_mastery Float64
) ENGINE = Kafka SETTINGS
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'concept_events',
    kafka_group_name = 'clickhouse_consumer',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 1;

CREATE TABLE analytics.actions (
    id UInt64,
    session_id UUID,
    user_id UUID,
    action_type String,
    action_details String,
    timestamp DateTime64(3)
) ENGINE = Kafka SETTINGS
    kafka_broker_list = 'localhost:9092',
    kafka_topic_list = 'actions',
    kafka_group_name = 'clickhouse_consumer',
    kafka_format = 'JSONEachRow',
    kafka_num_consumers = 1;
```

## 3. Materialization to MergeTree

Materialize the streaming tables into durable MergeTree tables for efficient querying:

```sql
CREATE TABLE analytics.concept_events_mt AS analytics.concept_events
ENGINE = MergeTree()
ORDER BY (session_id, timestamp);

CREATE TABLE analytics.actions_mt AS analytics.actions
ENGINE = MergeTree()
ORDER BY (session_id, timestamp);
```

## 4. Nightly Weight Update

A nightly job (e.g. via cron) runs:

```bash
python scripts/update_action_weights.py
```

This script:

1. Fetches entries from `public.actions` and `public.concept_events`.
2. Associates each concept event's `delta_mastery` with the most recent preceding action of the same session/user.
3. Computes average rewards per action type and applies a softmax transform.
4. Upserts new sampling weights into `public.action_weights`.

Ensure the environment variables `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set before running the script. 