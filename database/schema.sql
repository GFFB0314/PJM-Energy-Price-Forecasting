-- The ugly CSVs downloaded from ISO (Independent Sources Operator)/JPM


-- Create a schema to keep things organized (Optional, but pro)-- 1. Create the schema
-- A schema is like a folder inside a database
-- It helps organize tables, views, functions, etc.
CREATE SCHEMA IF NOT EXISTS pjm_market;

-- Drop the table if it exists so we can rerun this script safely
DROP TABLE IF EXISTS pjm_market.raw_lmp;

-- Create the table matching CSV columns exactly
CREATE TABLE pjm_market.raw_lmp (
    datetime_beginning_utc TIMESTAMP,
    datetime_beginning_ept TIMESTAMP,
    pnode_id BIGINT,
    pnode_name TEXT,
    voltage TEXT,              
    equipment TEXT,            
    type TEXT,
    zone TEXT,
    system_energy_price_rt DECIMAL(10, 4),
    total_lmp_rt DECIMAL(10, 4),           -- TARGET VARIABLE
    congestion_price_rt DECIMAL(10, 4),
    marginal_loss_price_rt DECIMAL(10, 4),
    row_is_current BOOLEAN,
    version_nbr INTEGER
);