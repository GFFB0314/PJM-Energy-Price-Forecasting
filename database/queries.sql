-- Complex Window Function Queries for Energy Arbitrage Analysis


/*			VIEW
 * A view is a virtual table.
 * It runs the calculations everytime you look at it.
 * 
 * 			WHY USING VIEW INSTEAD OF TABLE
 * 1. CREATE TABLE makes a photocopy of your data. If you find a mistake in the raw data later and fix it, your new table is 
 * now "stale" (outdated). You have to remember to delete and recreate it.
 * 2. CREATE VIEW is like putting a Lens over the raw table. It saves the code, not the data. Every time you run SELECT * FROM 
 * view, it runs the calculation fresh on the raw data.
 * 	- It saves hard drive space.
 *  - It ensures you are always looking at the most current version of the data.
 * 	- In Data Engineering, this is called "idempotency"—you can run the code 100 times and it never breaks or duplicates data.
 * 
 * */
 
CREATE OR REPLACE VIEW pjm_market.features_v AS
-- "Save this recipe as 'features_v'. If it exists, overwrite it."

SELECT
    datetime_beginning_ept,
    -- We keep the local timestamp as our index.

    total_lmp_rt as price_actual,
    -- I renamed this because "total_lmp_rt" is jargon. 
    -- "price_actual" tells us this is the ground truth.

    EXTRACT(HOUR FROM datetime_beginning_ept) as hour_of_day,
    -- Machine Learning models cannot read dates. They need numbers.
    -- We extract "17" from "2024-01-01 17:00:00".
    -- This helps the model learn: "Hour 17 is usually expensive."

    EXTRACT(DOW FROM datetime_beginning_ept) as day_of_week,
    -- DOW = Day of Week (0 is Sunday, 6 is Saturday).
    -- This helps the model learn: "Weekends are usually cheaper."

    EXTRACT(MONTH FROM datetime_beginning_ept) as month,
    -- Helps the model learn: "July (Month 7) is hot and expensive."

    -- WINDOW FUNCTION #1: Lag
    LAG(total_lmp_rt, 1) OVER (ORDER BY datetime_beginning_ept) as price_1h_ago,
    -- "Sort all rows by time."
    -- "Look at the row immediately before this one (Lag 1)."
    -- "Grab that price."
    -- Logic: The price 1 hour ago is highly correlated to the price now.

    -- WINDOW FUNCTION #2: 24h Lag
    LAG(total_lmp_rt, 24) OVER (ORDER BY datetime_beginning_ept) as price_24h_ago,
    -- "Look at the row exactly 24 steps back."
    -- Logic: The price at 5 PM yesterday is a great guess for 5 PM today.

    -- WINDOW FUNCTION #3: Rolling Average (The Complex One)
    AVG(total_lmp_rt) OVER (
        ORDER BY datetime_beginning_ept
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    ) as avg_price_last_24h
    -- "Create a window."
    -- "Start 24 rows back."
    -- "End 1 row back."
    -- "Take the Average of those 24 numbers."
    
    -- CRITICAL: Why "AND 1 PRECEDING"? Why not "CURRENT ROW"?
    -- If we include the CURRENT ROW, we are using the answer to predict the answer.
    -- That is called "Data Leakage." It gives you 100% accuracy in training 
    -- but fails in the real world. We MUST stop looking 1 hour ago.

FROM pjm_market.raw_lmp
ORDER BY datetime_beginning_ept;