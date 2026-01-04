"""CLI Entry-point for the entire workflow"""

import argparse
import logging
import sys
from src.etl import run_etl_pipeline
from src.model import load_processed_data, prepare_features, train_and_evaluate, run_pnl_simulation, save_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Energy Arbitrage Project CLI")
    parser.add_argument("--step", type=str, choices=["etl", "train", "all"], required=True, help="Step to run")
    
    args = parser.parse_args()
    
    if args.step in ["etl", "all"]:
        logger.info(">>> STEP 1: ETL")
        run_etl_pipeline()
        
    if args.step in ["train", "all"]:
        logger.info(">>> STEP 2: TRAINING")
        try:
            # 1. Load Data
            df = load_processed_data()
            
            # 2. Prepare
            X_train, X_test, y_train, y_test = prepare_features(df)
            
            # 3. Train
            best_model, leaderboard = train_and_evaluate(X_train, y_train, X_test, y_test)
            print("\nLeaderboard:\n", leaderboard)
            
            # 4. PnL Simulation
            run_pnl_simulation(best_model, X_test, y_test)
            
            # 5. Save
            save_model(best_model)
            
        except FileNotFoundError:
            logger.error("Processed data not found. Run --step etl first.")
            sys.exit(1)

if __name__ == "__main__":
    main()