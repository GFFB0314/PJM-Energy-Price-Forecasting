# Intelligent Energy Arbitrage Engine ⚡

Welcome to the **Intelligent Energy Arbitrage Engine** – a production-grade data science system designed to optimize battery storage revenue in the PJM energy market. This project demonstrates the end-to-end lifecycle of a machine learning application, from raw data ingestion to financial impact analysis, moving beyond simple metrics to solve a real-world business problem.

## Table of Contents
- [Intelligent Energy Arbitrage Engine ⚡](#intelligent-energy-arbitrage-engine-)
  - [Table of Contents](#table-of-contents)
  - [About the Project 📖](#about-the-project-)
    - [✨ Key Features:](#-key-features)
  - [Data Strategy \& Engineering Pipeline 🏗️](#data-strategy--engineering-pipeline-️)
    - [1. Data Sourcing](#1-data-sourcing)
    - [2. Database Engineering (SQL)](#2-database-engineering-sql)
    - [3. Physics-Aware Feature Engineering](#3-physics-aware-feature-engineering)
    - [4. Financial Simulation (PnL)](#4-financial-simulation-pnl)
  - [Technologies \& Libraries Used 🛠️](#technologies--libraries-used-️)
  - [Getting Started 🚀](#getting-started-)
    - [Prerequisites](#prerequisites)
    - [⚙️ Installation](#️-installation)
  - [Usage Guide 🖥️](#usage-guide-️)
  - [Results \& Impact 📊](#results--impact-)
  - [Contributing 🤝](#contributing-)
  - [Contact ✉️](#contact-️)
  - [License ©️](#license-️)

## About the Project 📖

The goal of this project is to identify profitable arbitrage opportunities (buying low, selling high) in the PJM Western Hub Real-Time Hourly Market. It moves beyond standard linear forecasting by incorporating "Physics-Aware" feature engineering to model the complex relationship between weather and energy demand.

This project was built to demonstrate **robust programming skills and Data Science expertise**, focusing on reproducibility, statistical rigor, and business ROI.

### ✨ Key Features:
*   **Physics-Aware Modeling:** Uses Polynomial Features to capture the non-linear "Duck Curve" and seasonal demand shifts (Heating vs. Cooling loads).
*   **Modular Architecture:** Clean Python package structure (`src/`) separating ETL, Modeling, and Configuration logic.
*   **Robust ETL Pipeline:** Automated extraction, cleaning, and merging of disparate data sources.
*   **Financial Impact Analysis:** Evaluates the model based on **Realized Profit ($)** and **Efficiency (%)**, not just RMSE.
*   **Integrated Testing:** Includes unit tests to verify data transformations and pipeline integrity.
*   **Secure Configuration:** Uses environment variables (`.env`) to handle credentials securely.

---

## Data Strategy & Engineering Pipeline 🏗️

This project follows a strict **ELT (Extract, Load, Transform)** workflow to ensure data quality and scalability.

### 1. Data Sourcing
*   **Energy Prices:** Extracted hourly "Real-Time Locational Marginal Pricing" (LMP) from the **PJM Data Miner** (Western Hub Node).
*   **Weather Data:** Ingested historical hourly weather data (Temperature, Wind Speed, Solar Radiation) via the **Open-Meteo API**, using Harrisburg, PA as the geospatial proxy for the Western Hub.

### 2. Database Engineering (SQL)
Instead of relying solely on Pandas, I utilized **PostgreSQL** for heavy data lifting.
*   **Ingestion:** Raw CSV data is loaded into a staging table (`raw_lmp`).
*   **Window Functions:** I utilized SQL Window Functions (`LAG`, `AVG OVER`) to engineer temporal features directly in the database.
    *   *Example:* Creating `price_24h_ago` allows the model to capture daily seasonality without complex Python loops.
    *   *Example:* Creating `avg_price_last_24h` captures the immediate market trend/momentum.

### 3. Physics-Aware Feature Engineering
Exploratory Data Analysis (EDA) revealed a non-linear relationship between Temperature and Price (a "U-Shape").
*   **The Physics:** Extreme Cold (Heating) and Extreme Heat (AC) both drive prices up, while mild temperatures lower demand.
*   **The Engineering:** I implemented a `ColumnTransformer` with `PolynomialFeatures` to mathematically represent this "U-curve" and capture interactions (e.g., *High Temp* × *High Solar* = Lower Price due to solar generation offset).

### 4. Financial Simulation (PnL)
A machine learning model is only as good as the value it creates.
*   **Scenario:** A 100 MWh Battery Asset with 1 cycle per day.
*   **Strategy:** The system uses the **Gradient Boosting Regressor** to predict tomorrow's prices, generating "Buy" signals at the predicted daily low and "Sell" signals at the predicted daily high.

---

## Technologies & Libraries Used 🛠️

This project leverages a modern Python Data Science stack.

*   🐍 **Python 3.10+:** The core language.
*   🐼 **Pandas & NumPy:** For high-performance data manipulation and vectorization.
*   📊 **Matplotlib & Seaborn:** For visualizing EDA, correlation heatmaps, and price volatility.
*   🤖 **Scikit-Learn:** For machine learning pipelines, regression models, and cross-validation.
*   🗄️ **SQLAlchemy & Psycopg2:** For robust database interaction.
*   🧪 **Pytest:** For unit testing and verifying pipeline integrity.
*   ☁️ **Open-Meteo API:** For historical weather data ingestion.

---

## Getting Started 🚀

Follow these steps to get a local copy up and running.

### Prerequisites

*   **Python 3.10+**
*   **PostgreSQL** (Local Instance)

### ⚙️ Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/GFFB0314/Energy_Arbitrage_Project_Repo.git
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Database Setup:**
    *   Create a local PostgreSQL database (e.g., `energy_db`).
    *   Create a `.env` file in the root directory (see `src/config.py` for keys).
    *   Run the schema creation scripts found in `database/`.

4.  **Run Tests:**
    Verify the logic by running the test suite.
    ```bash
    pytest tests/
    ```

---

## Usage Guide 🖥️

You can interact with the project via Jupyter Notebooks for exploration or CLI for execution.

*   **Notebooks (`notebooks/`):**
    *   `01_data_extraction.ipynb`: API extraction & SQL Loading.
    *   `02_eda_and_sql.ipynb`: EDA & SQL Feature Engineering visualization.
    *   `03_modeling.ipynb`: Pipeline training and PnL backtesting.

*   **Command Line Interface:**
    To run the full ETL or Training pipeline from the terminal:
    ```bash
    # Run ETL
    python main.py --step etl
    
    # Run Training & PnL
    python main.py --step train

    # Run the entire workflow
    python main.py --step all
    ```

---

## Results & Impact 📊

In a simulated backtest on the Holdout Set (Q4 2024):
*   **Total Market Potential:** $428,227 (Theoretical Max)
*   **Realized Revenue:** **$202,434** (Captured by Model) 💰
*   **Capture Efficiency:** **47.3%** 📈

The model successfully captures nearly half of the theoretical maximum profit available in the market, significantly outperforming naive baselines and validating the use of non-linear feature engineering.

---

## Contributing 🤝

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  **Fork** the Project
2.  Create your Feature Branch (`git checkout -b feature/NewAlgorithm`)
3.  Commit your Changes (`git commit -m 'Add LSTM Model'`)
4.  Push to the Branch (`git push origin feature/NewAlgorithm`)
5.  Open a **Pull Request**

---

## Contact ✉️
For any questions, issues, or suggestions, please feel free to contact:
- Email: gbetnkom.bechir@gmail.com
- LinkedIn: [Fares Fahim Bechir Gbetnkom](www.linkedin.com/in/fares-fahim-bechir-gbetnkom-386a782a6)
- GitHub Issues: [Project Issues](https://github.com/GFFB0314/GB_Interpreter/issues)

---

## License ©️
**MIT License** 📝

**© 2026 Fares Gbetnkom**. This project is licensed under the **MIT License** — feel free to use, modify, and distribute it. See the full license text [here](LICENSE).

Happy Forecasting! ⚡