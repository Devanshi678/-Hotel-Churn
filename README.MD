# Title : Hotel-Churn-Prediction-E2E

Date : 1/29/26 | Name : Devanshikumari Patel 

---

## Table of Contents
1. [Project Summary](#project-summary)
2. [Scoping](#scoping)
3. [Project Setup & Data Pipeline (COMPLETED)](#project-setup--data-pipeline-completed)
4. [Next Steps](#next-steps)

---

## Project Summary

● **Problem:**

"Silent Churn" in hotels where regular guests stop booking without warning. Compounding this, generic retention strategies fail because they treat "Business Regulars" and "Seasonal Vacationers" exactly the same.

● **Why it matters / impact:**

Acquiring new guests costs 5x-25x more than retaining existing ones. Reducing churn by 5% can increase profits by 25-95%. Furthermore, misidentifying churn (e.g., flagging a seasonal guest as "churned" in the off-season) leads to wasted marketing budget on unnecessary discounts.

● **Proposed approach:** A Hybrid ML Pipeline.

1. Unsupervised Learning (K-Prototypes): To auto-segment customers into behavioral personas (e.g., "Corporate Traveler," "Summer Vacationer") based on spending, payment, and stay patterns.
2. Supervised Learning (XGBoost): Uses these personas as "context features" alongside historical data to accurately predict churn probability.

● **Primary success metric:**

Precision @ Top-20% (Ensuring we only target high-probability churners) and Segmented Retention ROI.

● **Key risks / unknowns:**

"Cluster Stability"—if customer behavior changes rapidly, the personas (clusters) might become invalid and need re-training.

---

## Scoping

### Business Problem
Hotels are currently facing a "Leaky Bucket" problem with a blindfold on. They only realize a customer has left months after the fact. Worse, they lack context: they might panic because a "Summer Vacationer" hasn't visited in November, or ignore a "Weekly Business Traveler" who missed two weeks. They need a system that understands who the customer is to correctly predict if they are leaving.

### Goal

1. **Segmentation:** Automatically group the customer base into 3-5 distinct behavioral personas using unsupervised learning.
2. **Prediction:** Predict churn within a 90-day window with >75% precision for high-risk segments.
3. **Actionability:** Deliver a weekly "Target List" that tells Marketing who is at risk and what type of customer they are (to tailor the offer).

### Stakeholders

1. **Project Lead:** Devanshikumari Patel (End-to-end Architecture & Dev).
2. **Hotel Operations/Franchise Owner:** (Consumer of the "At Risk" reports).
3. **Marketing Team:** (Executes segmented campaigns: Upgrades for Business, Discounts for Vacationers).

### Prior Work

**Current State:** Manual "Recency Rules" (e.g., "If last_booking > 365 days, send email").

**Flaws:** This ignores seasonality. A customer who only visits in July will always be flagged as "churned" in December, causing spam and brand fatigue.

### Online Metrics (Real-world business impact)

- **Campaign Conversion Rate:** % of targeted guests who book within 30 days of receiving an offer.
- **Retention ROI:** (Revenue from Saved Guests) - (Cost of Incentives).

### Offline Metrics (Model performance)

- **Silhouette Score:** To measure how well-separated our Unsupervised Clusters are.
- **Precision @ Top-K:** To ensure the churn model is accurate specifically on the highest-risk customers we intend to contact.
- **F1-Score (Macro):** To ensure the model performs well across all different customer clusters, not just the majority group.

### Online vs. Batch

**Mode:** Batch Prediction (Weekly).

**Reasoning:** Customer habits change slowly. Re-clustering and Churn prediction can be run every Monday morning to generate the weekly marketing list. Real-time inference is overkill and costly.

### Requirements

- **Cost:** < $10/month (Local/Colab training, free-tier tools).
- **Interpretability:** The clusters must be explainable to business owners (e.g., "Cluster 1 is clearly Business Travelers because they pay with Corporate Cards and stay weekdays").
- **Pipeline:** Fully automated via YAML configs and GitHub Actions (optional).

### Project Risks

- **Cluster Drift:** If the hotel changes pricing or strategy, the "Personas" might change, requiring the unsupervised model to be retrained.
- **Data Leakage:** Using future booking info to define current clusters. (Mitigation: Strict time-based splitting).
- **Cold Start:** New customers with only 1 booking are hard to cluster and hard to predict.

### Target Launch Dates

- **Week 1:** Data Ingestion, Cleaning, and Unsupervised Clustering (MVP).
- **Week 2:** Feature Engineering & Supervised Churn Model.
- **Week 3:** Pipeline Orchestration, Evaluation, and Documentation, deployment.

---

## Project Setup & Data Pipeline (COMPLETED)

**Date Completed:** February 24, 2026

### Hotel Context
- **Type:** Hill Station Hotel (mid-sized town location)
- **Room Types:** Queen Bed ($120/night), King Bed ($150/night), Suite ($300/night)
- **Booking Channels:** Direct Website, Agent, OTA (Booking.com), Walk-in
- **Special Requests:** Pet Friendly, Accessible Room, Crib, High Floor

### Customer Segments
The system generates three customer personas with segment-specific churn definitions:

| Segment | Booking Frequency | Avg Stay | Churn Definition |
|---------|------------------|----------|------------------|
| Business Traveler | High (monthly) | 3 nights | No booking in 180 days |
| Vacation Traveler | Regular (quarterly) | 7 nights | No booking in 90 days |
| Occasional Visitor | Low (yearly) | 5 nights | No booking in 365 days |

### Technology Stack

**Infrastructure:**
- Docker (PostgreSQL 15)
- Git/GitHub for version control
- Python 3.12+ with virtual environment (venv)

**Python Libraries:**
- `faker` - Synthetic data generation
- `sqlalchemy` - Database ORM
- `psycopg2-binary` - PostgreSQL driver
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `pandera` - Data quality validation
- `python-dotenv` - Environment variable management
- `pyyaml` - Configuration management
- `prefect` - Workflow orchestration (scheduled for Week 2)

### Project Structure

```
Hotel-Churn-Prediction/
├── config/
│   └── config.yaml              # Data generation & business rules config
├── src/
│   ├── __init__.py
│   ├── schema.py                # PostgreSQL table definitions (SQLAlchemy ORM)
│   ├── database.py              # Database connection management
│   ├── data_generator.py        # Faker-based synthetic data generation
│   ├── quality_check.py         # Pandera validation schemas
│   └── ingest.py                # Pipeline orchestration
├── venv/                        # Virtual environment (not in Git)
├── .env                         # Database credentials (not in Git)
├── .gitignore                   # Excludes .env, venv, __pycache__
├── docker-compose.yml           # PostgreSQL container configuration
├── requirements.txt             # Python dependencies
├── main.py                      # CLI entry point
└── README.md                    # This file
```

### Database Schema

**Table 1: customers**
| Column | Type | Description |
|--------|------|-------------|
| customer_id | INT (PK) | Auto-incrementing unique ID |
| join_date | DATE | When customer first signed up |
| age | INT | 21-75 years |
| job_category | VARCHAR | business_traveler, vacation_traveler, occasional_visitor |

**Table 2: bookings**
| Column | Type | Description |
|--------|------|-------------|
| booking_id | UUID (PK) | Unique transaction ID |
| customer_id | INT (FK) | Links to customers table |
| check_in_date | DATE | Booking start date |
| check_out_date | DATE | Booking end date |
| amount_spent | FLOAT | Total booking cost |
| room_type | VARCHAR | Queen Bed, King Bed, Suite |
| booking_channel | VARCHAR | Direct Website, Agent, OTA, Walk-in |
| num_adults | INT | 1-4 adults |
| num_children | INT | 0-3 children |
| special_requests | VARCHAR | None, Pet Friendly, Accessible Room, Crib, High Floor |
| status | VARCHAR | Checked-Out, Cancelled, No-Show |

**Table 3: support_logs**
| Column | Type | Description |
|--------|------|-------------|
| interaction_id | UUID (PK) | Unique log ID |
| customer_id | INT (FK) | Links to customers table |
| date | TIMESTAMP | When interaction occurred |
| log_text | TEXT | Unstructured support text (for AI/NLP analysis) |

### Data Pipeline Features

**1. Historical Data Generation (One-Time)**
- Generates 4 years of synthetic booking history
- Creates 5,000 customers with realistic personas
- Produces ~110,000 bookings with varying patterns based on customer segment
- Generates ~4,500 support interaction logs (30% of customers)
- Run via: `python main.py` → Option 1

**Current Database Stats:**
- ✅ 5,000 customers
- ✅ 111,555 bookings
- ✅ 4,463 support logs

**2. Weekly Incremental Data (Scheduled)**
- Adds 7 days of new data each week
- Generates 50 new customers + 200 new bookings
- Implements 5-year rolling window (auto-deletes data older than 5 years)
- Maintains constant database size
- Run via: `python main.py` → Option 2
- **To be scheduled with Prefect in Week 2**

**3. Data Quality Checks (Pandera)**
Validation rules enforce:
- Booking dates logic (check-out > check-in)
- Price ranges ($0-$5000 per booking)
- Room types match allowed values (Queen Bed, King Bed, Suite)
- Stay duration is 1-30 nights
- No duplicate IDs
- Customer age range (18-100)
- Support logs have non-empty text
- All required fields are present

### Configuration Management

**config.yaml controls:**
- Number of customers/bookings to generate
- Customer segment definitions and behaviors
- Segment-specific churn thresholds (180/90/365 days)
- Cancellation rates (15%)
- Data quality validation rules

**Benefits:**
- Change data volume without touching code
- Adjust business rules in one place
- Easy experimentation with different scenarios
- All pipelines read from single source of truth

### Setup Instructions

**1. Clone Repository**
```bash
git clone https://github.com/Devanshi678/-Hotel-Churn.git
cd Hotel-Churn-Prediction
```

**2. Create Virtual Environment**
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows Git Bash
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Start Docker Database**
```bash
docker-compose up -d
```

**5. Run Historical Data Pipeline**
```bash
python main.py
# Select Option 1: Historical Data Generation
```

**6. (Weekly) Run Incremental Data Pipeline**
```bash
python main.py
# Select Option 2: Weekly Data Generation
```

### Data Generation Logic

**Business Travelers:**
- Book ~12 times per year (monthly)
- Stay 3 nights on average
- Higher proportion use Corporate Portal channel
- Prefer King Bed rooms
- Lower cancellation rate

**Vacation Travelers:**
- Book ~4 times per year (quarterly)
- Stay 7 nights on average
- Mix of Direct Website and OTA bookings
- Higher proportion request Cribs (families)
- Standard cancellation rate

**Occasional Visitors:**
- Book ~1 time per year
- Stay 5 nights on average
- Higher proportion are Walk-ins
- Random room preferences
- Standard cancellation rate

### Data Quality Metrics

**Achieved Quality:**
- 0% duplicate customer IDs
- 0% duplicate booking IDs
- 100% valid date ranges (check-in < check-out)
- 100% bookings within 1-30 night range
- 100% prices within $0-$5000 range
- 15% cancellation rate (as configured)
- 30% customers with support logs

---

## Next Steps

### Week 2: Feature Engineering & Model Training
- [ ] Calculate RFM (Recency, Frequency, Monetary) features
- [ ] Create rolling time-window aggregates
- [ ] Engineer engagement metrics from support logs
- [ ] Label churn based on segment-specific thresholds
- [ ] Implement time-aware train/test split
- [ ] Train baseline models (Logistic Regression, XGBoost)
- [ ] Handle class imbalance (SMOTE, class weights)

### Week 3: Advanced Modeling & Deployment
- [ ] Unsupervised clustering (K-Prototypes) for persona detection
- [ ] Hyperparameter tuning (Bayesian optimization)
- [ ] Neural network implementation
- [ ] API-based AI component integration
- [ ] Prefect workflow scheduling
- [ ] Model evaluation (Precision@K, ROI metrics)
- [ ] Documentation and GitHub repository finalization

---

## Current Status

✅ **COMPLETED (Week 1 - Day 1):**
- Project repository setup with Git/GitHub
- Docker PostgreSQL database configured
- Complete data generation pipeline with Faker
- Data quality validation with Pandera
- 4 years of historical data generated
- Database contains 5,000 customers, 111,555 bookings, 4,463 support logs
- Segment-specific churn definitions implemented
- 5-year rolling window data retention policy
- All code pushed to GitHub

🔄 **IN PROGRESS:**
- Feature engineering pipeline
- Churn labeling logic

⏳ **UPCOMING:**
- Model training and evaluation
- Prefect scheduling setup
- Production deployment

---

## Repository Information

**GitHub:** https://github.com/Devanshi678/-Hotel-Churn  
**Author:** Devanshikumari Patel  
**Started:** January 29, 2026  
**Last Updated:** February 24, 2026