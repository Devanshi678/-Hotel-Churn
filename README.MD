# Title : Hotel-Churn-Prediction-E2E

Date : 1/29/26 | Name : Devanshikumari Patel 



### Project Summary

● Problem:

"Silent Churn" in hotels where regular guests stop booking without warning. Compounding this, generic retention strategies fail because they treat "Business Regulars" and "Seasonal Vacationers" exactly the same.

● Why it matters / impact:

Acquiring new guests costs 5x-25x more than retaining existing ones. Reducing churn by 5% can increase profits by 25-95%.Furthermore, misidentifying churn (e.g., flagging a seasonal guest as "churned" in the off-season) leads to wasted marketing budget on unnecessary discounts.

● Proposed approach:A Hybrid ML Pipeline.

1. Unsupervised Learning (K-Prototypes): To auto-segment customers into behavioral personas (e.g., "Corporate Traveler," "Summer Vacationer") based on spending, payment, and stay patterns.
2. Supervised Learning (XGBoost): Uses these personas as "context features" alongside historical data to accurately predict churn probability.

● Primary success metric:

Precision @ Top-20% (Ensuring we only target high-probability churners) and Segmented Retention ROI.

● Key risks / unknowns:

"Cluster Stability"—if customer behavior changes rapidly, the personas (clusters) might become invalid and need re-training.

##### 

##### Scoping

* Business problem : Hotels are currently facing a "Leaky Bucket" problem with a blindfold on. They only realize a customer has left months after the fact. Worse, they lack context: they might panic because a "Summer Vacationer" hasn't visited in November, or ignore a "Weekly Business Traveler" who missed two weeks. They need a system that understands who the customer is to correctly predict if they are leaving.



* Goal  :

1. Segmentation: Automatically group the customer base into 3-5 distinct behavioral personas using unsupervised learning.
2. Prediction: Predict churn within a 90-day window with >75% precision for high-risk segments.
3. Actionability: Deliver a weekly "Target List" that tells Marketing who is at risk and what type of customer they are (to tailor the offer).



* Stakeholders:

1. Project Lead: \[Your Name] (End-to-end Architecture \& Dev).
2. Hotel Operations/Franchise Owner: (Consumer of the "At Risk" reports).
3. Marketing Team: (Executes segmented campaigns: Upgrades for Business, Discounts for Vacationers).



* Prior Work

Current State: Manual "Recency Rules" (e.g., "If last\_booking > 365 days, send email").

Flaws: This ignores seasonality. A customer who only visits in July will always be flagged as "churned" in December, causing spam and brand fatigue.



* Online Metrics (Real-world business impact)

Campaign Conversion Rate: % of targeted guests who book within 30 days of receiving an offer.

Retention ROI: (Revenue from Saved Guests) - (Cost of Incentives).



* Offline Metrics (Model performance)

Silhouette Score: To measure how well-separated our Unsupervised Clusters are.

Precision @ Top-K: To ensure the churn model is accurate specifically on the highest-risk customers we intend to contact.

F1-Score (Macro): To ensure the model performs well across all different customer clusters, not just the majority group.



* Online vs. Batch

Mode: Batch Prediction (Weekly).

Reasoning: Customer habits change slowly. Re-clustering and Churn prediction can be run every Monday morning to generate the weekly marketing list. Real-time inference is overkill and costly.



* Requirements

Cost: < $10/month (Local/Colab training, free-tier tools).

Interpretability: The clusters must be explainable to business owners (e.g., "Cluster 1 is clearly Business Travelers because they pay with Corporate Cards and stay weekdays").

Pipeline: Fully automated via YAML configs and GitHub Actions (optional).



* Project Risks

Cluster Drift: If the hotel changes pricing or strategy, the "Personas" might change, requiring the unsupervised model to be retrained.

Data Leakage: Using future booking info to define current clusters. (Mitigation: Strict time-based splitting).

Cold Start: New customers with only 1 booking are hard to cluster and hard to predict.



* Target Launch Dates

Week 1: Data Ingestion, Cleaning, and Unsupervised Clustering (MVP).

Week 2: Feature Engineering \& Supervised Churn Model.

Week 3: Pipeline Orchestration, Evaluation, and Documentation, deployment.





