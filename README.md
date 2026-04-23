# DS 4320 Project 2: Credit Card Fraud Detection

---

## Executive Summary

This repository contains a full data science pipeline for detecting credit card fraud using the document model. The project draws on two publicly available datasets — the IEEE-CIS Fraud Detection dataset (590,540 real e-commerce transactions) and PaySim (200,000 synthetic mobile money transactions) — merged and stored in a MongoDB Atlas cluster as a single `transactions` collection. The repository includes all code for data ingestion, database creation, exploratory analysis, and a machine learning classification pipeline, along with complete metadata, a data dictionary, and a press release explaining the project's findings and motivation.

---

**Name:** Amelia Vasiliu

**NetID:** ega9cw 

**DOI:** *[Add your Zenodo DOI here after creating it]*

**Press Release:** [Catching Fraud Before It Happens: Can an Algorithm Protect Your Credit Card?](https://github.com/ameliavasiliu/Project-2-Document-Model/blob/main/Press_Release.md)

**Pipeline:** [Code/pipeline.ipynb](Code/pipeline.ipynb)

**License:** MIT — [LICENSE](LICENSE)

## Problem Definition

### General and Specific Problem Statement

**General Problem:** Detecting credit card fraud.

**Refined Specific Problem:** Can we distinguish fraudulent credit card transactions from legitimate ones before they are approved? Despite fraud accounting for less than 0.2% of all transactions, the financial and personal harm it causes is significant. We aim to identify the key behavioral and transactional signals that separate fraud from legitimate purchases, and use those signals to build a classifier that catches fraud without incorrectly blocking real customers.

### Motivation

Credit card fraud is a massive and growing financial threat. According to the Nilson Report, global card fraud losses exceed $30 billion annually, with cardholders, banks, and merchants all bearing the cost. Traditional rule-based fraud detection systems struggle to keep up with increasingly sophisticated fraud patterns that evolve rapidly. A data-driven approach that learns from historical transaction patterns has the potential to catch fraud earlier, reduce false positives that frustrate legitimate customers, and adapt to new fraud strategies over time. Building and understanding such a pipeline is directly relevant to careers in fintech, banking, insurance, and data engineering.

### Rationale for Refinement

Credit card fraud is a broad problem that can be tackled from many angles — identifying stolen card numbers, detecting account takeovers, flagging suspicious merchant behavior, or auditing transactions after the fact. We refined our focus specifically to the moment of transaction approval: can we catch fraud before the charge goes through? This is the highest-value intervention point because once a fraudulent transaction is approved, the damage is done — the money is gone, the cardholder is harmed, and the bank must spend resources on chargebacks and disputes. By framing the problem as a real-time binary decision at the point of sale — approve or flag — we create a concrete, measurable objective with direct real-world impact. This refinement also makes the problem well-suited to a classification approach, where each transaction is an independent decision that must be made instantly using only the information available at that moment.

### Press Release Headline

[Catching Fraud Before It Happens: Can an Algorithm Protect Your Credit Card?](https://github.com/ameliavasiliu/Project-2-Document-Model/blob/main/Press_Release.md)

---

## Domain Exposition

### Terminology

| Term | Definition |
|------|------------|
| Fraud | An unauthorized transaction made using a stolen or compromised card |
| Legitimate Transaction | A transaction made by the actual cardholder with their consent |
| Chargeback | The process by which a bank reverses a fraudulent transaction and returns funds to the cardholder |
| Card-Not-Present Fraud | Fraud occurring in online purchases where the physical card is not required |
| Card-Present Fraud | Fraud occurring at a physical point of sale using a counterfeit or stolen card |
| False Positive | A legitimate transaction incorrectly flagged as fraudulent, blocking a real customer |
| False Negative | A fraudulent transaction that slips through undetected and is approved |
| Class Imbalance | The extreme rarity of fraud relative to legitimate transactions, typically less than 0.2% |
| Transaction Velocity | The rate at which transactions are made, often used as a signal for fraud |
| Authorization | The real-time approval process a transaction undergoes before being accepted |
| Recall | The proportion of actual fraud cases that the model successfully catches |
| Precision | Of all transactions flagged as fraud, the proportion that are actually fraudulent |

### Domain Overview

Credit card fraud detection operates within the financial services industry, specifically at the intersection of banking, payment processing, and consumer protection. When a cardholder makes a purchase, the transaction passes through a payment network — such as Visa or Mastercard — where it is routed to the issuing bank for authorization. In that fraction of a second, the bank must decide whether to approve or decline the charge. Fraud detection systems sit inside this authorization pipeline, analyzing signals from the transaction in real time to assess risk. Historically, these systems relied on hand-crafted rules — flag any transaction over a certain amount, or any purchase made in a foreign country within hours of a domestic one. Today, machine learning has largely replaced or augmented these rules, allowing banks to learn complex and evolving fraud patterns directly from historical transaction data. The stakes are high on both sides: missing a fraudulent transaction harms the cardholder, while incorrectly blocking a legitimate one damages customer trust and costs the bank business. This tension between recall and precision sits at the heart of every fraud detection system in production today.

### Background Reading

Folder Link: https://drive.google.com/drive/folders/1iev7pKJt8QqFFafOsGCl2OkG5yNseINi?usp=sharing

| Title | Description | Link |
|-------|-------------|------|
| Fraud In America 2025: The Consumer Threat Landscape | Forbes article examining the current state of consumer fraud in the U.S., including credit/debit card fraud trends, account takeover data from FinCEN, and underground fraud markets | https://drive.google.com/file/d/1Wrv5QjeGy-bqZtRZhjCo8H_FUgpkdAir/view?usp=sharing |
| Credit Card Fraud Detection using Machine Learning Algorithms | Peer-reviewed academic paper from Procedia Computer Science comparing multiple ML classifiers (Random Forest, Logistic Regression, SVM, Decision Tree) on the European credit card fraud dataset, with and without SMOTE oversampling | https://drive.google.com/file/d/1D1kbsB9d24LQJ493_LTBGnta2p3TvvpA/view?usp=sharing |
| 2025 Credit Card Fraud Statistics | Consumer survey report from Security.org finding that 62 million Americans experienced credit card fraud in the past year, with data on fraud habits, reporting behavior, and victim recovery rates | https://drive.google.com/file/d/13T6H0hIBsTAwm2qPDJcsSj_qzBz1UXnn/view?usp=sharing |
| Transforming Risk Management in Financial Services with Generative AI | Harvard Business Review sponsored article examining how generative AI is being used by financial institutions to modernize fraud detection, risk assessment, and compliance workflows | https://drive.google.com/file/d/11Sj_o-XqxuWtTNIdY4f1jtDw-kWErr40/view?usp=sharing |
| A Brief History of Credit Card Fraud | Article tracing the evolution of credit card fraud from its first recorded instance in 1899 through the internet age, EMV era, and rise of card-not-present fraud in the digital economy | https://drive.google.com/file/d/1camLgzix5PfsJh3CQnX5k49L6oBGXhJR/view?usp=sharing |

---

## Data Creation

### Data Acquisition (Provenance)

This project draws from two publicly available datasets hosted on Kaggle. The first is the IEEE-CIS Fraud Detection dataset, released by Vesta Corporation in partnership with the IEEE Computational Intelligence Society for a 2019 Kaggle competition. It contains 590,540 real-world e-commerce transactions collected over six months, with 431 features split across two files — `train_transaction.csv` and `train_identity.csv` — joined on `TransactionID`. The transaction file contains core payment fields including amount, card metadata, billing address, email domains, and Vesta's internal match flags M1 through M9. The identity file contains device type and device info, which are only available for a subset of transactions. Both files were downloaded as zip archives from Kaggle, uploaded to Google Drive, extracted in Google Colab, and loaded into pandas with column filtering to manage memory on the free tier. The two files were then merged on `TransactionID` using a left join to preserve all transactions regardless of whether identity information was available.

The second dataset is PaySim, a synthetic mobile money transaction dataset created by researchers at the NTNU using a private financial dataset from a real mobile money service in Africa as a simulation base. It contains 6.3 million simulated transactions across five transaction types: CASH_OUT, PAYMENT, CASH_IN, TRANSFER, and DEBIT. For this project, 200,000 rows were loaded to manage Colab RAM constraints. PaySim was chosen to complement IEEE-CIS because it includes account balance fields — origin and destination balances before and after each transaction — which are not present in IEEE-CIS and represent a distinct category of fraud signal.

Both datasets were inserted into a MongoDB Atlas cluster in a single `transactions` collection with a `source` field tagging each document as either `ieee_cis` or `paysim`.

### Code Table

| File | Description | Link |
|------|-------------|------|
| `ingestion.ipynb` | Mounts Google Drive, unzips raw data files, loads IEEE-CIS and PaySim CSVs with memory optimization, merges IEEE-CIS transaction and identity files on `TransactionID`, and inserts all documents into MongoDB Atlas in batches of 1000 using `bulk_write` | https://github.com/ameliavasiliu/Project-2/blob/main/Code/DataBase_Creation.ipynb |

### Bias Identification

Several sources of bias were introduced during data collection. The IEEE-CIS dataset is heavily class-imbalanced — only 3.5% of transactions are labeled fraud — meaning any model trained on raw counts will be biased toward predicting legitimate transactions and will miss a large proportion of fraud. This reflects real-world transaction distributions but makes model evaluation on accuracy alone misleading. The dataset is also geographically biased toward e-commerce transactions processed through Vesta Corporation's payment infrastructure, which serves primarily US and Western European merchants. Transaction patterns, fraud strategies, and card types common in other regions are underrepresented. PaySim introduces a different kind of bias: because it is a simulation built on one private African mobile money dataset, its fraud patterns were generated by a model rather than real criminals, meaning the fraud signatures in PaySim may not reflect actual adversarial behavior.

### Bias Mitigation

Class imbalance can be addressed at the modeling stage using SMOTE to oversample the minority fraud class, or by passing `class_weight='balanced'` to scikit-learn classifiers so the model penalizes missed fraud detections more heavily than missed legitimate transactions. Model performance should be reported using precision, recall, and AUC-ROC rather than accuracy, since accuracy is a misleading metric on imbalanced data. To mitigate the simulation bias introduced by PaySim, the final model should be trained and evaluated primarily on the IEEE-CIS data, with PaySim used as a supplementary source to increase document count and introduce account-balance features. The `source` field on every document makes it straightforward to filter by dataset at any stage of the pipeline.

### Rationale for Critical Decisions

The most significant judgment call in data creation was applying `usecols` to the IEEE-CIS load to skip the 300+ V-columns included in the raw file. These V-columns are Vesta's masked and anonymized internal features whose meaning is not publicly documented. Dropping them reduces memory usage by approximately 80%, which was necessary to prevent RAM exhaustion on Colab's free tier, and keeps the document schema focused on interpretable features. The tradeoff is that these columns likely carry predictive signal — published solutions to the original Kaggle competition used them extensively. For this project, interpretability and schema clarity take priority over maximum model performance, and the retained features are sufficient to build a functional classifier.

The second major decision was the left join when merging the IEEE-CIS transaction and identity files. A left join preserves all 590,540 transactions regardless of whether identity information was available, with missing device fields stored as null in MongoDB. An inner join would have dropped all transactions without identity records, reducing the dataset by roughly 85% and introducing selection bias since transactions with device information are likely skewed toward certain channels and merchants.

The third decision was capping PaySim at 200,000 rows rather than loading all 6.3 million. This was a practical constraint imposed by Colab free tier RAM limits. 200,000 rows was chosen because it is large enough to provide meaningful representation of all five transaction types and both fraud patterns (CASH_OUT and TRANSFER) while keeping total insertion time under one hour.

---

## Metadata

### Implicit Schema Guidelines

All documents in the `transactions` collection share three top-level fields: `source` (string), `is_fraud` (boolean), and `transaction` (sub-document). Beyond these three fields, document structure diverges by source. IEEE-CIS documents additionally contain `card`, `address`, `email`, `device`, and `match_flags` sub-documents. PaySim documents additionally contain an `account` sub-document. No field outside the three shared fields is guaranteed to exist in every document, and no field is guaranteed to be non-null even when present. For example, `device.type` is null for IEEE-CIS transactions that had no matching identity record. This divergence is intentional and represents the implicit schema design: the application enforces consistency within each source type, and the `source` field allows downstream queries to apply the correct expectations for each document type.

#### IEEE-CIS Document Structure
```json
{
  "source":      "ieee_cis",
  "is_fraud":    false,
  "transaction": { "id", "amount", "dt", "product_cd" },
  "card":        { "card1", "card2", "card3", "card4", "card5", "card6" },
  "address":     { "billing_region", "billing_country" },
  "email":       { "purchaser", "recipient" },
  "device":      { "type", "info" },
  "match_flags": { "M1" – "M9" }
}
```

#### PaySim Document Structure
```json
{
  "source":      "paysim",
  "is_fraud":    false,
  "transaction": { "amount", "type", "step" },
  "account":     { "origin_balance_before", "origin_balance_after",
                   "dest_balance_before",   "dest_balance_after" }
}
```

### Data Summary

| Property | Value |
|----------|-------|
| Total documents | 823,540 |
| IEEE-CIS documents | 623,540 |
| PaySim documents | 200,000 |
| Fraudulent transactions | 21,905 |
| Legitimate transactions | 801,635 |
| Overall fraud rate | 2.66% |
| Collections | 1 (`transactions`) |
| Database | `fraud_db` |
| Cluster | `credit-card-fraud.cmgs8u1.mongodb.net` |

### Data Dictionary

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `source` | string | Dataset origin tag | `"ieee_cis"` |
| `is_fraud` | boolean | Whether the transaction was fraudulent | `true` |
| `transaction.id` | integer | Unique transaction ID (IEEE-CIS only) | `3312000` |
| `transaction.amount` | float | Transaction amount in USD | `29.0` |
| `transaction.dt` | integer | Timedelta from reference date (IEEE-CIS only) | `8023328` |
| `transaction.product_cd` | string | Product code (IEEE-CIS only) | `"W"` |
| `transaction.type` | string | Transaction type (PaySim only) | `"CASH_OUT"` |
| `transaction.step` | integer | Hour of simulation (PaySim only) | `1` |
| `card.card4` | string | Card network | `"visa"` |
| `card.card6` | string | Card category | `"debit"` |
| `address.billing_region` | float | Billing region code | `315.0` |
| `address.billing_country` | float | Billing country code | `87.0` |
| `email.purchaser` | string | Purchaser email domain | `"gmail.com"` |
| `email.recipient` | string | Recipient email domain | `"gmail.com"` |
| `device.type` | string | Device category (IEEE-CIS only) | `"mobile"` |
| `device.info` | string | Device model string (IEEE-CIS only) | `"moto x4 Build/NPW26.83-18-2-0-4"` |
| `match_flags.M1–M9` | string | Vesta internal match flags (IEEE-CIS only) | `"T"` |
| `account.origin_balance_before` | float | Sender balance before transaction (PaySim only) | `181.0` |
| `account.origin_balance_after` | float | Sender balance after transaction (PaySim only) | `0.0` |
| `account.dest_balance_before` | float | Receiver balance before transaction (PaySim only) | `0.0` |
| `account.dest_balance_after` | float | Receiver balance after transaction (PaySim only) | `0.0` |

### Data Dictionary: Uncertainty Quantification

All statistics are computed from 823,540 documents in the `transactions` collection, split by `is_fraud` status. Standard deviation is the primary measure of uncertainty. Features with large std dev differences between legitimate and fraudulent classes are the strongest candidates for the fraud detection model.

#### IEEE-CIS Numerical Features (623,540 documents)

**transaction.amount**

| Metric | Legitimate | Fraudulent |
|--------|-----------|------------|
| Count | 601,782 | 21,758 |
| Mean | 134.58 | 150.11 |
| Std Dev | 239.03 | 232.21 |
| Min | 0.25 | 0.29 |
| 25% | 44.00 | 35.35 |
| Median | 68.50 | 75.00 |
| 75% | 120.50 | 161.78 |
| Max | 31,937.39 | 5,191.00 |

Fraudulent transactions have a slightly higher mean and elevated 75th percentile, suggesting fraud skews toward mid-to-high value purchases. High std dev on both classes reflects wide variability in transaction size.

**transaction.dt** (timedelta from reference date)

| Metric | Legitimate | Fraudulent |
|--------|-----------|------------|
| Count | 601,782 | 21,758 |
| Mean | 7,416,837 | 7,726,597 |
| Std Dev | 4,506,280 | 4,307,238 |
| Min | 86,400 | 89,760 |
| Median | 7,755,852 | 7,947,641 |
| Max | 15,811,131 | 15,810,876 |

This is a timedelta from an undisclosed reference date, not a real timestamp. Absolute values are not interpretable; use only as a relative temporal feature. Std dev is large on both classes reflecting the full six-month span of the dataset.

**card.card3** (issuing bank country)

| Metric | Legitimate | Fraudulent |
|--------|-----------|------------|
| Count | 600,254 | 21,719 |
| Mean | 152.84 | 162.29 |
| Std Dev | 10.86 | 17.48 |
| Min | 100.00 | 100.00 |
| Median | 150.00 | 150.00 |
| Max | 231.00 | 231.00 |

Fraudulent transactions have a notably higher std dev (17.48 vs 10.86), indicating fraud is more geographically dispersed across issuing banks than legitimate transactions. This is a useful discriminating feature.

**address.billing_country**

| Metric | Legitimate | Fraudulent |
|--------|-----------|------------|
| Count | 540,928 | 13,649 |
| Mean | 86.82 | 86.30 |
| Std Dev | 2.57 | 4.87 |
| Min | 13.00 | 10.00 |
| Median | 87.00 | 87.00 |
| Max | 102.00 | 96.00 |

The std dev for fraudulent transactions is nearly double that of legitimate ones (4.87 vs 2.57). Despite similar medians, fraud is spread across more billing countries, making this a strong fraud signal despite low absolute variance.

**card.card1, card.card2, card.card5** (anonymized card identifiers)

| Field | Legit Mean | Fraud Mean | Legit Std Dev | Fraud Std Dev |
|-------|-----------|------------|---------------|---------------|
| card1 | 9,908.89 | 9,545.22 | 4,902.80 | 4,750.05 |
| card2 | 362.10 | 364.63 | 157.67 | 159.70 |
| card5 | 199.54 | 192.20 | 41.03 | 45.65 |

These are anonymized numerical identifiers. Magnitude has no inherent meaning and std dev does not reflect a measurable physical quantity. Treat as categorical features in modeling despite being stored as floats.

**address.billing_region**

| Metric | Legitimate | Fraudulent |
|--------|-----------|------------|
| Count | 540,928 | 13,649 |
| Mean | 290.62 | 293.93 |
| Std Dev | 101.71 | 103.74 |
| Median | 299.00 | 299.00 |
| Max | 540.00 | 536.00 |

Similar distributions between classes. Low discriminating power on its own but may be useful in combination with `billing_country`.

#### PaySim Numerical Features (200,000 documents)

**transaction.amount**

| Metric | Legitimate | Fraudulent |
|--------|-----------|------------|
| Count | 199,853 | 147 |
| Mean | 180,476.44 | 635,893.20 |
| Std Dev | 326,487.53 | 1,522,141.80 |
| Min | 0.32 | 164.00 |
| 25% | 12,015.01 | 13,707.11 |
| Median | 68,726.58 | 43,092.00 |
| 75% | 229,040.46 | 361,559.69 |
| Max | 6,419,835.27 | 10,000,000.00 |

The strongest numerical fraud signal in the dataset. Fraudulent transactions average 3.5x more than legitimate ones. The std dev for fraud is nearly 5x higher, reflecting extreme variability — fraud ranges from small test transactions to the dataset maximum of 10,000,000.

**account.origin_balance_after**

| Metric | Legitimate | Fraudulent |
|--------|-----------|------------|
| Count | 199,853 | 147 |
| Mean | 900,839.02 | 22,947.86 |
| Std Dev | 2,804,680.93 | 244,194.47 |
| Median | 0.00 | 0.00 |
| Max | 38,946,233.02 | 2,930,418.44 |

The most reliable low-uncertainty fraud signal in PaySim. Fraudulent transfers drain the origin account to zero in nearly all cases — the median is 0 and the mean is only 22,948 vs 900,839 for legitimate transactions. Low std dev among fraud cases confirms this is a consistent pattern, not noise.

**account.origin_balance_before**

| Metric | Legitimate | Fraudulent |
|--------|-----------|------------|
| Count | 199,853 | 147 |
| Mean | 882,381.96 | 628,992.79 |
| Std Dev | 2,766,908.58 | 1,660,480.84 |
| Median | 19,486.34 | 29,707.86 |
| Max | 38,939,424.03 | 12,930,418.44 |

Fraudulent accounts tend to have lower balances before the transaction than legitimate ones on average. High std dev on both classes reflects the wide range of account sizes in mobile money.

**account.dest_balance_before and dest_balance_after**

| Field | Legit Mean | Fraud Mean | Legit Std Dev | Fraud Std Dev |
|-------|-----------|------------|---------------|---------------|
| dest_balance_before | 941,689 | 220,507 | 2,373,587 | 1,182,831 |
| dest_balance_after | 1,192,238 | 686,499 | 2,655,656 | 1,947,014 |

Destination balances are lower for fraudulent transactions both before and after, suggesting fraud targets accounts with less existing funds. High std dev on both classes limits the usefulness of these fields as standalone features but they contribute signal in combination with origin balance fields.
