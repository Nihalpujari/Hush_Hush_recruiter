# hushHush Recruiter
### Automated Candidate Selection System

**Course:** Advanced Python  
**Project Type:** Final Exam Project  

**Team:**
- Namrata Bhoyar
- Nihal Pujari
- Anuj Kamble
- Pramodkumar Shivanna

---

## Project Overview
hushHush Recruiter is an automated system that identifies and ranks software engineering candidates using public technical signals from platforms like **GitHub, StackOverflow, and Kaggle**.

The system converts these signals into a **final candidate score** and automatically assigns a role:

- Developer
- Senior Developer
- Solution Architect

It also provides a **hiring manager dashboard** for candidate review.

---

## Problem Statement
Doodle currently relies on recruiting agencies, which leads to:

- Manual CV screening
- High costs
- Slow hiring cycles
- Inconsistent candidate quality

The goal is to build an **automated, data-driven candidate selection system** using public developer activity.

---

## Data Sources

### GitHub
- Total stars
- Total forks
- Commits in last 12 months  
→ Measures coding activity and impact

### StackOverflow
- Reputation
- Average answer score
- Number of tags  
→ Reflects knowledge and problem-solving ability

### Kaggle
- Upvotes
- Usability
- Number of files  
→ Indicates data science activity

---

## System Pipeline

Data Sources (GitHub, SO, Kaggle)
↓
Feature Extraction
↓
Percentile Normalization
↓
Feature Scaling
↓
K-Means Clustering (2 stages)
↓
Candidate Scoring
(Random Forest + Linear Regression)
↓
Final Score
↓
Role Assignment
↓
Hiring Manager Dashboard


---

## Clustering Approach

The system uses a **two-stage K-Means process**:

### 1. Tiering (k = 2)
- Separates weak and strong candidates

### 2. Role Profiling (k = 3)
Clusters strong candidates into:
- Developer
- Senior Developer
- Solution Architect

---

## Scoring Model

Two models are combined:

- **Random Forest** → estimates candidate strength
- **Linear Regression** → measures cluster fit

### Final Score
Final Score = weighted(RF score + cluster-fit score)


Score range:
0 → weakest candidate
1 → strongest candidate


---

## Hiring Manager Dashboard

The dashboard allows:

- Viewing candidates
- Checking platform signals
- Seeing predicted role
- Comparing final scores
- Identifying top candidates per role

---

## How to Run

### Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib

