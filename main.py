import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# 1. GITHUB 

def analyze_github():
    print("="*60)
    print("STARTING GITHUB CANDIDATE ANALYSIS")
    print("="*60)
    
    try:
        df = pd.read_csv('github_candidates_1.csv')
    except FileNotFoundError:
        print("Error: 'github_candidates_1.csv' not found. Skipping GitHub analysis.\n")
        return

    # Cleaning
    df['dominant_language'] = df['dominant_language'].str.strip()
    
    # Aggregation
    candidates = df.groupby('username').agg({
        'public_repos': 'sum',
        'total_stars': 'sum',
        'total_forks': 'sum',
        'commits_12m': 'sum',
        'has_cicd': 'max'
    }).reset_index()
    
    # Outlier Handling (Log Scaling)
    candidates['log_stars'] = np.log1p(candidates['total_stars'])
    candidates['log_forks'] = np.log1p(candidates['total_forks'])
    
    # ML Scaling
    scaler = MinMaxScaler()
    cols_to_scale = ['public_repos', 'log_stars', 'log_forks', 'commits_12m', 'has_cicd']
    scaled_data = scaler.fit_transform(candidates[cols_to_scale])
    df_s = pd.DataFrame(scaled_data, columns=cols_to_scale)

    # Scoring Logic
    candidates['Developer_Score'] = (df_s['commits_12m'] * 0.7) + (df_s['public_repos'] * 0.3)
    candidates['Senior_Developer_Score'] = (df_s['log_stars'] * 0.6) + (df_s['public_repos'] * 0.4)
    candidates['Solution_Architect_Score'] = (df_s['log_forks'] * 0.5) + (df_s['has_cicd'] * 0.4) + (df_s['public_repos'] * 0.1)

    # Reasoning Logic
    def get_github_reason(role, row):
        if role == "Developer":
            return (f"Selected for high shipping frequency. Contributed {int(row['commits_12m'])} commits "
                    f"across {int(row['public_repos'])} repositories.")
        if role == "Senior Developer":
            return (f"Selected for community authority. Work has earned {int(row['total_stars'])} stars, "
                    f"indicating industry trust.")
        if role == "Solution Architect":
            status = "Integrated" if row['has_cicd'] == 1 else "Not Found"
            return (f"Selected for architectural reuse. Systems forked {int(row['total_forks'])} times "
                    f"with CI/CD status: {status}.")

    # Output
    roles = [("Developer", "Developer_Score"), 
             ("Senior Developer", "Senior_Developer_Score"), 
             ("Solution Architect", "Solution_Architect_Score")]

    for role_name, score_col in roles:
        print(f"\n--- TOP 3 {role_name.upper()} (GitHub) ---")
        top_3 = candidates.sort_values(score_col, ascending=False).head(3)
        for i, (_, row) in enumerate(top_3.iterrows()):
            print(f"{i+1}. {row['username']}")
            print(f"   BASIS: {get_github_reason(role_name, row)}")
    print("\n")

# 2. KAGGLE 

def analyze_kaggle():
    print("="*60)
    print("STARTING KAGGLE CANDIDATE ANALYSIS")
    print("="*60)

    try:
        df = pd.read_csv('kaggle-preprocessed.csv', index_col=0)
    except FileNotFoundError:
        print("Error: 'kaggle-preprocessed.csv' not found. Skipping Kaggle analysis.\n")
        return

    # Helper: Standardize Size
    def standardize_size_to_mb(val):
        val = str(val).upper()
        try:
            if 'GB' in val: return float(val.replace('GB','').strip()) * 1024
            if 'KB' in val: return float(val.replace('KB','').strip()) / 1024
            if 'B' in val: return 0.01 
            return float(val.replace('MB','').strip())
        except: return 0.0

    df['size_mb'] = df['size'].apply(standardize_size_to_mb)

    # Clean and Aggregate
    medal_map = {'Gold': 5, 'Silver': 3, 'Bronze': 1, 'No Medal': 0}
    df['medal_points'] = df['Medals'].map(medal_map).fillna(0)
    df['Usability'] = df['Usability'].fillna(df['Usability'].median())

    authors = df.groupby('Author_name').agg({
        'Dataset_name': 'count',
        'No_of_files': 'sum',
        'Upvotes': 'sum',
        'Usability': 'mean',
        'medal_points': 'sum',
        'size_mb': 'sum'
    }).reset_index()

    # Transformations
    authors['log_upvotes'] = np.log1p(authors['Upvotes'])
    authors['log_size'] = np.log1p(authors['size_mb'])

    # ML Scaling
    scaler = MinMaxScaler()
    cols_to_scale = ['Dataset_name', 'No_of_files', 'log_upvotes', 'Usability', 'medal_points', 'log_size']
    authors_scaled = pd.DataFrame(scaler.fit_transform(authors[cols_to_scale]), columns=cols_to_scale)

    # Scoring Logic
    authors['Developer_Score'] = (authors_scaled['Usability'] * 0.7) + (authors_scaled['Dataset_name'] * 0.3)
    authors['Senior_Developer_Score'] = (authors_scaled['medal_points'] * 0.6) + (authors_scaled['log_upvotes'] * 0.4)
    authors['Solution_Architect_Score'] = (authors_scaled['No_of_files'] * 0.4) + (authors_scaled['log_size'] * 0.4) + (authors_scaled['Usability'] * 0.2)

    # Reasoning Logic
    def get_kaggle_reason(role, row):
        if role == "Developer":
            return (f"Selected for high documentation standards. Avg Usability: {row['Usability']:.1f}/10 "
                    f"across {int(row['Dataset_name'])} projects.")
        if role == "Senior Developer":
            return (f"Selected for peer vetting. Earned {int(row['medal_points'])} medal points "
                    f"and {int(row['Upvotes'])} peer upvotes.")
        if role == "Solution Architect":
            size_val = row['size_mb'] / 1024 if row['size_mb'] > 1024 else row['size_mb']
            unit = "GB" if row['size_mb'] > 1024 else "MB"
            return (f"Selected for scale mastery. Managed {int(row['No_of_files'])} files "
                    f"across {size_val:.1f} {unit} of data.")

    # Output
    roles = [("Developer", "Developer_Score"), 
             ("Senior Developer", "Senior_Developer_Score"), 
             ("Solution Architect", "Solution_Architect_Score")]

    for role_label, score_col in roles:
        print(f"\n--- TOP 3 {role_label.upper()} (Kaggle) ---")
        top_3 = authors.sort_values(score_col, ascending=False).head(3)
        for i, (_, row) in enumerate(top_3.iterrows()):
            print(f"{i+1}. {row['Author_name']}")
            print(f"   BASIS: {get_kaggle_reason(role_label, row)}")
    print("\n")


# 3. STACKOVERFLOW 

def analyze_stackoverflow():
    print("="*60)
    print("STARTING STACKOVERFLOW CANDIDATE ANALYSIS")
    print("="*60)

    try:
        df = pd.read_csv('stackoverflow_200.csv')
    except FileNotFoundError:
        print("Error: 'stackoverflow_200.csv' not found. Skipping SO analysis.\n")
        return

    # Cleaning
    df['top_tags'] = df['top_tags'].fillna('general')

    # Outlier Capping (95th percentile)
    cols_to_cap = ['reputation', 'total_answer_score_fetched', 'avg_score_per_answer']
    for col in cols_to_cap:
        limit = df[col].quantile(0.95)
        df[col] = np.where(df[col] > limit, limit, df[col])

    # Feature Engineering
    df['tag_breadth'] = df['top_tags'].apply(lambda x: len(x.split(',')))
    arch_keywords = ['azure', 'aws', 'kubernetes', 'sql-server', 'database', 'asynchronous', 'git-submodules']
    df['arch_keywords_count'] = df['top_tags'].apply(lambda x: sum(1 for kw in arch_keywords if kw in x.lower()))

    # ML Scaling
    scaler = MinMaxScaler()
    metrics = ['reputation', 'accepted_answer_ratio', 'avg_score_per_answer', 'tag_breadth', 'arch_keywords_count']
    df_scaled = pd.DataFrame(scaler.fit_transform(df[metrics]), columns=metrics)

    # Scoring Logic
    df['Developer_Score'] = (df_scaled['accepted_answer_ratio'] * 0.7) + (df_scaled['avg_score_per_answer'] * 0.3)
    df['Senior_Developer_Score'] = (df_scaled['reputation'] * 0.6) + (df_scaled['avg_score_per_answer'] * 0.4)
    df['Solution_Architect_Score'] = (df_scaled['arch_keywords_count'] * 0.5) + (df_scaled['tag_breadth'] * 0.3) + (df_scaled['reputation'] * 0.2)

    # Reasoning Logic
    def get_so_reason(role, row):
        if role == "Developer":
            return f"High execution reliability. {row['accepted_answer_ratio']:.0%} acceptance rate for code solutions."
        if role == "Senior Developer":
            return f"Established technical authority. {int(row['reputation'])}+ reputation points via peer validation."
        if role == "Solution Architect":
            # UPDATED: Removed slicing [:40] so it prints the full list of tags
            return f"Vast technical breadth. Specialized in {row['tag_breadth']} domains. Key tags: {row['top_tags']}"

    # Output
    roles = [("Developer", "Developer_Score"), 
             ("Senior Developer", "Senior_Developer_Score"), 
             ("Solution Architect", "Solution_Architect_Score")]

    for role_label, score_col in roles:
        print(f"\n--- TOP 3 {role_label.upper()} (StackOverflow) ---")
        top_3 = df.sort_values(score_col, ascending=False).head(3)
        for i, (_, row) in enumerate(top_3.iterrows()):
            print(f"{i+1}. {row['display_name']}")
            print(f"   BASIS: {get_so_reason(role_label, row)}")
    print("\n")

if __name__ == "__main__":
    analyze_github()
    analyze_kaggle()
    analyze_stackoverflow()