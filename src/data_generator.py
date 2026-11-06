"""
CI/CD Build Dataset Generator based on TravisTorrent empirical metrics.
Simulates realistic Continuous Integration build histories for Build-Failure Prediction.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_ci_dataset(num_samples: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic yet empirically grounded CI build data.
    Metrics reflect real-world software repository mining (Travis CI / GitHub Actions).
    """
    np.random.seed(random_state)
    
    start_time = datetime(2025, 1, 1, 8, 0, 0)
    current_time = start_time
    
    records = []
    
    # State tracking across builds (temporal context)
    prev_build_status = 0 # 0 = Pass, 1 = Fail
    consecutive_failures = 0
    
    # Pool of simulated developers with varying experience and reliability
    num_devs = 40
    dev_pool = []
    for d in range(num_devs):
        exp = int(np.random.lognormal(mean=3.5, sigma=1.0)) + 5
        base_err = np.clip(np.random.normal(loc=0.15, scale=0.06), 0.04, 0.40)
        dev_pool.append({
            'dev_id': f"dev_{d:02d}",
            'experience': exp,
            'base_failure_rate': float(base_err)
        })
        
    branches = ['main', 'feature', 'hotfix', 'release', 'refactor']
    branch_weights = [0.35, 0.42, 0.08, 0.08, 0.07]
    
    for i in range(1, num_samples + 1):
        # Time progression (between 10 mins and 6 hours)
        delta_min = int(np.random.exponential(scale=55)) + 5
        current_time += timedelta(minutes=delta_min)
        
        # Developer selection
        dev = np.random.choice(dev_pool)
        
        branch = np.random.choice(branches, p=branch_weights)
        is_pr = 1 if branch in ['feature', 'hotfix', 'refactor'] and np.random.rand() > 0.10 else (1 if np.random.rand() > 0.65 else 0)
        
        # Commit characteristics
        num_commits = np.random.geometric(p=0.40)
        files_changed = np.random.geometric(p=0.22)
        
        # Code churn
        lines_added = int(np.random.exponential(scale=35) * files_changed) + 2
        lines_deleted = int(np.random.exponential(scale=18) * files_changed)
        churn_total = lines_added + lines_deleted
        
        # Source vs Test churn
        test_ratio_chance = np.random.beta(a=2, b=4)
        has_tests = (np.random.rand() > 0.30)
        test_churn = int(churn_total * test_ratio_chance) if has_tests else 0
        src_churn = max(1, churn_total - test_churn)
        test_to_src_ratio = round(test_churn / (src_churn + 1e-5), 4)
        
        team_size = min(num_devs, max(5, int(np.random.normal(loc=20, scale=4))))
        
        # Committer stats
        committer_exp = dev['experience'] + int(i / 120)
        committer_fail_rate = round(dev['base_failure_rate'] + np.random.normal(0, 0.02), 4)
        committer_fail_rate = float(np.clip(committer_fail_rate, 0.02, 0.50))
        
        # Logit calculation for failure probability
        # Base logit tuned for realistic ~22-25% baseline failure rate
        logit = -2.10
        
        # Effect of previous failure (strongest signal in empirical literature)
        if prev_build_status == 1:
            logit += 1.30 + (0.30 * min(consecutive_failures, 3))
        else:
            logit -= 0.20
            
        # Code churn impact: larger changes introduce more defects
        if churn_total > 600:
            logit += 0.95
        elif churn_total > 250:
            logit += 0.50
        elif churn_total < 30:
            logit -= 0.35
            
        # Lack of tests increases failure risk
        if test_to_src_ratio < 0.05 and src_churn > 40:
            logit += 0.80
        elif test_to_src_ratio > 0.40:
            logit -= 0.60
            
        # Files changed breadth
        if files_changed > 8:
            logit += 0.55
        elif files_changed <= 2:
            logit -= 0.30
            
        # Committer experience attenuation
        if committer_exp < 15:
            logit += 0.60
        elif committer_exp > 75:
            logit -= 0.45
            
        # Committer historical failure tendency
        logit += (committer_fail_rate - 0.18) * 3.0
        
        # Branch sensitivity
        if branch == 'hotfix':
            logit += 0.55
        elif branch == 'release':
            logit -= 0.45
            
        # Sigmoid probability
        prob = 1.0 / (1.0 + np.exp(-logit))
        build_failed = 1 if np.random.rand() < prob else 0
        
        records.append({
            'build_id': f"B-{i:05d}",
            'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'git_branch': branch,
            'is_pull_request': is_pr,
            'num_commits': num_commits,
            'files_changed': files_changed,
            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'churn_total': churn_total,
            'src_churn': src_churn,
            'test_churn': test_churn,
            'test_to_src_ratio': test_to_src_ratio,
            'time_since_last_build_min': delta_min,
            'committer_experience': committer_exp,
            'committer_recent_failure_rate': committer_fail_rate,
            'team_size': team_size,
            'prev_build_status': prev_build_status,
            'consecutive_failures': consecutive_failures,
            'build_failed': build_failed
        })
        
        # State transition for next build
        if build_failed == 1:
            consecutive_failures += 1
            prev_build_status = 1
        else:
            consecutive_failures = 0
            prev_build_status = 0
        
    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    df = generate_ci_dataset(num_samples=6000, random_state=42)
    output_path = os.path.join('data', 'ci_builds_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} CI builds at {output_path}")
    print(f"Build Failure Rate: {df['build_failed'].mean():.2%}")
    print(df['build_failed'].value_counts())
