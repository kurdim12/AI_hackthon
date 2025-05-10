import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_process_data(file_path):
    """Load and preprocess transaction data"""
    # Read CSV
    df = pd.read_csv(file_path)
    
    # Convert date string to datetime
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%d/%m/%Y %H:%M')
    
    # Extract time features
    df['hour'] = df['transaction_date'].dt.hour
    df['day_of_week'] = df['transaction_date'].dt.day_name()
    df['date'] = df['transaction_date'].dt.date
    df['week'] = df['transaction_date'].dt.isocalendar().week
    df['month'] = df['transaction_date'].dt.month
    
    # Add derived metrics
    df['is_failed'] = df['transaction_status'] == 'Failed'
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    
    return df

def get_key_metrics(df):
    """Calculate key performance metrics with exact precision"""
    total_transactions = len(df)
    failed_transactions = int(df['is_failed'].sum())
    
    metrics = {
        'total_transactions': total_transactions,
        'failed_transactions': failed_transactions,
        'failure_rate': float((failed_transactions / total_transactions) * 100) if total_transactions > 0 else 0.0,
        'total_amount': float(df['transaction_amount'].sum()),
        'failed_amount': float(df[df['is_failed']]['transaction_amount'].sum()),
        'avg_transaction': float(df['transaction_amount'].mean()),
        'total_tax': float(df['tax_amount'].sum()),
        'unique_branches': int(df['branch_name'].nunique()),
        'unique_malls': int(df['mall_name'].nunique())
    }
    
    # Ensure precision
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, 2)
    
    return metrics

def get_branch_analytics(df):
    """Analyze performance by branch with exact calculations"""
    branch_stats = df.groupby('branch_name').agg({
        'transaction_id': 'count',
        'transaction_amount': 'sum',
        'tax_amount': 'sum',
        'is_failed': 'sum'
    }).round(2)
    
    branch_stats.columns = ['transaction_count', 'total_amount', 'total_tax', 'failed_count']
    
    # Calculate failure rate with proper precision
    branch_stats['failure_rate'] = (branch_stats['failed_count'] / branch_stats['transaction_count'] * 100).round(2)
    branch_stats['success_count'] = branch_stats['transaction_count'] - branch_stats['failed_count']
    branch_stats['avg_transaction'] = (branch_stats['total_amount'] / branch_stats['transaction_count']).round(2)
    
    branch_stats = branch_stats.sort_values('failure_rate', ascending=False)
    
    return branch_stats

def get_time_patterns(df):
    """Analyze time-based patterns with exact counts"""
    patterns = {
        'hourly_failures': df[df['is_failed']].groupby('hour').size(),
        'hourly_total': df.groupby('hour').size(),
        'hourly_rate': (df.groupby('hour')['is_failed'].mean() * 100).round(2),
        'daily_failures': df[df['is_failed']].groupby('day_of_week').size(),
        'daily_total': df.groupby('day_of_week').size(),
        'daily_rate': (df.groupby('day_of_week')['is_failed'].mean() * 100).round(2),
        'peak_failure_hour': df[df['is_failed']]['hour'].mode()[0] if len(df[df['is_failed']]) > 0 else 0,
        'peak_failure_day': df[df['is_failed']]['day_of_week'].mode()[0] if len(df[df['is_failed']]) > 0 else 'None'
    }
    
    # Add more detailed patterns
    patterns['hourly_pattern'] = df.groupby('hour').agg({
        'transaction_id': 'count',
        'is_failed': ['sum', 'mean']
    })
    patterns['hourly_pattern'].columns = ['total', 'failures', 'failure_rate']
    patterns['hourly_pattern']['failure_rate'] = (patterns['hourly_pattern']['failure_rate'] * 100).round(2)
    
    return patterns

def detect_anomalies(df):
    """Detect unusual patterns with precise thresholds"""
    anomalies = []
    
    # Recent failure spike detection
    current_time = datetime.now()
    recent = df[df['transaction_date'] >= current_time - timedelta(days=7)]
    
    if len(recent) > 0 and len(df) > 0:
        recent_failure_rate = recent['is_failed'].mean() * 100
        overall_failure_rate = df['is_failed'].mean() * 100
        
        if recent_failure_rate > overall_failure_rate * 1.2:
            anomalies.append({
                'type': 'failure_spike',
                'message': f'Recent failure rate ({recent_failure_rate:.2f}%) is {((recent_failure_rate - overall_failure_rate) / overall_failure_rate * 100):.1f}% higher than average ({overall_failure_rate:.2f}%)',
                'severity': 'high',
                'data': {
                    'recent_rate': recent_failure_rate,
                    'overall_rate': overall_failure_rate,
                    'increase_percentage': ((recent_failure_rate - overall_failure_rate) / overall_failure_rate * 100)
                }
            })
    
    # Branch anomalies with exact thresholds
    branch_failures = df.groupby('branch_name').agg({
        'is_failed': ['count', 'sum', 'mean']
    })
    branch_failures.columns = ['total', 'failed', 'failure_rate']
    branch_failures['failure_rate'] = branch_failures['failure_rate'] * 100
    
    high_failure_branches = branch_failures[branch_failures['failure_rate'] > 20].index.tolist()
    
    if high_failure_branches:
        anomalies.append({
            'type': 'branch_failure',
            'message': f'Branches with critically high failure rates (>20%): {", ".join(high_failure_branches)}',
            'severity': 'high',
            'data': {
                'branches': high_failure_branches,
                'rates': {branch: f"{branch_failures.loc[branch, 'failure_rate']:.2f}%" for branch in high_failure_branches}
            }
        })
    
    # Hourly pattern anomalies
    hourly_stats = df.groupby('hour').agg({
        'is_failed': ['count', 'sum', 'mean']
    })
    hourly_stats.columns = ['total', 'failed', 'failure_rate']
    hourly_stats['failure_rate'] = hourly_stats['failure_rate'] * 100
    
    # Find hours with unusually high failure rates
    if len(hourly_stats) > 0:
        mean_hourly_rate = hourly_stats['failure_rate'].mean()
        std_hourly_rate = hourly_stats['failure_rate'].std()
        
        if std_hourly_rate > 0:
            anomaly_threshold = mean_hourly_rate + 2 * std_hourly_rate
            anomalous_hours = hourly_stats[hourly_stats['failure_rate'] > anomaly_threshold].index.tolist()
            
            if anomalous_hours:
                anomalies.append({
                    'type': 'hourly_anomaly',
                    'message': f'Unusually high failure rates at hours: {", ".join([f"{h}:00" for h in anomalous_hours])}',
                    'severity': 'medium',
                    'data': {
                        'hours': anomalous_hours,
                        'rates': {hour: f"{hourly_stats.loc[hour, 'failure_rate']:.2f}%" for hour in anomalous_hours}
                    }
                })
    
    return anomalies

def calculate_exact_metrics(df):
    """Calculate exact metrics for validation"""
    metrics = {
        'total_transactions': len(df),
        'failed_transactions': int(df['is_failed'].sum()),
        'success_transactions': int((~df['is_failed']).sum()),
        'failure_rate': float((df['is_failed'].sum() / len(df)) * 100) if len(df) > 0 else 0.0,
        'success_rate': float(((~df['is_failed']).sum() / len(df)) * 100) if len(df) > 0 else 0.0,
        'total_amount': float(df['transaction_amount'].sum()),
        'failed_amount': float(df[df['is_failed']]['transaction_amount'].sum()),
        'success_amount': float(df[~df['is_failed']]['transaction_amount'].sum()),
        'avg_transaction': float(df['transaction_amount'].mean()),
        'total_tax': float(df['tax_amount'].sum()),
        'unique_branches': int(df['branch_name'].nunique()),
        'unique_malls': int(df['mall_name'].nunique())
    }
    
    # Add validation
    if len(df) > 0:
        assert metrics['failed_transactions'] + metrics['success_transactions'] == metrics['total_transactions']
        assert abs(metrics['failure_rate'] + metrics['success_rate'] - 100.0) < 0.01
    
    return metrics