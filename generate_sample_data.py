import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_transactions(output_path="jordan_transactions.csv", num_records=10000):
    """Generate sample transaction data similar to the expected format"""
    
    # Define possible values
    malls = ["Mall A", "Mall B", "Mall C", "Mall D", "Mall E"]
    branches = ["Branch North", "Branch South", "Branch East", "Branch West", "Branch Central"]
    transaction_types = ["Sale", "Refund"]
    transaction_statuses = ["Completed", "Failed"]
    
    # Generate dates for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = []
    
    for i in range(num_records):
        # Generate random datetime within the range
        random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
        transaction_date = start_date + timedelta(seconds=random_seconds)
        
        # Generate transaction details
        transaction_type = np.random.choice(transaction_types, p=[0.95, 0.05])  # 95% sales, 5% refunds
        transaction_status = np.random.choice(transaction_statuses, p=[0.85, 0.15])  # 15% failure rate
        
        # Generate amounts
        base_amount = np.random.lognormal(mean=3.5, sigma=1.2)  # Log-normal distribution for realistic amounts
        transaction_amount = round(max(10, min(base_amount, 5000)), 2)  # Clip between 10 and 5000
        tax_amount = round(transaction_amount * 0.16, 2)  # 16% tax
        
        data.append({
            'transaction_id': f'TXN_{i+1:06d}',
            'mall_name': np.random.choice(malls),
            'branch_name': np.random.choice(branches),
            'transaction_date': transaction_date.strftime('%d/%m/%Y %H:%M'),
            'tax_amount': tax_amount,
            'transaction_amount': transaction_amount,
            'transaction_type': transaction_type,
            'transaction_status': transaction_status
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic patterns
    # Higher failure rates during peak hours
    df['hour'] = pd.to_datetime(df['transaction_date'], format='%d/%m/%Y %H:%M').dt.hour
    peak_hours = df['hour'].isin([12, 13, 17, 18, 19])
    peak_indices = df[peak_hours].index
    
    # Increase failure rate during peak hours
    num_peak_failures = int(len(peak_indices) * 0.25)  # 25% failure rate during peak
    failure_indices = np.random.choice(peak_indices, size=num_peak_failures, replace=False)
    df.loc[failure_indices, 'transaction_status'] = 'Failed'
    
    # Create mall-specific patterns
    # Mall C has higher failure rate
    mall_c_indices = df[df['mall_name'] == 'Mall C'].index
    num_mall_c_failures = int(len(mall_c_indices) * 0.3)  # 30% failure rate at Mall C
    mall_c_failure_indices = np.random.choice(mall_c_indices, size=num_mall_c_failures, replace=False)
    df.loc[mall_c_failure_indices, 'transaction_status'] = 'Failed'
    
    # Remove temporary hour column
    df = df.drop('hour', axis=1)
    
    # Sort by date
    df['temp_date'] = pd.to_datetime(df['transaction_date'], format='%d/%m/%Y %H:%M')
    df = df.sort_values('temp_date')
    df = df.drop('temp_date', axis=1)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"Generated {num_records} sample transactions and saved to {output_path}")
    print(f"Overall failure rate: {(df['transaction_status'] == 'Failed').mean() * 100:.1f}%")
    print(f"Number of unique malls: {df['mall_name'].nunique()}")
    print(f"Number of unique branches: {df['branch_name'].nunique()}")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_transactions()
    
    # Display sample rows
    print("\nSample data:")
    print(df.head())
    
    # Display statistics
    print("\nStatistics:")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"Average transaction amount: ${df['transaction_amount'].mean():.2f}")
    print(f"Total revenue: ${df['transaction_amount'].sum():,.2f}")
    print(f"Failed transactions: {(df['transaction_status'] == 'Failed').sum()}")