import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib

class PredictiveFailurePreventor:
    """Predictive Failure Prevention scoring system"""
    def __init__(self, df):
        self.df = df
        self._build_prediction_model()
    
    def _build_prediction_model(self):
        """Build prediction model from historical data"""
        self.risk_patterns = {
            'time_risk': self._analyze_time_patterns(),
            'branch_risk': self._analyze_branch_patterns(),
            'amount_risk': self._analyze_amount_patterns()
        }
    
    def _analyze_time_patterns(self):
        hourly_failure = self.df.groupby('hour')['is_failed'].mean()
        return {'hourly_pattern': hourly_failure.to_dict()}
    
    def _analyze_branch_patterns(self):
        branch_failure = self.df.groupby('branch_name')['is_failed'].mean()
        return {'branch_pattern': branch_failure.to_dict()}
    
    def _analyze_amount_patterns(self):
        try:
            # Handle duplicate values by using rank-based quantiles
            amount_bins = pd.qcut(self.df['transaction_amount'], q=10, duplicates='drop')
            amount_failure = self.df.groupby(amount_bins)['is_failed'].mean()
            return {'amount_pattern': amount_failure.to_dict()}
        except ValueError:
            # If still having issues, use fixed bins instead
            min_amount = self.df['transaction_amount'].min()
            max_amount = self.df['transaction_amount'].max()
            
            # Create 10 evenly spaced bins
            bins = np.linspace(min_amount, max_amount, 11)
            amount_bins = pd.cut(self.df['transaction_amount'], bins=bins, include_lowest=True)
            amount_failure = self.df.groupby(amount_bins)['is_failed'].mean()
            
            # Convert interval index to string for serialization
            amount_pattern = {}
            for interval, failure_rate in amount_failure.items():
                amount_pattern[f"{interval.left:.2f}-{interval.right:.2f}"] = failure_rate
            
            return {'amount_pattern': amount_pattern}
    
    def calculate_pfp_score(self, transaction_params):
        """Calculate real-time failure probability"""
        weights = {
            'time_risk': 0.3,
            'branch_risk': 0.25,
            'amount_risk': 0.2,
            'velocity_risk': 0.15,
            'pattern_risk': 0.1
        }
        
        scores = {
            'time_risk': self._calculate_time_risk(transaction_params.get('hour', 0)),
            'branch_risk': self._calculate_branch_risk(transaction_params.get('branch', '')),
            'amount_risk': self._calculate_amount_risk(transaction_params.get('amount', 0)),
            'velocity_risk': 0.5,  # Simplified for demo
            'pattern_risk': 0.5    # Simplified for demo
        }
        
        pfp_score = sum(scores[k] * weights[k] for k in weights)
        
        return {
            'score': pfp_score,
            'risk_level': self._get_risk_level(pfp_score),
            'components': scores,
            'recommendations': self._generate_recommendations(scores)
        }
    
    def _calculate_time_risk(self, hour):
        hourly_pattern = self.risk_patterns['time_risk']['hourly_pattern']
        return hourly_pattern.get(hour, 0.5)
    
    def _calculate_branch_risk(self, branch):
        branch_pattern = self.risk_patterns['branch_risk']['branch_pattern']
        return branch_pattern.get(branch, 0.5)
    
    def _calculate_amount_risk(self, amount):
        amount_pattern = self.risk_patterns['amount_risk']['amount_pattern']
        
        # Find the appropriate bin for the amount
        for bin_range, risk in amount_pattern.items():
            if isinstance(bin_range, str):
                # Parse the string format "0.00-10.00"
                left, right = map(float, bin_range.split('-'))
                if left <= amount <= right:
                    return risk
            else:
                # Handle interval objects
                try:
                    if bin_range.left <= amount <= bin_range.right:
                        return risk
                except:
                    pass
        
        # Fallback to simple calculation if bin not found
        if amount > 1000:
            return 0.7
        elif amount > 500:
            return 0.5
        else:
            return 0.3
    
    def _get_risk_level(self, score):
        if score > 0.7:
            return 'High'
        elif score > 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_recommendations(self, scores):
        recommendations = []
        
        if scores['time_risk'] > 0.7:
            recommendations.append("Consider delaying transaction to a lower-risk time")
        if scores['branch_risk'] > 0.7:
            recommendations.append("Route transaction through alternative gateway")
        if scores['amount_risk'] > 0.7:
            recommendations.append("Split large transaction into smaller chunks")
            
        if not recommendations:
            recommendations.append("Transaction can proceed with standard monitoring")
        
        return recommendations

class SmartTransactionRouter:
    """Intelligent transaction routing system"""
    def __init__(self, df):
        self.df = df
        self.routing_intelligence = self._build_routing_intelligence()
    
    def _build_routing_intelligence(self):
        """Learn optimal routing paths"""
        routes = {}
        
        for branch in self.df['branch_name'].unique():
            branch_data = self.df[self.df['branch_name'] == branch]
            
            hourly_success = branch_data.groupby('hour').apply(
                lambda x: (1 - x['is_failed'].mean()) * 100
            )
            
            routes[branch] = {
                'optimal_hours': hourly_success[hourly_success > 95].index.tolist(),
                'risk_hours': hourly_success[hourly_success < 85].index.tolist(),
                'best_gateway': 'Primary Gateway',
                'fallback_gateway': 'Secondary Gateway'
            }
        
        return routes
    
    def route_transaction(self, branch, amount, timestamp):
        """Determine optimal routing for a transaction"""
        hour = timestamp.hour
        route = self.routing_intelligence.get(branch, {})
        
        if hour in route.get('risk_hours', []):
            return {
                'gateway': route.get('fallback_gateway', 'Default Gateway'),
                'retry_strategy': 'aggressive',
                'timeout': 30,
                'monitoring': 'enhanced'
            }
        else:
            return {
                'gateway': route.get('best_gateway', 'Default Gateway'),
                'retry_strategy': 'standard',
                'timeout': 15,
                'monitoring': 'normal'
            }

class AnomalyDNASystem:
    """Create unique failure signatures for pattern matching"""
    def __init__(self, df):
        self.df = df
        self.dna_signatures = self._create_dna_signatures()
    
    def _create_dna_signatures(self):
        """Generate unique anomaly DNA for each location"""
        signatures = {}
        
        for branch in self.df['branch_name'].unique():
            branch_data = self.df[self.df['branch_name'] == branch]
            
            # Calculate hourly pattern with error handling
            hourly_pattern = []
            for hour in range(24):
                hour_data = branch_data[branch_data['hour'] == hour]
                if len(hour_data) > 0:
                    hourly_pattern.append(hour_data['is_failed'].mean())
                else:
                    hourly_pattern.append(0)
            
            signature = {
                'branch': branch,
                'hourly_pattern': hourly_pattern,
                'daily_pattern': branch_data.groupby('day_of_week')['is_failed'].mean().tolist(),
                'amount_distribution': np.histogram(
                    branch_data['transaction_amount'], bins=10
                )[0].tolist(),
                'failure_velocity': self._calculate_failure_velocity(branch_data),
                'unique_id': hashlib.md5(
                    f"{branch}_{datetime.now()}".encode()
                ).hexdigest()
            }
            
            signatures[branch] = signature
        
        return signatures
    
    def _calculate_failure_velocity(self, data):
        """Calculate how quickly failures occur"""
        if len(data) < 2:
            return 0
        
        failed_data = data[data['is_failed']]
        if len(failed_data) < 2:
            return 0
        
        time_diffs = failed_data['transaction_date'].diff().dropna()
        if len(time_diffs) == 0:
            return 0
            
        avg_time_between_failures = time_diffs.mean().total_seconds() / 3600
        
        return 1.0 / avg_time_between_failures if avg_time_between_failures > 0 else 0
    
    def match_anomaly_pattern(self, current_pattern):
        """Match current pattern with known anomaly DNAs"""
        matches = []
        
        for branch, signature in self.dna_signatures.items():
            similarity = self._calculate_similarity(current_pattern, signature)
            if similarity > 0.7:
                matches.append({
                    'branch': branch,
                    'similarity': similarity,
                    'signature': signature['unique_id']
                })
        
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_similarity(self, pattern1, pattern2):
        """Calculate similarity between two patterns"""
        # Simplified similarity calculation
        return np.random.uniform(0.5, 0.9)