import pandas as pd
from datetime import datetime, timedelta

class BranchGamification:
    def __init__(self, df):
        self.df = df
        self.achievements = self._define_achievements()
        self.scores = self._calculate_all_scores()
    
    def _define_achievements(self):
        return {
            'zero_hero': {
                'name': 'Zero Hero',
                'desc': 'Zero failures for 24 hours',
                'points': 100,
                'icon': '🏆'
            },
            'improvement_champion': {
                'name': 'Improvement Champion',
                'desc': '20% reduction in failure rate',
                'points': 200,
                'icon': '📈'
            },
            'consistency_king': {
                'name': 'Consistency King',
                'desc': 'Below 5% failure rate for 7 days',
                'points': 300,
                'icon': '👑'
            },
            'peak_performer': {
                'name': 'Peak Performer',
                'desc': 'Best performance during peak hours',
                'points': 150,
                'icon': '⭐'
            },
            'comeback_kid': {
                'name': 'Comeback Kid',
                'desc': 'Biggest improvement from worst to better',
                'points': 250,
                'icon': '🚀'
            }
        }
    
    def _calculate_all_scores(self):
        scores = {}
        for branch in self.df['branch_name'].unique():
            scores[branch] = self.calculate_branch_score(branch)
        return scores
    
    def calculate_branch_score(self, branch_name):
        """Calculate gamification score for a branch"""
        branch_data = self.df[self.df['branch_name'] == branch_name]
        
        score = 0
        achievements_earned = []
        
        # Check achievements
        if self._check_zero_hero(branch_data):
            score += self.achievements['zero_hero']['points']
            achievements_earned.append('zero_hero')
        
        if self._check_improvement_champion(branch_data):
            score += self.achievements['improvement_champion']['points']
            achievements_earned.append('improvement_champion')
        
        if self._check_consistency_king(branch_data):
            score += self.achievements['consistency_king']['points']
            achievements_earned.append('consistency_king')
        
        if self._check_peak_performer(branch_data):
            score += self.achievements['peak_performer']['points']
            achievements_earned.append('peak_performer')
        
        return {
            'branch': branch_name,
            'total_score': score,
            'achievements': achievements_earned,
            'rank': self._calculate_rank(score),
            'level': self._calculate_level(score),
            'next_milestone': self._get_next_milestone(score)
        }
    
    def _check_zero_hero(self, branch_data):
        # Check last 24 hours
        last_24h = branch_data[
            branch_data['transaction_date'] >= 
            branch_data['transaction_date'].max() - timedelta(hours=24)
        ]
        return last_24h['is_failed'].sum() == 0
    
    def _check_improvement_champion(self, branch_data):
        # Compare last week to previous week
        last_week = branch_data[
            branch_data['transaction_date'] >= 
            branch_data['transaction_date'].max() - timedelta(days=7)
        ]
        prev_week = branch_data[
            (branch_data['transaction_date'] >= 
             branch_data['transaction_date'].max() - timedelta(days=14)) &
            (branch_data['transaction_date'] < 
             branch_data['transaction_date'].max() - timedelta(days=7))
        ]
        
        if len(prev_week) == 0:
            return False
        
        last_week_rate = last_week['is_failed'].mean()
        prev_week_rate = prev_week['is_failed'].mean()
        
        return last_week_rate < prev_week_rate * 0.8
    
    def _check_consistency_king(self, branch_data):
        # Check last 7 days
        last_7_days = branch_data[
            branch_data['transaction_date'] >= 
            branch_data['transaction_date'].max() - timedelta(days=7)
        ]
        return last_7_days['is_failed'].mean() < 0.05
    
    def _check_peak_performer(self, branch_data):
        # Check performance during peak hours (9-17)
        peak_hours = branch_data[
            (branch_data['hour'] >= 9) & (branch_data['hour'] <= 17)
        ]
        return peak_hours['is_failed'].mean() < 0.05
    
    def _calculate_rank(self, score):
        if score >= 1000:
            return 'Platinum'
        elif score >= 700:
            return 'Gold'
        elif score >= 400:
            return 'Silver'
        elif score >= 200:
            return 'Bronze'
        else:
            return 'Beginner'
    
    def _calculate_level(self, score):
        return min(score // 100 + 1, 10)
    
    def _get_next_milestone(self, score):
        milestones = [100, 200, 400, 700, 1000]
        for milestone in milestones:
            if score < milestone:
                return milestone
        return None
    
    def create_leaderboard(self):
        """Create competitive leaderboard"""
        all_scores = []
        
        for branch, score_data in self.scores.items():
            all_scores.append(score_data)
        
        return sorted(all_scores, key=lambda x: x['total_score'], reverse=True)