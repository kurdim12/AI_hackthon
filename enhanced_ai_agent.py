from ai_agent import FinancialAnalysisAgent
from advanced_features import PredictiveFailurePreventor, SmartTransactionRouter, AnomalyDNASystem
from datetime import datetime

class SuperFinancialAgent(FinancialAnalysisAgent):
    def __init__(self, df):
        super().__init__(df)
        self.pfp = PredictiveFailurePreventor(df)
        self.router = SmartTransactionRouter(df)
        self.dna = AnomalyDNASystem(df)
    
    def analyze_with_prediction(self, question):
        """Enhanced analysis with predictive capabilities"""
        # Get standard analysis
        standard_result = self.query(question)
        
        # Add predictive insights if relevant
        if any(keyword in question.lower() for keyword in ['failure', 'risk', 'predict', 'future']):
            pfp_insights = self._generate_pfp_insights()
            standard_result['response'] += f"\n\n**Predictive Analysis:**\n{pfp_insights}"
        
        # Add routing recommendations if relevant
        if any(keyword in question.lower() for keyword in ['route', 'gateway', 'process']):
            routing_insights = self._generate_routing_insights()
            standard_result['response'] += f"\n\n**Smart Routing Recommendations:**\n{routing_insights}"
        
        return standard_result
    
    def _generate_pfp_insights(self):
        """Generate predictive insights"""
        high_risk_branches = []
        
        for branch in self.df['branch_name'].unique():
            score = self.pfp.calculate_pfp_score({
                'branch': branch,
                'hour': datetime.now().hour,
                'amount': self.df[self.df['branch_name'] == branch]['transaction_amount'].mean()
            })
            
            if score['risk_level'] == 'High':
                high_risk_branches.append((branch, score['score']))
        
        insights = f"Currently {len(high_risk_branches)} branches show high risk patterns.\n"
        if high_risk_branches:
            insights += "High-risk branches: " + ", ".join([f"{b[0]} (Risk Score: {b[1]:.2f})" for b in high_risk_branches[:3]])
            insights += "\n\nRecommended Actions:\n"
            insights += "1. Route transactions through secondary gateways for high-risk branches\n"
            insights += "2. Increase monitoring frequency during peak risk hours\n"
            insights += "3. Consider implementing transaction chunking for large amounts"
        
        return insights
    
    def _generate_routing_insights(self):
        """Generate routing insights"""
        insights = []
        current_hour = datetime.now().hour
        
        for branch in self.df['branch_name'].unique()[:3]:  # Top 3 branches
            routing = self.router.route_transaction(branch, 500, datetime.now())
            insights.append(f"{branch}: Use {routing['gateway']} with {routing['retry_strategy']} retry strategy")
        
        return "\n".join(insights)
    
    def get_real_time_recommendations(self, branch, amount):
        """Get real-time recommendations for a specific transaction"""
        # PFP Score
        pfp_result = self.pfp.calculate_pfp_score({
            'branch': branch,
            'hour': datetime.now().hour,
            'amount': amount
        })
        
        # Routing recommendation
        routing = self.router.route_transaction(branch, amount, datetime.now())
        
        # DNA pattern matching
        current_pattern = {'branch': branch, 'hour': datetime.now().hour}
        dna_matches = self.dna.match_anomaly_pattern(current_pattern)
        
        return {
            'pfp_score': pfp_result,
            'routing': routing,
            'dna_matches': dna_matches,
            'overall_recommendation': self._generate_overall_recommendation(pfp_result, routing, dna_matches)
        }
    
    def _generate_overall_recommendation(self, pfp_result, routing, dna_matches):
        if pfp_result['risk_level'] == 'High':
            return "HIGH RISK: Consider delaying transaction or using enhanced monitoring"
        elif pfp_result['risk_level'] == 'Medium':
            return f"MODERATE RISK: Proceed with {routing['gateway']} and standard monitoring"
        else:
            return "LOW RISK: Transaction can proceed normally"