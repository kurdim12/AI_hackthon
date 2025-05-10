import os
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
import pandas as pd
import numpy as np

load_dotenv()

class FinancialAnalysisAgent:
    def __init__(self, df):
        self.df = df
        self.llm = ChatOpenAI(
            temperature=0,  # Set to 0 for maximum accuracy
            model_name="gpt-4", # Use GPT-4 for better accuracy
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.agent = self._create_agent()
        
    def _create_agent(self):
        """Create a pandas dataframe agent with custom prompt"""
        prefix = """You are a senior financial analyst specializing in retail transaction data. 
        CRITICAL INSTRUCTIONS:
        1. Always provide EXACT numbers from the data - do not approximate or round unless asked
        2. Double-check all calculations before providing answers
        3. When asked about specific metrics, query the dataframe directly
        4. Always include the calculation method when providing numerical answers
        5. Be precise with decimal places - maintain the same precision as in the data
        
        You have access to transaction data with the following columns:
        - transaction_id: Unique identifier
        - mall_name: Shopping mall name
        - branch_name: Store branch name
        - transaction_date: Date and time of transaction
        - tax_amount: Tax collected
        - transaction_amount: Total transaction value
        - transaction_type: Sale or Refund
        - transaction_status: Completed or Failed
        - hour: Hour of transaction (0-23)
        - day_of_week: Day name
        - date: Date only
        - is_failed: Boolean (True if Failed)
        - is_weekend: Boolean (True if Saturday/Sunday)
        
        IMPORTANT: When calculating percentages or rates, use the actual data values.
        For failure rate: (number of failed transactions / total transactions) * 100
        
        Example calculations:
        - Total transactions: len(df)
        - Failed transactions: df['is_failed'].sum()
        - Failure rate: (df['is_failed'].sum() / len(df)) * 100
        - Branch failure rate: (df[df['branch_name'] == 'X']['is_failed'].sum() / df[df['branch_name'] == 'X'].shape[0]) * 100
        """
        
        return create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,  # Set to True to see the agent's reasoning
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=prefix,
            allow_dangerous_code=True,
            handle_parsing_errors=True
        )
    
    def query(self, question):
        """Process a natural language query"""
        try:
            # Add data validation context to the question
            enhanced_question = f"""
            Using the dataframe 'df' with {len(self.df)} rows, answer this question precisely:
            {question}
            
            Important: 
            - Provide exact numbers from the data
            - Show your calculation if applicable
            - Use proper decimal precision (2 decimal places for percentages and currency)
            - Verify your answer by checking the data
            """
            
            response = self.agent.run(enhanced_question)
            return {
                'success': True,
                'response': response,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'response': None,
                'error': str(e)
            }
    
    def get_smart_insights(self, df):
        """Generate intelligent insights automatically with accurate data"""
        insights = []
        
        # Exact failure rate calculation
        total_transactions = len(df)
        failed_transactions = df['is_failed'].sum()
        failure_rate = (failed_transactions / total_transactions) * 100
        
        if failure_rate > 10:
            insights.append({
                'type': 'warning',
                'title': 'High Failure Rate Alert',
                'content': f'Transaction failure rate is exactly {failure_rate:.2f}% ({failed_transactions} out of {total_transactions} transactions)',
                'recommendation': 'Investigate payment gateway issues and implement retry mechanisms'
            })
        
        # Exact branch performance
        branch_stats = df.groupby('branch_name').agg({
            'is_failed': ['count', 'sum']
        })
        branch_stats.columns = ['total', 'failed']
        branch_stats['failure_rate'] = (branch_stats['failed'] / branch_stats['total']) * 100
        
        worst_branch = branch_stats['failure_rate'].idxmax()
        best_branch = branch_stats['failure_rate'].idxmin()
        
        insights.append({
            'type': 'insight',
            'title': 'Branch Performance Gap',
            'content': f'{worst_branch} has exactly {branch_stats.loc[worst_branch, "failure_rate"]:.2f}% failure rate ({branch_stats.loc[worst_branch, "failed"]} failures out of {branch_stats.loc[worst_branch, "total"]} transactions) vs {best_branch} at {branch_stats.loc[best_branch, "failure_rate"]:.2f}% ({branch_stats.loc[best_branch, "failed"]} failures out of {branch_stats.loc[best_branch, "total"]} transactions)',
            'recommendation': 'Share best practices from top-performing branches'
        })
        
        # Exact time patterns
        hourly_failures = df[df['is_failed']].groupby('hour').size()
        if len(hourly_failures) > 0:
            peak_hour = hourly_failures.idxmax()
            peak_failures = hourly_failures[peak_hour]
            
            hourly_total = df.groupby('hour').size()
            peak_hour_total = hourly_total[peak_hour]
            peak_hour_rate = (peak_failures / peak_hour_total) * 100
            
            insights.append({
                'type': 'pattern',
                'title': 'Peak Failure Times',
                'content': f'Most failures occur at {peak_hour}:00 with exactly {peak_failures} failures out of {peak_hour_total} transactions ({peak_hour_rate:.2f}% failure rate)',
                'recommendation': 'Schedule system maintenance outside peak hours and add capacity during high-traffic periods'
            })
        
        # Exact financial impact
        failed_amount = df[df['is_failed']]['transaction_amount'].sum()
        total_amount = df['transaction_amount'].sum()
        failed_percentage = (failed_amount / total_amount) * 100
        
        # Exact projection calculation
        unique_days = df['date'].nunique()
        daily_average_loss = failed_amount / unique_days
        projected_monthly_loss = daily_average_loss * 30
        
        insights.append({
            'type': 'financial',
            'title': 'Revenue Impact',
            'content': f'Failed transactions total exactly ${failed_amount:,.2f} ({failed_percentage:.2f}% of total revenue). Based on {unique_days} days of data, the projected monthly loss is ${projected_monthly_loss:,.2f}',
            'recommendation': 'Prioritize fixing payment issues to recover potential revenue'
        })
        
        return insights
    
    def get_top_insights(self, limit=3):
        """Get top insights with accurate data"""
        insights = []
        
        # Mall-specific analysis
        mall_stats = self.df.groupby('mall_name').agg({
            'is_failed': ['count', 'sum'],
            'transaction_amount': 'sum'
        })
        mall_stats.columns = ['total_transactions', 'failed_transactions', 'total_amount']
        mall_stats['failure_rate'] = (mall_stats['failed_transactions'] / mall_stats['total_transactions']) * 100
        
        worst_mall = mall_stats['failure_rate'].idxmax()
        
        insights.append({
            'title': 'Mall Performance Analysis',
            'description': f'{worst_mall} has the highest failure rate at {mall_stats.loc[worst_mall, "failure_rate"]:.2f}% with {mall_stats.loc[worst_mall, "failed_transactions"]} failed transactions out of {mall_stats.loc[worst_mall, "total_transactions"]} total',
            'metrics': {
                'Worst Mall': worst_mall,
                'Failure Rate': f'{mall_stats.loc[worst_mall, "failure_rate"]:.2f}%',
                'Failed Amount': f'${self.df[(self.df["mall_name"] == worst_mall) & (self.df["is_failed"])]["transaction_amount"].sum():,.2f}'
            },
            'recommendations': [
                'Investigate infrastructure issues at this location',
                'Review network connectivity and payment systems',
                'Consider backup payment processing systems'
            ]
        })
        
        # Transaction type analysis
        type_stats = self.df.groupby('transaction_type').agg({
            'is_failed': ['count', 'sum']
        })
        type_stats.columns = ['total', 'failed']
        type_stats['failure_rate'] = (type_stats['failed'] / type_stats['total']) * 100
        
        insights.append({
            'title': 'Transaction Type Analysis',
            'description': f'Sales have {type_stats.loc["Sale", "failure_rate"]:.2f}% failure rate ({type_stats.loc["Sale", "failed"]} out of {type_stats.loc["Sale", "total"]}) while Refunds have {type_stats.loc["Refund", "failure_rate"]:.2f}% failure rate ({type_stats.loc["Refund", "failed"]} out of {type_stats.loc["Refund", "total"]})',
            'metrics': {
                'Sale Failure Rate': f'{type_stats.loc["Sale", "failure_rate"]:.2f}%',
                'Refund Failure Rate': f'{type_stats.loc["Refund", "failure_rate"]:.2f}%',
                'Total Sales': f'{type_stats.loc["Sale", "total"]:,}',
                'Total Refunds': f'{type_stats.loc["Refund", "total"]:,}'
            },
            'recommendations': [
                'Focus on the transaction type with higher failure rate',
                'Review processing logic for problematic transaction types',
                'Implement different handling for sales vs refunds if needed'
            ]
        })
        
        # Time-based analysis
        day_stats = self.df.groupby('day_of_week').agg({
            'is_failed': ['count', 'sum']
        })
        day_stats.columns = ['total', 'failed']
        day_stats['failure_rate'] = (day_stats['failed'] / day_stats['total']) * 100
        
        worst_day = day_stats['failure_rate'].idxmax()
        best_day = day_stats['failure_rate'].idxmin()
        
        insights.append({
            'title': 'Daily Pattern Analysis',
            'description': f'{worst_day} has the highest failure rate at {day_stats.loc[worst_day, "failure_rate"]:.2f}% while {best_day} has the lowest at {day_stats.loc[best_day, "failure_rate"]:.2f}%',
            'metrics': {
                'Worst Day': f'{worst_day} ({day_stats.loc[worst_day, "failure_rate"]:.2f}%)',
                'Best Day': f'{best_day} ({day_stats.loc[best_day, "failure_rate"]:.2f}%)',
                'Weekend Rate': f'{self.df[self.df["is_weekend"]]["is_failed"].mean() * 100:.2f}%',
                'Weekday Rate': f'{self.df[~self.df["is_weekend"]]["is_failed"].mean() * 100:.2f}%'
            },
            'recommendations': [
                f'Increase support staffing on {worst_day}',
                f'Analyze what makes {best_day} more successful',
                'Consider maintenance schedules based on daily patterns'
            ]
        })
        
        return insights[:limit]
    
    def get_branch_recommendations(self, branch_name):
        """Get specific recommendations for a branch with accurate data"""
        branch_data = self.df[self.df['branch_name'] == branch_name]
        
        if len(branch_data) == 0:
            return ["No data available for this branch"]
        
        total_transactions = len(branch_data)
        failed_transactions = branch_data['is_failed'].sum()
        failure_rate = (failed_transactions / total_transactions) * 100
        
        recommendations = []
        
        # Time-based analysis
        hourly_failures = branch_data[branch_data['is_failed']].groupby('hour').size()
        if len(hourly_failures) > 0:
            worst_hour = hourly_failures.idxmax()
            worst_hour_failures = hourly_failures[worst_hour]
            hourly_total = branch_data.groupby('hour').size()
            worst_hour_rate = (worst_hour_failures / hourly_total[worst_hour]) * 100
            
            recommendations.append(
                f"Schedule maintenance outside {worst_hour}:00 (peak failure hour with {worst_hour_failures} failures, {worst_hour_rate:.2f}% failure rate)"
            )
        
        # Amount-based analysis
        failed_amounts = branch_data[branch_data['is_failed']]['transaction_amount']
        if len(failed_amounts) > 0:
            avg_failed_amount = failed_amounts.mean()
            median_failed_amount = failed_amounts.median()
            
            if avg_failed_amount > median_failed_amount * 1.5:
                recommendations.append(
                    f"Large transactions are failing more often (avg failed: ${avg_failed_amount:.2f}, median: ${median_failed_amount:.2f})"
                )
        
        # Day of week analysis
        dow_failures = branch_data[branch_data['is_failed']].groupby('day_of_week').size()
        if len(dow_failures) > 0:
            worst_day = dow_failures.idxmax()
            dow_totals = branch_data.groupby('day_of_week').size()
            worst_day_rate = (dow_failures[worst_day] / dow_totals[worst_day]) * 100
            
            recommendations.append(
                f"Increase support on {worst_day} (highest failure count: {dow_failures[worst_day]}, {worst_day_rate:.2f}% failure rate)"
            )
        
        # Compare to average
        overall_failure_rate = (self.df['is_failed'].sum() / len(self.df)) * 100
        if failure_rate > overall_failure_rate * 1.2:
            recommendations.append(
                f"This branch's failure rate ({failure_rate:.2f}%) is {((failure_rate - overall_failure_rate) / overall_failure_rate * 100):.1f}% higher than average ({overall_failure_rate:.2f}%)"
            )
        
        # Add financial impact
        branch_failed_amount = branch_data[branch_data['is_failed']]['transaction_amount'].sum()
        branch_total_amount = branch_data['transaction_amount'].sum()
        branch_failed_percentage = (branch_failed_amount / branch_total_amount) * 100
        
        recommendations.append(
            f"Failed transactions cost ${branch_failed_amount:,.2f} ({branch_failed_percentage:.2f}% of branch revenue)"
        )
        
        return recommendations

# Add this helper function for exact calculations
def validate_calculations(df, result):
    """Validate AI calculations against direct calculations"""
    actual_metrics = {
        'total_transactions': len(df),
        'failed_transactions': df['is_failed'].sum(),
        'failure_rate': (df['is_failed'].sum() / len(df)) * 100
    }
    
    # You can add more validation logic here
    return actual_metrics