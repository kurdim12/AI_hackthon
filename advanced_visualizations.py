import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_real_time_risk_radar(df):
    """Create live risk radar chart"""
    current_hour = datetime.now().hour
    
    # Get top 5 branches for radar chart
    branch_names = df['branch_name'].unique()[:5]
    
    fig = go.Figure()
    
    categories = ['Failure Rate', 'Transaction Volume', 'Average Amount', 'Peak Hour Risk', 'Overall Risk']
    
    for branch in branch_names:
        branch_data = df[df['branch_name'] == branch]
        hour_data = branch_data[branch_data['hour'] == current_hour]
        
        # Calculate metrics
        failure_rate = branch_data['is_failed'].mean() * 100
        tx_volume = len(branch_data) / 100  # Normalized
        avg_amount = branch_data['transaction_amount'].mean() / 100  # Normalized
        peak_risk = hour_data['is_failed'].mean() * 100 if len(hour_data) > 0 else 50
        overall_risk = (failure_rate + peak_risk) / 2
        
        values = [failure_rate, tx_volume, avg_amount, peak_risk, overall_risk]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=branch
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title=f"Real-Time Risk Radar (Hour: {current_hour}:00)",
        height=500
    )
    
    return fig

def create_predictive_timeline(df, predictions):
    """Create predictive failure timeline"""
    fig = go.Figure()
    
    # Historical data
    historical = df.groupby('date')['is_failed'].mean() * 100
    
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical.values,
        mode='lines',
        name='Historical Failure Rate',
        line=dict(color='blue', width=3)
    ))
    
    # Future predictions
    last_date = df['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted Failure Rate',
        line=dict(color='red', dash='dash', width=3)
    ))
    
    # Confidence intervals
    upper_bound = [p * 1.1 for p in predictions]
    lower_bound = [p * 0.9 for p in predictions]
    
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=upper_bound + lower_bound[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,0,0,0.1)'),
        name='Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title="Failure Rate: Historical vs Predicted",
        xaxis_title="Date",
        yaxis_title="Failure Rate (%)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_anomaly_dna_visualization(dna_signatures):
    """Visualize anomaly DNA patterns"""
    fig = go.Figure()
    
    branches = list(dna_signatures.keys())[:5]  # Top 5 branches
    
    for i, branch in enumerate(branches):
        signature = dna_signatures[branch]
        
        # Create a unique pattern for each branch
        hourly_pattern = signature['hourly_pattern']
        
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=hourly_pattern,
            mode='lines+markers',
            name=branch,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Anomaly DNA Signatures by Branch",
        xaxis_title="Hour of Day",
        yaxis_title="Failure Rate",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_branch_risk_heatmap(df):
    """Create advanced risk heatmap"""
    # Create risk matrix
    risk_data = []
    
    branches = df['branch_name'].unique()
    hours = range(24)
    
    for branch in branches:
        branch_risks = []
        for hour in hours:
            hour_data = df[(df['branch_name'] == branch) & (df['hour'] == hour)]
            
            if len(hour_data) > 0:
                failure_rate = hour_data['is_failed'].mean() * 100
                transaction_count = len(hour_data)
                
                # Complex risk calculation
                risk_score = failure_rate * (1 + np.log1p(transaction_count) / 10)
            else:
                risk_score = 0
            
            branch_risks.append(risk_score)
        
        risk_data.append(branch_risks)
    
    fig = go.Figure(data=go.Heatmap(
        z=risk_data,
        x=list(hours),
        y=branches,
        colorscale='RdYlGn_r',
        text=[[f"{val:.1f}" for val in row] for row in risk_data],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Risk Score")
    ))
    
    # Add current hour indicator
    current_hour = datetime.now().hour
    fig.add_vline(x=current_hour, line_dash="dash", line_color="blue", line_width=3)
    
    fig.update_layout(
        title="Live Transaction Risk Heatmap",
        xaxis_title="Hour of Day",
        yaxis_title="Branch",
        height=600
    )
    
    return fig

def create_financial_impact_gauge(impact_data):
    """Create gauge chart for financial impact"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=impact_data['current'],
        title={'text': "Revenue Impact (%)"},
        delta={'reference': impact_data['target']},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': impact_data['critical']
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig