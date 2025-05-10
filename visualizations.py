import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_failure_heatmap(df):
    """Create heatmap showing failure patterns by hour and mall"""
    pivot_data = df.pivot_table(
        values='is_failed',
        index='hour',
        columns='mall_name',
        aggfunc='mean'
    ) * 100
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Mall", y="Hour of Day", color="Failure Rate %"),
        title="Failure Rate Patterns by Hour and Mall",
        color_continuous_scale="RdYlBu_r",
        aspect="auto"
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Shopping Mall",
        yaxis_title="Hour of Day"
    )
    
    return fig

def create_daily_trends(df):
    """Create daily transaction volume chart"""
    daily_stats = df.groupby('date').agg({
        'transaction_id': 'count',
        'is_failed': 'sum',
        'transaction_amount': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    # Total transactions
    fig.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['transaction_id'],
        mode='lines+markers',
        name='Total Transactions',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Failed transactions
    fig.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['is_failed'],
        mode='lines+markers',
        name='Failed Transactions',
        line=dict(color='#d62728', width=2)
    ))
    
    fig.update_layout(
        title="Daily Transaction Trends",
        xaxis_title="Date",
        yaxis_title="Number of Transactions",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_branch_performance(df):
    """Create branch performance comparison chart"""
    branch_stats = df.groupby('branch_name').agg({
        'transaction_id': 'count',
        'is_failed': 'mean',
        'transaction_amount': 'sum'
    }).reset_index()
    
    branch_stats['failure_rate'] = branch_stats['is_failed'] * 100
    branch_stats = branch_stats.sort_values('failure_rate', ascending=True)
    
    fig = go.Figure()
    
    # Add bars for failure rate
    fig.add_trace(go.Bar(
        y=branch_stats['branch_name'],
        x=branch_stats['failure_rate'],
        orientation='h',
        marker_color=branch_stats['failure_rate'],
        marker_colorscale='RdYlGn_r',
        text=branch_stats['failure_rate'].round(1),
        textposition='inside',
        name='Failure Rate %'
    ))
    
    fig.update_layout(
        title="Branch Performance - Failure Rates",
        xaxis_title="Failure Rate (%)",
        yaxis_title="Branch",
        height=400,
        showlegend=False
    )
    
    return fig

def create_financial_impact(df):
    """Create financial impact visualization"""
    daily_impact = df.groupby('date').agg({
        'transaction_amount': 'sum',
        'is_failed': lambda x: (x * df.loc[x.index, 'transaction_amount']).sum()
    }).reset_index()
    
    daily_impact.columns = ['date', 'total_amount', 'failed_amount']
    daily_impact['success_amount'] = daily_impact['total_amount'] - daily_impact['failed_amount']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=daily_impact['date'],
        y=daily_impact['success_amount'],
        name='Successful',
        marker_color='#2ca02c'
    ))
    
    fig.add_trace(go.Bar(
        x=daily_impact['date'],
        y=daily_impact['failed_amount'],
        name='Failed',
        marker_color='#d62728'
    ))
    
    fig.update_layout(
        title="Daily Revenue Impact",
        xaxis_title="Date",
        yaxis_title="Transaction Amount ($)",
        barmode='stack',
        height=400
    )
    
    return fig

def create_time_analysis(df):
    """Create time-based analysis visualization"""
    hourly_stats = df.groupby('hour').agg({
        'transaction_id': 'count',
        'is_failed': 'mean'
    }).reset_index()
    
    hourly_stats['failure_rate'] = hourly_stats['is_failed'] * 100
    
    fig = go.Figure()
    
    # Transaction volume
    fig.add_trace(go.Bar(
        x=hourly_stats['hour'],
        y=hourly_stats['transaction_id'],
        name='Transaction Volume',
        marker_color='#1f77b4',
        opacity=0.7
    ))
    
    # Failure rate line
    fig.add_trace(go.Scatter(
        x=hourly_stats['hour'],
        y=hourly_stats['failure_rate'],
        name='Failure Rate %',
        mode='lines+markers',
        line=dict(color='#d62728', width=3),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Hourly Transaction Patterns",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Transactions",
        yaxis2=dict(
            title="Failure Rate (%)",
            overlaying='y',
            side='right'
        ),
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_risk_matrix(df):
    """Create risk assessment matrix for branches"""
    branch_risk = df.groupby('branch_name').agg({
        'transaction_id': 'count',
        'is_failed': 'mean',
        'transaction_amount': 'sum'
    }).reset_index()
    
    branch_risk['failure_rate'] = branch_risk['is_failed'] * 100
    branch_risk['avg_transaction'] = branch_risk['transaction_amount'] / branch_risk['transaction_id']
    
    fig = px.scatter(
        branch_risk,
        x='failure_rate',
        y='avg_transaction',
        size='transaction_id',
        color='failure_rate',
        text='branch_name',
        title="Branch Risk Matrix",
        labels={
            'failure_rate': 'Failure Rate (%)',
            'avg_transaction': 'Average Transaction Value ($)',
            'transaction_id': 'Transaction Volume'
        },
        color_continuous_scale='RdYlGn_r'
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=400)
    
    return fig