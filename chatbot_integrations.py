from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from datetime import datetime
import pandas as pd
from ai_agent import FinancialAnalysisAgent
import asyncio
from typing import Dict, Any
import json
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request

class FinanceGuardTelegramBot:
    """Telegram bot for FinanceGuard AI"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.agent = FinancialAnalysisAgent(df)
        self.user_sessions = {}
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await update.message.reply_text(
            "🤖 Welcome to FinanceGuard AI!\n\n"
            "I can help you analyze transaction data and answer questions about:\n"
            "• Failure rates and patterns\n"
            "• Branch performance\n"
            "• Revenue impact\n"
            "• Transaction trends\n\n"
            "Just ask me anything! For example:\n"
            "- What's the failure rate at Mall A?\n"
            "- Show me today's performance\n"
            "- Which branch has the most failures?\n\n"
            "Commands:\n"
            "/help - Show this help message\n"
            "/status - Current system status\n"
            "/alerts - Show active alerts\n"
            "/report - Generate quick report"
        )
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        await self.start(update, context)
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        metrics = self._calculate_current_metrics()
        
        status_message = f"""
📊 *Current System Status*

🔢 Total Transactions: {metrics['total_transactions']:,}
❌ Failed Transactions: {metrics['failed_transactions']:,}
📉 Failure Rate: {metrics['failure_rate']:.1f}%
💰 Failed Amount: ${metrics['failed_amount']:,.2f}
🏢 Worst Branch: {metrics['worst_branch']} ({metrics['worst_branch_rate']:.1f}%)
⏰ Peak Failure Hour: {metrics['peak_hour']}:00

_Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        
        await update.message.reply_text(
            status_message,
            parse_mode='Markdown'
        )
    
    async def alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alerts command"""
        alerts = self._get_active_alerts()
        
        if not alerts:
            await update.message.reply_text("✅ No active alerts! System is operating normally.")
            return
        
        alert_message = "🚨 *Active Alerts*\n\n"
        for alert in alerts:
            emoji = "🔴" if alert['severity'] == 'high' else "🟡"
            alert_message += f"{emoji} {alert['message']}\n"
        
        await update.message.reply_text(alert_message, parse_mode='Markdown')
    
    async def report(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /report command"""
        report = self._generate_quick_report()
        
        await update.message.reply_text(
            report,
            parse_mode='Markdown'
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages"""
        user_id = update.effective_user.id
        message = update.message.text
        
        # Show typing indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        # Process with AI agent
        result = self.agent.query(message)
        
        if result['success']:
            response = result['response']
            
            # Format response for Telegram
            formatted_response = self._format_response(response)
            
            await update.message.reply_text(
                formatted_response,
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(
                "❌ Sorry, I encountered an error processing your request. Please try again."
            )
    
    def _calculate_current_metrics(self) -> Dict[str, Any]:
        """Calculate current system metrics"""
        branch_stats = self.df.groupby('branch_name')['is_failed'].agg(['count', 'sum', 'mean'])
        worst_branch = branch_stats['mean'].idxmax()
        
        hourly_failures = self.df[self.df['is_failed']].groupby('hour').size()
        peak_hour = hourly_failures.idxmax() if len(hourly_failures) > 0 else 0
        
        return {
            'total_transactions': len(self.df),
            'failed_transactions': self.df['is_failed'].sum(),
            'failure_rate': self.df['is_failed'].mean() * 100,
            'failed_amount': self.df[self.df['is_failed']]['transaction_amount'].sum(),
            'worst_branch': worst_branch,
            'worst_branch_rate': branch_stats.loc[worst_branch, 'mean'] * 100,
            'peak_hour': peak_hour
        }
    
    def _get_active_alerts(self) -> list:
        """Get currently active alerts"""
        alerts = []
        metrics = self._calculate_current_metrics()
        
        if metrics['failure_rate'] > 20:
            alerts.append({
                'severity': 'high',
                'message': f"High failure rate: {metrics['failure_rate']:.1f}%"
            })
        
        if metrics['worst_branch_rate'] > 25:
            alerts.append({
                'severity': 'high',
                'message': f"{metrics['worst_branch']} failure rate: {metrics['worst_branch_rate']:.1f}%"
            })
        
        return alerts
    
    def _generate_quick_report(self) -> str:
        """Generate a quick summary report"""
        metrics = self._calculate_current_metrics()
        
        report = f"""
📋 *Quick Performance Report*

*Overall Performance*
• Total Transactions: {metrics['total_transactions']:,}
• Failure Rate: {metrics['failure_rate']:.1f}%
• Revenue Impact: ${metrics['failed_amount']:,.2f}

*Top Issues*
1. Worst Branch: {metrics['worst_branch']} ({metrics['worst_branch_rate']:.1f}%)
2. Peak Failure Hour: {metrics['peak_hour']}:00

*Recommendations*
• Focus on {metrics['worst_branch']} branch
• Investigate issues during hour {metrics['peak_hour']}
• Consider implementing automated retry mechanisms

_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""
        return report
    
    def _format_response(self, response: str) -> str:
        """Format AI response for Telegram"""
        # Add basic Markdown formatting
        response = response.replace("**", "*")  # Convert bold
        
        # Limit response length for Telegram
        if len(response) > 4000:
            response = response[:3900] + "\n\n_...response truncated_"
        
        return response
    
    def run(self):
        """Run the Telegram bot"""
        application = Application.builder().token(self.bot_token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("help", self.help))
        application.add_handler(CommandHandler("status", self.status))
        application.add_handler(CommandHandler("alerts", self.alerts))
        application.add_handler(CommandHandler("report", self.report))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Run the bot
        application.run_polling()


class FinanceGuardWhatsAppBot:
    """WhatsApp bot for FinanceGuard AI using Twilio"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.agent = FinancialAnalysisAgent(df)
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Set up Flask routes for WhatsApp webhook"""
        
        @self.app.route("/whatsapp", methods=['POST'])
        def whatsapp_webhook():
            incoming_msg = request.values.get('Body', '').lower()
            from_number = request.values.get('From', '')
            
            resp = MessagingResponse()
            msg = resp.message()
            
            # Process different commands
            if 'help' in incoming_msg or 'start' in incoming_msg:
                response = self.get_help_message()
            elif 'status' in incoming_msg:
                response = self.get_status_message()
            elif 'alert' in incoming_msg:
                response = self.get_alerts_message()
            elif 'report' in incoming_msg:
                response = self.get_report_message()
            else:
                # Process with AI agent
                response = self.process_ai_query(incoming_msg)
            
            msg.body(response)
            return str(resp)
    
    def get_help_message(self) -> str:
        """Get help message for WhatsApp"""
        return """
🤖 Welcome to FinanceGuard AI!

I can help you analyze transaction data. Try:
• "status" - Current system status
• "alerts" - Show active alerts
• "report" - Quick performance report
• Or ask any question about the data!

Examples:
- What's the failure rate?
- Which branch has most failures?
- Show today's performance
"""
    
    def get_status_message(self) -> str:
        """Get current status for WhatsApp"""
        metrics = self._calculate_current_metrics()
        
        return f"""
📊 Current Status

Transactions: {metrics['total_transactions']:,}
Failed: {metrics['failed_transactions']:,}
Failure Rate: {metrics['failure_rate']:.1f}%
Failed Amount: ${metrics['failed_amount']:,.2f}
Worst Branch: {metrics['worst_branch']}
"""
    
    def get_alerts_message(self) -> str:
        """Get active alerts for WhatsApp"""
        alerts = self._get_active_alerts()
        
        if not alerts:
            return "✅ No active alerts!"
        
        message = "🚨 Active Alerts:\n"
        for alert in alerts:
            message += f"• {alert['message']}\n"
        
        return message
    
    def get_report_message(self) -> str:
        """Get quick report for WhatsApp"""
        metrics = self._calculate_current_metrics()
        
        return f"""
📋 Quick Report

Performance:
• Failure Rate: {metrics['failure_rate']:.1f}%
• Failed Amount: ${metrics['failed_amount']:,.2f}

Issues:
• Worst Branch: {metrics['worst_branch']}
• Peak Hour: {metrics['peak_hour']}:00

Action: Focus on {metrics['worst_branch']}
"""
    
    def process_ai_query(self, query: str) -> str:
        """Process query with AI agent"""
        result = self.agent.query(query)
        
        if result['success']:
            # Truncate response for WhatsApp
            response = result['response']
            if len(response) > 1500:
                response = response[:1450] + "..."
            return response
        else:
            return "Sorry, I couldn't process that request. Please try again."
    
    def _calculate_current_metrics(self) -> Dict[str, Any]:
        """Calculate current metrics"""
        # Same as Telegram bot method
        branch_stats = self.df.groupby('branch_name')['is_failed'].agg(['count', 'sum', 'mean'])
        worst_branch = branch_stats['mean'].idxmax()
        
        hourly_failures = self.df[self.df['is_failed']].groupby('hour').size()
        peak_hour = hourly_failures.idxmax() if len(hourly_failures) > 0 else 0
        
        return {
            'total_transactions': len(self.df),
            'failed_transactions': self.df['is_failed'].sum(),
            'failure_rate': self.df['is_failed'].mean() * 100,
            'failed_amount': self.df[self.df['is_failed']]['transaction_amount'].sum(),
            'worst_branch': worst_branch,
            'worst_branch_rate': branch_stats.loc[worst_branch, 'mean'] * 100,
            'peak_hour': peak_hour
        }
    
    def _get_active_alerts(self) -> list:
        """Get active alerts"""
        alerts = []
        metrics = self._calculate_current_metrics()
        
        if metrics['failure_rate'] > 20:
            alerts.append({
                'severity': 'high',
                'message': f"High failure rate: {metrics['failure_rate']:.1f}%"
            })
        
        return alerts
    
    def run(self, port=5000):
        """Run the WhatsApp bot server"""
        self.app.run(debug=True, port=port)


# Bot runner script
def run_telegram_bot(df: pd.DataFrame):
    """Run the Telegram bot"""
    bot = FinanceGuardTelegramBot(df)
    bot.run()

def run_whatsapp_bot(df: pd.DataFrame, port=5000):
    """Run the WhatsApp bot"""
    bot = FinanceGuardWhatsAppBot(df)
    bot.run(port=port)

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(r'C:\Users\abdal\Downloads\jordan_transactions.csv')
    
    # Choose which bot to run
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "whatsapp":
        run_whatsapp_bot(df)
    else:
        run_telegram_bot(df)