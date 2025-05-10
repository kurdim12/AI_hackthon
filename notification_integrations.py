import os
import requests
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import json
from twilio.rest import Client  # For WhatsApp via Twilio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import matplotlib.pyplot as plt
import io
import base64
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class NotificationHub:
    """Centralized notification system for multiple platforms"""
    
    def __init__(self):
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.slack_client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
        self.twilio_client = None
        
        # Initialize Twilio for WhatsApp
        if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
            self.twilio_client = Client(
                os.getenv("TWILIO_ACCOUNT_SID"),
                os.getenv("TWILIO_AUTH_TOKEN")
            )
    
    async def send_telegram(self, message: str, chat_id: str = None, 
                          parse_mode: str = "HTML", 
                          include_chart: bool = False,
                          chart_data: Dict = None):
        """Send Telegram notification with optional chart"""
        if not self.telegram_token:
            print("Telegram token not configured")
            return
        
        chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        # Send text message
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode
        }
        
        response = requests.post(url, json=payload)
        
        # Send chart if requested
        if include_chart and chart_data:
            chart_buffer = self._create_chart(chart_data)
            
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
            files = {"photo": ("chart.png", chart_buffer, "image/png")}
            data = {"chat_id": chat_id, "caption": "Transaction Analysis"}
            
            requests.post(url, files=files, data=data)
    
    async def send_slack(self, message: str, channel: str = "#alerts",
                        attachments: List[Dict] = None,
                        include_chart: bool = False,
                        chart_data: Dict = None):
        """Send Slack notification with rich formatting"""
        try:
            if include_chart and chart_data:
                # Create chart and upload to Slack
                chart_buffer = self._create_chart(chart_data)
                response = self.slack_client.files_upload(
                    channels=channel,
                    file=chart_buffer,
                    filename="chart.png",
                    title="Transaction Analysis"
                )
            
            # Send message with attachments
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text=message,
                attachments=attachments or []
            )
            
        except SlackApiError as e:
            print(f"Slack error: {e.response['error']}")
    
    async def send_whatsapp(self, message: str, to_number: str,
                          media_url: str = None):
        """Send WhatsApp message via Twilio"""
        if not self.twilio_client:
            print("Twilio client not configured")
            return
        
        from_whatsapp = os.getenv("TWILIO_WHATSAPP_NUMBER")
        
        try:
            if media_url:
                message = self.twilio_client.messages.create(
                    body=message,
                    from_=f'whatsapp:{from_whatsapp}',
                    to=f'whatsapp:{to_number}',
                    media_url=[media_url]
                )
            else:
                message = self.twilio_client.messages.create(
                    body=message,
                    from_=f'whatsapp:{from_whatsapp}',
                    to=f'whatsapp:{to_number}'
                )
            
            return message.sid
            
        except Exception as e:
            print(f"WhatsApp error: {e}")
    
    async def send_email(self, subject: str, body: str, 
                        recipients: List[str],
                        include_chart: bool = False,
                        chart_data: Dict = None,
                        priority: str = "normal"):
        """Send email with optional attachments"""
        smtp_server = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        message = MIMEMultipart()
        message["From"] = smtp_user
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject
        
        # Set priority
        if priority == "high":
            message["X-Priority"] = "1"
            message["Importance"] = "high"
        
        # Add body
        message.attach(MIMEText(body, "html"))
        
        # Add chart if requested
        if include_chart and chart_data:
            chart_buffer = self._create_chart(chart_data)
            image = MIMEImage(chart_buffer.read())
            image.add_header('Content-Disposition', 'attachment', filename='chart.png')
            message.attach(image)
        
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, recipients, message.as_string())
        except Exception as e:
            print(f"Email error: {e}")
    
    def _create_chart(self, data: Dict) -> io.BytesIO:
        """Create a chart from data"""
        plt.figure(figsize=(10, 6))
        
        if data.get("type") == "line":
            plt.plot(data["x"], data["y"], marker='o')
            plt.title(data.get("title", "Transaction Analysis"))
            plt.xlabel(data.get("xlabel", "Time"))
            plt.ylabel(data.get("ylabel", "Value"))
        elif data.get("type") == "bar":
            plt.bar(data["x"], data["y"])
            plt.title(data.get("title", "Branch Performance"))
            plt.xlabel(data.get("xlabel", "Branch"))
            plt.ylabel(data.get("ylabel", "Failure Rate %"))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        plt.close()
        
        return buffer
    
    async def broadcast(self, message: str, platforms: List[str] = None,
                      priority: str = "normal", include_chart: bool = False,
                      chart_data: Dict = None):
        """Broadcast message to multiple platforms"""
        platforms = platforms or ["telegram", "slack", "email"]
        
        tasks = []
        
        if "telegram" in platforms:
            tasks.append(self.send_telegram(
                message, 
                include_chart=include_chart, 
                chart_data=chart_data
            ))
        
        if "slack" in platforms:
            attachments = [{
                "color": "#ff0000" if priority == "high" else "#36a64f",
                "title": "FinanceGuard Alert",
                "text": message,
                "footer": "FinanceGuard AI",
                "ts": int(datetime.now().timestamp())
            }]
            tasks.append(self.send_slack(
                message, 
                attachments=attachments,
                include_chart=include_chart,
                chart_data=chart_data
            ))
        
        if "email" in platforms:
            recipients = os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
            if recipients:
                tasks.append(self.send_email(
                    subject=f"FinanceGuard Alert - {priority.upper()}",
                    body=self._format_email_body(message),
                    recipients=recipients,
                    include_chart=include_chart,
                    chart_data=chart_data,
                    priority=priority
                ))
        
        if "whatsapp" in platforms:
            whatsapp_numbers = os.getenv("ALERT_WHATSAPP_NUMBERS", "").split(",")
            for number in whatsapp_numbers:
                tasks.append(self.send_whatsapp(message, number))
        
        # Execute all notifications concurrently
        await asyncio.gather(*tasks)
    
    def _format_email_body(self, message: str) -> str:
        """Format message for email"""
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px;">
                <h2 style="color: #1e3d59;">FinanceGuard AI Alert</h2>
                <p style="font-size: 16px; color: #333;">{message}</p>
                <hr style="border: 1px solid #ddd;">
                <p style="font-size: 12px; color: #666;">
                    Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
            </div>
        </body>
        </html>
        """

class AlertManager:
    """Manages different types of alerts and their routing"""
    
    def __init__(self, notification_hub: NotificationHub):
        self.hub = notification_hub
        self.alert_rules = self._load_alert_rules()
    
    def _load_alert_rules(self) -> Dict[str, Dict]:
        """Load alert routing rules"""
        return {
            "critical": {
                "platforms": ["telegram", "slack", "email", "whatsapp"],
                "priority": "high",
                "include_chart": True
            },
            "warning": {
                "platforms": ["telegram", "slack"],
                "priority": "normal",
                "include_chart": True
            },
            "info": {
                "platforms": ["slack"],
                "priority": "low",
                "include_chart": False
            }
        }
    
    async def send_alert(self, level: str, message: str, 
                        metrics: Dict[str, Any] = None):
        """Send alert based on level and rules"""
        rule = self.alert_rules.get(level, self.alert_rules["info"])
        
        # Prepare chart data if needed
        chart_data = None
        if rule["include_chart"] and metrics:
            chart_data = {
                "type": "bar",
                "x": list(metrics.keys()),
                "y": list(metrics.values()),
                "title": f"{level.upper()} Alert - Metrics",
                "xlabel": "Metric",
                "ylabel": "Value"
            }
        
        # Format message with emojis based on level
        emoji_map = {
            "critical": "🚨",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        
        formatted_message = f"{emoji_map.get(level, '')} {level.upper()}: {message}"
        
        # Send to configured platforms
        await self.hub.broadcast(
            formatted_message,
            platforms=rule["platforms"],
            priority=rule["priority"],
            include_chart=rule["include_chart"],
            chart_data=chart_data
        )

# Example usage in workflow
async def example_notification_workflow(df):
    """Example of how to use notifications in workflows"""
    hub = NotificationHub()
    alert_manager = AlertManager(hub)
    
    # Calculate metrics
    failure_rate = df['is_failed'].mean() * 100
    failed_amount = df[df['is_failed']]['transaction_amount'].sum()
    
    # Determine alert level
    if failure_rate > 25:
        level = "critical"
        message = f"Failure rate critically high: {failure_rate:.1f}%"
    elif failure_rate > 15:
        level = "warning"
        message = f"Failure rate elevated: {failure_rate:.1f}%"
    else:
        level = "info"
        message = f"System operating normally. Failure rate: {failure_rate:.1f}%"
    
    # Send alert with metrics
    metrics = {
        "Failure Rate %": failure_rate,
        "Failed Transactions": df['is_failed'].sum(),
        "Failed Amount $": failed_amount
    }
    
    await alert_manager.send_alert(level, message, metrics)

# Test function
async def test_notifications():
    """Test notification system"""
    hub = NotificationHub()
    
    # Test Telegram
    await hub.send_telegram("🧪 Test notification from FinanceGuard AI")
    
    # Test Slack
    await hub.send_slack("Test notification", attachments=[{
        "color": "#36a64f",
        "title": "Test Alert",
        "text": "This is a test notification from FinanceGuard AI"
    }])
    
    # Test Email
    await hub.send_email(
        subject="Test Alert",
        body="<h1>Test Notification</h1><p>This is a test email from FinanceGuard AI</p>",
        recipients=["test@example.com"]
    )
    
    # Test broadcast
    await hub.broadcast(
        "System-wide test notification",
        platforms=["telegram", "slack"],
        priority="normal"
    )

if __name__ == "__main__":
    # Run test
    asyncio.run(test_notifications())