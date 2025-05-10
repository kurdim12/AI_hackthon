# FinanceGuard AI - Chatbot & Notification System

This guide covers the chatbot and notification features of FinanceGuard AI.

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install python-telegram-bot twilio slack-sdk python-dotenv
   ```

2. **Run setup**:
   ```bash
   python setup_bots.py
   ```

3. **Start bot manager**:
   ```bash
   python run_bots.py
   ```

## 🤖 Telegram Bot

### Features
- Real-time system status
- Transaction analysis
- Active alerts monitoring
- Natural language queries
- Quick performance reports

### Commands
- `/start` - Start the bot and see help
- `/status` - View current system status
- `/alerts` - Check active alerts
- `/report` - Generate quick report
- Ask any question in natural language!

### Setup
1. Create a bot with [@BotFather](https://t.me/botfather) on Telegram
2. Get your bot token
3. Get your chat ID (send a message to your bot, then visit `https://api.telegram.org/bot<YourBOTToken>/getUpdates`)
4. Run `python setup_bots.py` and enter your credentials

### Example Usage
```
You: What's the failure rate at Mall A?
Bot: The failure rate at Mall A is 15.3% with 234 failed transactions out of 1,529 total.

You: /status
Bot: 📊 Current System Status
     🔢 Total Transactions: 10,234
     ❌ Failed Transactions: 1,825
     📉 Failure Rate: 17.8%
     ...
```

## 📱 WhatsApp Bot

### Features
- Status updates via WhatsApp
- Alert notifications
- Quick reports
- Natural language processing

### Setup
1. Create a Twilio account
2. Get a WhatsApp sandbox number
3. Configure webhook URL: `http://your-server:5000/whatsapp`
4. Run `python setup_bots.py` and enter Twilio credentials

### Commands
Send these messages to your WhatsApp bot:
- `status` - System status
- `alerts` - Active alerts
- `report` - Quick report
- Any question about your data

### Example Usage
```
You: status
Bot: 📊 Current Status
     Transactions: 10,234
     Failed: 1,825
     Failure Rate: 17.8%
     ...

You: which branch has most failures?
Bot: Branch North has the highest failure rate at 23.5% with 456 failed transactions.
```

## 🔔 Notification System

### Supported Platforms
- **Telegram** - Instant messages with charts
- **Slack** - Rich formatted messages
- **Email** - HTML emails with attachments
- **WhatsApp** - Text alerts via Twilio

### Alert Levels
- **Critical** 🚨 - Sent to all platforms
- **Warning** ⚠️ - Telegram and Slack
- **Info** ℹ️ - Slack only

### Features
- Multi-platform broadcasting
- Chart attachments
- Priority-based routing
- Customizable alert rules

### Usage Example
```python
# Send a critical alert
hub = NotificationHub()
alert_manager = AlertManager(hub)

await alert_manager.send_alert(
    level="critical",
    message="Failure rate exceeded 25%",
    metrics={
        "Failure Rate": 26.5,
        "Failed Count": 2650
    }
)
```

## 📝 Environment Variables

Create a `.env` file with:

```env
# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# WhatsApp (Twilio)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_WHATSAPP_NUMBER=your_whatsapp_number

# Slack
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_WEBHOOK_URL=your_webhook_url

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_RECIPIENTS=alert1@email.com,alert2@email.com

# OpenAI (optional)
OPENAI_API_KEY=your_openai_key
```

## 🛠️ Bot Manager Features

The `run_bots.py` script provides:

1. **Start Telegram Bot** - Run the Telegram bot
2. **Start WhatsApp Bot** - Run the WhatsApp server
3. **Test Notifications** - Test all configured channels
4. **Send Custom Alert** - Send alerts manually
5. **Environment Check** - Verify configuration

## 🧪 Testing

### Test Notifications
```python
# Test all channels
python run_bots.py
# Choose option 3 (Test Notifications)
```

### Test Individual Platforms
```python
from notification_integrations import NotificationHub
import asyncio

hub = NotificationHub()

# Test Telegram
await hub.send_telegram("Test message")

# Test Slack
await hub.send_slack("Test message")

# Test Email
await hub.send_email(
    subject="Test",
    body="Test email",
    recipients=["test@example.com"]
)
```

## 📊 Chart Attachments

The notification system can include charts:

```python
chart_data = {
    "type": "bar",
    "x": ["Branch A", "Branch B", "Branch C"],
    "y": [15.5, 22.3, 18.7],
    "title": "Branch Failure Rates",
    "xlabel": "Branch",
    "ylabel": "Failure Rate %"
}

await hub.send_telegram(
    message="Performance Report",
    include_chart=True,
    chart_data=chart_data
)
```

## 🔧 Troubleshooting

### Telegram Bot Not Responding
1. Check bot token is correct
2. Ensure bot is not blocked
3. Verify chat ID
4. Check internet connection

### WhatsApp Not Receiving Messages
1. Verify Twilio credentials
2. Check webhook URL configuration
3. Ensure phone number format is correct (+1234567890)
4. Check Twilio account balance

### Notifications Not Sending
1. Run environment check: `python run_bots.py` → Option 5
2. Verify all credentials in `.env`
3. Check error messages in console
4. Test each platform individually

## 📚 Advanced Usage

### Custom Alert Rules
```python
# Define custom alert rules
alert_rules = {
    "high_failure": {
        "condition": lambda metrics: metrics['failure_rate'] > 20,
        "level": "critical",
        "message": "High failure rate detected",
        "platforms": ["telegram", "slack", "email"]
    }
}
```

### Scheduled Reports
```python
# Schedule daily reports
import schedule

def daily_report():
    asyncio.run(send_daily_summary())

schedule.every().day.at("09:00").do(daily_report)
```

## 🤝 Contributing

To add new notification platforms:

1. Add platform client in `NotificationHub`
2. Implement send method
3. Update `broadcast` method
4. Add configuration in `setup_bots.py`

## 📄 License

This project is licensed under the MIT License.

---

For more information about the full FinanceGuard AI system, see the main README.md file.