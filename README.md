# FinanceGuard AI - Retail Intelligence Platform

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red.svg" alt="Framework">
  <img src="https://img.shields.io/badge/AI-OpenAI-black.svg" alt="AI">
</div>

## 🚀 Overview

FinanceGuard AI is an advanced retail financial intelligence and automation platform designed to help businesses monitor, analyze, and optimize their transaction systems in real-time. Using cutting-edge AI and machine learning, the platform provides actionable insights, predictive analytics, and automated solutions for retail financial operations.

## ✨ Key Features

### 🤖 AI-Powered Analysis
- Natural language querying with OpenAI GPT models
- Real-time transaction analysis and insights
- Automated anomaly detection and pattern recognition
- Predictive failure prevention system

### 📊 Advanced Analytics
- Interactive real-time dashboards
- Branch performance comparisons
- Time-based pattern analysis
- Financial impact assessments

### 🔔 Multi-Channel Notifications
- Telegram bot integration
- WhatsApp notifications via Twilio
- Slack alerts
- Email notifications
- Customizable alert thresholds

### 🎯 Intelligent Features
- Predictive Failure Prevention (PFP) scoring
- Smart transaction routing
- Anomaly DNA pattern matching
- Risk assessment matrices

### 🏆 Gamification System
- Branch performance leaderboards
- Achievement tracking
- Competitive scoring system
- Performance milestones

## 📋 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Transaction data (CSV format)
- Optional: Telegram/WhatsApp/Slack credentials for notifications

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financeguard-ai.git
cd financeguard-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. Run the application:
```bash
python run.py
```

The application will open in your browser at `http://localhost:8501`

## 🗂️ Project Structure

```
financeguard-ai/
├── app.py                      # Main Streamlit application
├── run.py                      # Application launcher
├── data_processor.py           # Data processing and metrics
├── ai_agent.py                 # Core AI agent functionality
├── enhanced_ai_agent.py        # Advanced AI features
├── advanced_features.py        # PFP, routing, DNA systems
├── visualizations.py           # Chart generation
├── advanced_visualizations.py  # Complex visualizations
├── gamification.py            # Gamification system
├── chatbot_integrations.py    # Bot implementations
├── notification_integrations.py # Multi-channel notifications
├── workflow_automation.py      # Workflow automation
├── voice_recognition.py       # Voice interface with Arabic support
├── setup_bots.py              # Bot configuration helper
├── run_bots.py               # Bot manager
├── generate_sample_data.py    # Sample data generator
├── requirements.txt           # Python dependencies
├── BOT_README.md             # Bot documentation
└── README.md                 # This file
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional - Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_WHATSAPP_NUMBER=your_whatsapp_number
SLACK_BOT_TOKEN=your_slack_token
SLACK_WEBHOOK_URL=your_slack_webhook
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_RECIPIENTS=alert1@email.com,alert2@email.com
```

### Data Format

The application expects transaction data in CSV format with the following columns:

- `transaction_id`: Unique identifier
- `mall_name`: Shopping mall name
- `branch_name`: Store branch name
- `transaction_date`: Date and time (dd/mm/yyyy HH:MM)
- `tax_amount`: Tax amount
- `transaction_amount`: Total transaction value
- `transaction_type`: Sale or Refund
- `transaction_status`: Completed or Failed

## 🤖 AI Features

### Natural Language Processing
- Ask questions in plain English
- Get instant insights about your data
- AI-powered recommendations

Example queries:
- "What's the failure rate at Mall A?"
- "Which branch has the most failures?"
- "Show me trends for the last week"
- "Why are transactions failing?"

### Predictive Analytics
- Failure prediction scoring
- Risk level assessment
- Trend forecasting
- Anomaly pattern matching

## 📊 Visualizations

The platform includes various interactive charts:
- Failure rate heatmaps
- Daily transaction trends
- Branch performance comparisons
- Financial impact analysis
- Real-time risk radar
- Predictive timelines

## 🔔 Notification System

### Telegram Bot
- Real-time status updates
- Transaction analysis
- Alert notifications
- Natural language queries

### WhatsApp Integration
- Status reports via Twilio
- Alert notifications
- Quick performance summaries

### Slack & Email
- Rich formatted messages
- Chart attachments
- Priority-based routing

## 📱 Voice Interface

The platform includes an Arabic-enabled voice interface:
- Voice commands in Arabic
- Real-time translation
- Text-to-speech responses
- Natural language processing

## 🎮 Gamification

Branch performance gamification includes:
- Achievement system
- Leaderboards
- Performance milestones
- Competitive scoring

## 🧪 Testing

### Generate Sample Data

If you don't have transaction data, generate sample data:

```bash
python generate_sample_data.py
```

### Test Notifications

```bash
python run_bots.py
# Select option 3 (Test Notifications)
```

## 🚀 Advanced Features

### Predictive Failure Prevention (PFP)
- Real-time risk scoring
- Pattern-based predictions
- Proactive recommendations

### Smart Transaction Router
- Intelligent gateway selection
- Load balancing
- Retry strategies

### Anomaly DNA System
- Unique failure signatures
- Pattern matching
- Historical comparison

## 📈 Performance Optimization

The platform is optimized for:
- Real-time data processing
- Concurrent notifications
- Efficient chart rendering
- Responsive UI updates

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models
- Streamlit for the web framework
- Plotly for interactive visualizations
- The open-source community

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact: abdalrhmankurdi12@gmial.com
- Documentation: docs.financeguard-ai.com

---

<div align="center">
  <p>Built with ❤️ for the retail financial industry</p>
  <p>© 2025 FinanceGuard AI - Advanced Retail Intelligence Platform</p>
</div>
