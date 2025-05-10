#!/usr/bin/env python
"""
Setup script for FinanceGuard AI chatbots and notifications
"""

import os
import sys
from dotenv import set_key, load_dotenv

def create_env_file():
    """Create or update .env file"""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write("# FinanceGuard AI Bot Configuration\n\n")
        print("Created new .env file")
    else:
        print("Updating existing .env file")
    
    return env_file

def setup_telegram():
    """Setup Telegram bot configuration"""
    print("\n🤖 Telegram Bot Setup")
    print("-" * 30)
    
    token = input("Enter your Telegram Bot Token (from @BotFather): ")
    chat_id = input("Enter your Telegram Chat ID: ")
    
    if token and chat_id:
        env_file = create_env_file()
        set_key(env_file, "TELEGRAM_BOT_TOKEN", token)
        set_key(env_file, "TELEGRAM_CHAT_ID", chat_id)
        print("✅ Telegram bot configured successfully!")
    else:
        print("❌ Telegram setup skipped - missing token or chat ID")

def setup_whatsapp():
    """Setup WhatsApp bot configuration (Twilio)"""
    print("\n📱 WhatsApp Bot Setup (Twilio)")
    print("-" * 30)
    
    account_sid = input("Enter your Twilio Account SID: ")
    auth_token = input("Enter your Twilio Auth Token: ")
    whatsapp_number = input("Enter your Twilio WhatsApp Number (e.g., +14155238886): ")
    
    if account_sid and auth_token and whatsapp_number:
        env_file = create_env_file()
        set_key(env_file, "TWILIO_ACCOUNT_SID", account_sid)
        set_key(env_file, "TWILIO_AUTH_TOKEN", auth_token)
        set_key(env_file, "TWILIO_WHATSAPP_NUMBER", whatsapp_number)
        print("✅ WhatsApp bot configured successfully!")
    else:
        print("❌ WhatsApp setup skipped - missing credentials")

def setup_slack():
    """Setup Slack bot configuration"""
    print("\n💬 Slack Bot Setup")
    print("-" * 30)
    
    bot_token = input("Enter your Slack Bot Token: ")
    webhook_url = input("Enter your Slack Webhook URL (optional): ")
    
    if bot_token:
        env_file = create_env_file()
        set_key(env_file, "SLACK_BOT_TOKEN", bot_token)
        if webhook_url:
            set_key(env_file, "SLACK_WEBHOOK_URL", webhook_url)
        print("✅ Slack bot configured successfully!")
    else:
        print("❌ Slack setup skipped - missing bot token")

def setup_email():
    """Setup email notification configuration"""
    print("\n📧 Email Notification Setup")
    print("-" * 30)
    
    print("Default SMTP server is Gmail. Press Enter to use defaults.")
    smtp_host = input("SMTP Host (default: smtp.gmail.com): ") or "smtp.gmail.com"
    smtp_port = input("SMTP Port (default: 587): ") or "587"
    smtp_user = input("Your email address: ")
    smtp_password = input("Your email password/app password: ")
    
    if smtp_user and smtp_password:
        env_file = create_env_file()
        set_key(env_file, "SMTP_HOST", smtp_host)
        set_key(env_file, "SMTP_PORT", smtp_port)
        set_key(env_file, "SMTP_USER", smtp_user)
        set_key(env_file, "SMTP_PASSWORD", smtp_password)
        
        # Alert recipients
        recipients = input("Alert recipient emails (comma-separated): ")
        if recipients:
            set_key(env_file, "ALERT_EMAIL_RECIPIENTS", recipients)
        
        print("✅ Email notifications configured successfully!")
    else:
        print("❌ Email setup skipped - missing credentials")

def setup_openai():
    """Setup OpenAI API key"""
    print("\n🧠 OpenAI API Setup (for AI features)")
    print("-" * 30)
    
    api_key = input("Enter your OpenAI API Key (optional): ")
    
    if api_key:
        env_file = create_env_file()
        set_key(env_file, "OPENAI_API_KEY", api_key)
        print("✅ OpenAI API configured successfully!")
    else:
        print("⚠️  OpenAI setup skipped - AI features will be disabled")

def main():
    """Main setup function"""
    print("=" * 50)
    print("FinanceGuard AI - Bot & Notification Setup")
    print("=" * 50)
    
    print("\nThis script will help you configure:")
    print("• Telegram Bot")
    print("• WhatsApp Bot (via Twilio)")
    print("• Slack Notifications")
    print("• Email Notifications")
    print("• OpenAI API (optional)")
    
    print("\nYou can skip any section by pressing Enter without input.")
    
    # Setup each service
    setup_telegram()
    setup_whatsapp()
    setup_slack()
    setup_email()
    setup_openai()
    
    # Load and display configuration
    load_dotenv()
    
    print("\n" + "=" * 50)
    print("Configuration Summary")
    print("=" * 50)
    
    services = {
        "Telegram": os.getenv("TELEGRAM_BOT_TOKEN") is not None,
        "WhatsApp": os.getenv("TWILIO_ACCOUNT_SID") is not None,
        "Slack": os.getenv("SLACK_BOT_TOKEN") is not None,
        "Email": os.getenv("SMTP_USER") is not None,
        "OpenAI": os.getenv("OPENAI_API_KEY") is not None
    }
    
    for service, configured in services.items():
        status = "✅ Configured" if configured else "❌ Not configured"
        print(f"{service:<15} {status}")
    
    print("\n✨ Setup complete!")
    print("\nNext steps:")
    print("1. Run 'python run_bots.py' to start the bot manager")
    print("2. For Telegram: Start a chat with your bot and send /start")
    print("3. For WhatsApp: Configure Twilio webhook to http://localhost:5000/whatsapp")
    print("4. Test notifications using the bot manager")

if __name__ == "__main__":
    main()