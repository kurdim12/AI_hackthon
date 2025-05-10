#!/usr/bin/env python
"""
Simple bot runner for FinanceGuard AI chatbots
"""

import os
import sys
import pandas as pd
from chatbot_integrations import run_telegram_bot, run_whatsapp_bot
from notification_integrations import NotificationHub, AlertManager
import asyncio

def print_menu():
    """Display menu options"""
    print("\n" + "=" * 50)
    print("FinanceGuard AI - Bot Manager")
    print("=" * 50)
    print("1. Run Telegram Bot")
    print("2. Run WhatsApp Bot")
    print("3. Test Notifications")
    print("4. Send Alert")
    print("5. Exit")
    print("=" * 50)

def load_data():
    """Load transaction data"""
    file_path = r'C:\Users\abdal\Downloads\jordan_transactions.csv'
    
    if not os.path.exists(file_path):
        print(f"Data file not found at: {file_path}")
        print("Please ensure the file exists or generate sample data.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} transactions successfully.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

async def test_notifications():
    """Test notification system"""
    hub = NotificationHub()
    
    print("\nTesting notification channels...")
    
    # Test Telegram
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        print("Testing Telegram...")
        await hub.send_telegram("🧪 Test notification from FinanceGuard AI")
        print("✓ Telegram test sent")
    else:
        print("✗ Telegram not configured")
    
    # Test Slack
    if os.getenv("SLACK_BOT_TOKEN"):
        print("Testing Slack...")
        await hub.send_slack("Test notification", attachments=[{
            "color": "#36a64f",
            "title": "Test Alert",
            "text": "This is a test notification from FinanceGuard AI"
        }])
        print("✓ Slack test sent")
    else:
        print("✗ Slack not configured")
    
    # Test Email
    if os.getenv("SMTP_USER"):
        print("Testing Email...")
        await hub.send_email(
            subject="Test Alert",
            body="<h1>Test Notification</h1><p>This is a test email from FinanceGuard AI</p>",
            recipients=[os.getenv("SMTP_USER")]
        )
        print("✓ Email test sent")
    else:
        print("✗ Email not configured")
    
    print("\nNotification test complete!")

async def send_custom_alert():
    """Send a custom alert"""
    hub = NotificationHub()
    alert_manager = AlertManager(hub)
    
    print("\nSend Custom Alert")
    print("-" * 30)
    
    # Get alert level
    level = input("Alert level (critical/warning/info): ").lower()
    if level not in ["critical", "warning", "info"]:
        level = "info"
    
    # Get message
    message = input("Alert message: ")
    
    # Get platforms
    platforms = input("Platforms (telegram,slack,email,whatsapp - default: all): ")
    if platforms:
        platforms = [p.strip() for p in platforms.split(",")]
    else:
        platforms = None
    
    # Send alert
    print(f"\nSending {level} alert...")
    await alert_manager.send_alert(level, message)
    print("Alert sent successfully!")

def check_environment():
    """Check if required environment variables are set"""
    required_vars = {
        "Telegram": ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"],
        "WhatsApp": ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_NUMBER"],
        "Slack": ["SLACK_BOT_TOKEN"],
        "Email": ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD"]
    }
    
    print("\nEnvironment Configuration Status:")
    print("-" * 35)
    
    for service, vars_list in required_vars.items():
        configured = all(os.getenv(var) is not None for var in vars_list)
        status = "✓ Configured" if configured else "✗ Not configured"
        print(f"{service:<15} {status}")
        
        if not configured:
            missing = [var for var in vars_list if not os.getenv(var)]
            print(f"  Missing: {', '.join(missing)}")
    
    print()

def main():
    """Main function"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check environment configuration
    check_environment()
    
    # Load data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            # Run Telegram bot
            if not os.getenv("TELEGRAM_BOT_TOKEN"):
                print("\n❌ Telegram bot token not configured!")
                print("Add TELEGRAM_BOT_TOKEN to your .env file")
                continue
            
            print("\nStarting Telegram bot...")
            print("Press Ctrl+C to stop")
            try:
                run_telegram_bot(df)
            except KeyboardInterrupt:
                print("\nTelegram bot stopped.")
        
        elif choice == '2':
            # Run WhatsApp bot
            if not os.getenv("TWILIO_ACCOUNT_SID"):
                print("\n❌ Twilio credentials not configured!")
                print("Add TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN to your .env file")
                continue
            
            print("\nStarting WhatsApp bot...")
            print("Server will run on http://localhost:5000/whatsapp")
            print("Configure this URL in your Twilio WhatsApp webhook")
            print("Press Ctrl+C to stop")
            try:
                run_whatsapp_bot(df)
            except KeyboardInterrupt:
                print("\nWhatsApp bot stopped.")
        
        elif choice == '3':
            # Test notifications
            asyncio.run(test_notifications())
        
        elif choice == '4':
            # Send custom alert
            asyncio.run(send_custom_alert())
        
        elif choice == '5':
            # Exit
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()