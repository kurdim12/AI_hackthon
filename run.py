import os
import sys
import subprocess

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'python-dotenv',
        'langchain',
        'langchain-experimental',
        'langchain-openai',
        'openai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        
        print("\nInstalling missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)

def check_data_file():
    """Check if data file exists"""
    primary_path = r'C:\Users\abdal\Downloads\jordan_transactions.csv'
    
    if not os.path.exists(primary_path):
        print(f"Data file not found at: {primary_path}")
        print("\nWould you like to generate sample data? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print("Generating sample data...")
            import generate_sample_data
            generate_sample_data.generate_sample_transactions()
            print("Sample data generated successfully!")
        else:
            print("Please ensure jordan_transactions.csv is available at the specified path.")
            sys.exit(1)

def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print("No .env file found.")
        print("Creating a template .env file...")
        
        template = """# FinanceGuard AI Environment Configuration
# Add your OpenAI API key below

OPENAI_API_KEY=your_openai_api_key_here
"""
        
        with open('.env', 'w') as f:
            f.write(template)
        
        print(".env file created. Please add your OpenAI API key.")
        print("Note: AI features will be disabled until you add your API key.")

def run_app():
    """Run the Streamlit application"""
    print("Starting FinanceGuard AI...")
    print("Opening in your browser at http://localhost:8501")
    
    try:
        subprocess.run(['streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\nShutting down FinanceGuard AI...")
        sys.exit(0)

def main():
    print("=" * 50)
    print("FinanceGuard AI - Retail Intelligence Platform")
    print("=" * 50)
    
    # Check requirements
    print("Checking requirements...")
    check_requirements()
    
    # Check data file
    print("\nChecking data file...")
    check_data_file()
    
    # Check environment file
    print("\nChecking environment configuration...")
    check_env_file()
    
    # Run the application
    print("\n" + "=" * 50)
    run_app()

if __name__ == "__main__":
    main()