# FinanceGuard AI - Run Instructions

## 🚀 Quick Start

The proper way to run this Streamlit application is:

```bash
python -m streamlit run app.py
```

Or simply:

```bash
streamlit run app.py
```

## 📋 Step-by-Step Instructions

### 1. Ensure Dependencies are Installed

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Application

```bash
# Using Python module
python -m streamlit run app.py

# Or using streamlit directly
streamlit run app.py

# With custom port
streamlit run app.py --server.port 8502

# With custom address
streamlit run app.py --server.address 0.0.0.0

# For development (auto-reload)
streamlit run app.py --server.runOnSave true
```

## 🔧 Alternative Run Methods

### Using the Launcher Script

```bash
# This script checks requirements before running
python run.py
```

### Direct Streamlit Commands

```bash
# Default run
streamlit run app.py

# With specific configuration
streamlit run app.py \
  --server.port 8501 \
  --server.address localhost \
  --browser.serverAddress localhost \
  --browser.serverPort 8501
```

### For Production

```bash
# Run with production settings
streamlit run app.py \
  --server.headless true \
  --server.port 80 \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection true
```

## 🌐 Access the Application

Once running, the application will be available at:

- Local: http://localhost:8501
- Network: http://YOUR_IP:8501

The browser should open automatically. If not, manually navigate to the URL shown in the terminal.

## 🛠️ Common Issues

### Port Already in Use

```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Module Not Found

```bash
# Ensure you're in the correct directory
cd path/to/financeguard-ai

# Install dependencies
pip install -r requirements.txt
```

### Streamlit Not Found

```bash
# Install streamlit specifically
pip install streamlit

# Or reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

## 📝 Configuration Options

### Streamlit CLI Options

```bash
# Show all available options
streamlit run --help

# Common options:
--server.port PORT                 # The port to run on
--server.address ADDRESS           # The address to bind to
--server.headless BOOL            # Run in headless mode
--server.runOnSave BOOL           # Auto-reload on file save
--browser.serverAddress ADDRESS    # Browser address
--browser.gatherUsageStats BOOL   # Disable telemetry
--global.developmentMode BOOL     # Development mode
```

### Environment Variables

You can also set Streamlit configuration via environment variables:

```bash
export STREAMLIT_SERVER_PORT=8502
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

streamlit run app.py
```

## 🔄 Development Mode

For development with hot reloading:

```bash
# Enable auto-reload
streamlit run app.py --server.runOnSave true

# With debug logging
streamlit run app.py --logger.level debug
```

## 🐳 Docker Run (Optional)

If using Docker:

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

```bash
# Build and run
docker build -t financeguard-ai .
docker run -p 8501:8501 financeguard-ai
```

## 📊 Resource Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Network**: Stable internet for API calls

## 🔍 Monitoring

To monitor the application:

```bash
# Run with metrics
streamlit run app.py --server.enableStaticServing true

# Check resource usage
# Linux/Mac
top -p $(pgrep -f "streamlit run")

# Windows
tasklist /FI "IMAGENAME eq python.exe"
```

## 📞 Troubleshooting

If you encounter issues:

1. Check Python version: `python --version` (should be 3.8+)
2. Verify Streamlit installation: `streamlit --version`
3. Check current directory: `pwd` or `cd`
4. Review error messages in terminal
5. Ensure `.env` file exists with valid API key

## 🚦 Stopping the Application

To stop the Streamlit application:

- Press `Ctrl+C` in the terminal
- Or close the terminal window

## 🔗 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Configuration](https://docs.streamlit.io/library/advanced-features/configuration)
- [Deployment Options](https://docs.streamlit.io/streamlit-community-cloud/get-started)

---

For more help, check the main README.md or create an issue on GitHub.