# GW150914 MCP Signal Search

A gravitational wave signal detection and optimization system using the Model Context Protocol (MCP) for efficient parameter space exploration.

## 🌊 Overview

This project provides tools for analyzing gravitational wave signals, specifically focusing on the GW150914 event. It uses MCP to create a client-server architecture where:

- **GW Analysis Server** (`mcp-server/gw_analysis_server.py`): Provides gravitational wave analysis tools via MCP
- **GW Optimization Client** (`mcp-client/gw_optimization_client.py`): Uses AI to optimize signal detection parameters

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- [UV](https://github.com/astral-sh/uv) package manager

Install UV if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd gw150914-mcp-signal-search
make setup
```

2. **Configure environment variables:**
```bash
cp env.template .env
# Edit .env with your OpenAI API key
```

3. **Install dependencies:**
```bash
make install-dev
```

### Running the System

1. **Start the analysis server andthe optimization client:**
```bash
make run-client SERVER_PATH=mcp-server/gw_analysis_server.py
```

2. **Or run a complete demo (includes both server and client):**
```bash
make demo
```

## 📋 Available Commands

Run `make help` to see all available commands:

### Installation & Setup
- `make install` - Install production dependencies
- `make install-dev` - Install with development dependencies  
- `make setup` - Complete development setup
- `make update` - Update all dependencies

### Code Quality
- `make format` - Format code with black and isort
- `make lint` - Run linting checks
- `make check` - Run all code quality checks
- `make test` - Run tests

### Running Applications
- `make run-server` - Start the GW analysis server
- `make run-client SERVER_PATH=<path>` - Start the optimization client
- `make demo` - Run complete demo

### Development
- `make notebook` - Start Jupyter notebook
- `make dev-shell` - Enter development shell
- `make clean` - Clean temporary files

### Data Management
- `make data-info` - Show information about data files
- `make clean-data` - Clean temporary data files (keeps strain data)
- `make backup-data` - Create backup of analysis results and logs

## 🔧 Project Structure

```
gw150914-mcp-signal-search/
├── data/                            # Gravitational wave data and analysis results
│   ├── H1-1126259446-1126259478.txt    # LIGO Hanford strain data (GW150914)
│   ├── L1-1126259446-1126259478.txt    # LIGO Livingston strain data (GW150914)
│   ├── matched_filter_records.jsonl    # Analysis results and optimization history
│   └── logs/                            # Application logs (auto-generated)
│       ├── server_YYYYMMDD_HHMMSS.log      # Server execution logs
│       ├── client_YYYYMMDD_HHMMSS.log      # Client execution logs
│       └── demo_YYYYMMDD_HHMMSS.log        # Combined demo logs
├── mcp-client/
│   └── gw_optimization_client.py    # AI-powered optimization client
├── mcp-server/
│   └── gw_analysis_server.py        # GW analysis MCP server
├── pyproject.toml                   # Project configuration
├── Makefile                         # Build and deployment commands
├── env.template                     # Environment variables template
└── README.md                        # This file
```

## 📊 Data Directory

The `data/` directory contains gravitational wave strain data and analysis results for the GW150914 event:

### Strain Data Files
- **`H1-1126259446-1126259478.txt`**: LIGO Hanford (H1) detector strain data
  - GPS time range: 1126259446 to 1126259478 (32 seconds around GW150914)
  - Format: Two columns - GPS time (seconds) and strain amplitude
  - Sample rate: 4096 Hz
  - Data points: ~131,072 samples per detector

- **`L1-1126259446-1126259478.txt`**: LIGO Livingston (L1) detector strain data
  - Same format and time range as H1 data
  - Contains the gravitational wave signal from the binary black hole merger

### Analysis Results
- **`matched_filter_records.jsonl`**: JSON Lines file containing optimization results
  - Each line represents one matched filter analysis result
  - Contains SNR values, template parameters, detector responses
  - Tracks optimization history and parameter exploration
  - Used by the AI client to learn from previous attempts

### Application Logs
- **`logs/`**: Directory containing timestamped execution logs (auto-generated)
  - **`server_YYYYMMDD_HHMMSS.log`**: Server execution logs with timestamps
  - **`client_YYYYMMDD_HHMMSS.log`**: Client optimization logs and AI interactions
  - **`demo_YYYYMMDD_HHMMSS.log`**: Combined logs from demo runs
  - All stdout and stderr output is captured with timestamps
  - Logs are automatically created when running `make run-server`, `make run-client`, or `make demo`

### Data Format Details

**Strain Data Format:**
```
GPS_TIME    STRAIN_AMPLITUDE
1.126259446000000000e+09    2.177040281448404468e-19
1.126259446000244141e+09    2.087638998822567751e-19
...
```

**Analysis Results Format:**
```json
{
  "timestamp": "010725",
  "max_network_snr": 7.3055496414353565,
  "max_snr_time": 1126259462.4360352,
  "template_parameters": {
    "mass1": 17.0,
    "mass2": 9.5,
    "ra": 0.94,
    "dec": -0.31
  },
  "detector_snrs_at_max": {
    "H1": {"snr_abs": 5.064289109514169},
    "L1": {"snr_abs": 5.2653614670535624}
  }
}
```

### Data Usage
- The strain data is automatically loaded by the MCP server for analysis
- Results are cached to `/tmp/gw-mcp/` during processing
- The optimization client uses historical results to guide parameter searches
- Data can be visualized using the server's plotting tools
- **All execution logs are automatically saved** to `data/logs/` with timestamps for debugging and analysis

## 🧪 Features

### GW Analysis Server Tools
- **Data Fetching**: Download and preprocess GW data from GWOSC
- **Matched Filter Search**: Perform template-based signal detection
- **Network Analysis**: Multi-detector coherent analysis
- **Visualization**: Generate plots and analysis results

### Optimization Client
- **AI-Powered Search**: Uses OpenAI GPT for intelligent parameter exploration
- **4D Parameter Space**: Optimizes mass1, mass2, right ascension, declination
- **Convergence Detection**: Automatically stops when optimal SNR is found
- **Progress Tracking**: Visual feedback during optimization

## 🔬 Scientific Background

This project focuses on the GW150914 event, the first direct detection of gravitational waves:
- **Event Time**: GPS 1126259462.427
- **Source**: Binary black hole merger (~36 + 29 solar masses)
- **Detectors**: LIGO Hanford (H1) and LIGO Livingston (L1)
- **Peak SNR**: ~24 in the network

## 📈 Results & Validation

This project was developed and validated at the **AI for Science Hackathon (Beijing)** with promising results demonstrating the effectiveness of LLM-agent powered gravitational wave detection.

### Key Findings

![Project Overview](docs/images/hackathon-title-slide.png)
*Project presentation at AI for Science Hackathon (Beijing) - LLM-agent powered gravitational wave detection and scientific discovery*

#### Reliability Analysis
The system demonstrates consistent convergence behavior across different initial conditions:

![Reliability Analysis](docs/images/reliability-analysis.png)
*Parameter evolution and convergence analysis showing gradual convergence to true values during exploration. The system shows consistent performance under different initial conditions with average iterations of 13.3 ± 3.5 to reach maximum SNR of 19.56.*

#### Detection Performance
The automated system successfully detects and analyzes gravitational wave signals:

![Detection Results](docs/images/detection-results.png)
*Network SNR time series showing successful detection of GW150914 signal. The system achieves peak network SNR of 19.6 at GPS time 1126259462.428, demonstrating automated GW data detection with LLM-agent validation on real GW data.*

### Validation Results
- **Convergence**: Parameters gradually converge to true values during exploration
- **Consistency**: Model shows consistent performance under different initial conditions  
- **Accuracy**: Successfully achieves maximum SNR of ~19.6, matching theoretical expectations
- **Automation**: Fully automated detection pipeline validated on real LIGO data

### Technical Report
For detailed methodology, results, and analysis, see: [`docs/Hackathon.pdf`](docs/Hackathon.pdf)

*Presented by He Wang (ICTP-AP, UCAS) and Yiming Dong (PKU) at AI for Science Hackathon (Beijing), August 24, 2025*

## 🛠️ Development

### Setting up Development Environment

```bash
# Complete development setup
make setup

# Install pre-commit hooks
make pre-commit-install

# Run tests
make test

# Check code quality
make check
```

### Dependencies

Core dependencies:
- `mcp`: Model Context Protocol framework
- `numpy`, `scipy`: Scientific computing
- `gwpy`, `pycbc`: Gravitational wave analysis
- `openai`: AI integration
- `matplotlib`: Visualization

## 📊 Usage Examples

### Basic Optimization
```python
# The client will automatically optimize parameters for GW150914
python mcp-client/gw_optimization_client.py mcp-server/gw_analysis_server.py
```

### Custom Parameter Ranges
Edit the optimization query in the client to specify different:
- Mass ranges (currently 10-80 solar masses)
- Sky location ranges (RA: 0-2π, Dec: -π/2 to π/2)
- Convergence criteria

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `make check` to ensure code quality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📚 Learning Resources
### MCP (Model Context Protocol) Resources
- **[Official MCP Documentation](https://modelcontextprotocol.io/docs/getting-started/intro)** - Complete getting started guide and API reference
- **[DeepLearning.AI MCP Course](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/)** - "MCP: Build Rich-Context AI Apps with Anthropic" - comprehensive hands-on course by Elie Schoppik
- **[Tutorial on Building AI Scientist Agents with Model Context Protocol](https://drive.google.com/file/d/1rg0ZZjU-OgPZ4ZGvfYJPWgapbLeqKz5j/view?pli=1)** - Comprehensive tutorial on building AI scientist agents using MCP framework
- **[MCP.Science Repository](https://github.com/pathintegral-institute/mcp.science)** - Open source MCP servers for scientific research with 65+ stars
- **[How to Build Your Own MCP Server](https://github.com/pathintegral-institute/mcp.science/blob/main/docs/how-to-build-your-own-mcp-server-step-by-step.md)** - Step-by-step tutorial for creating custom MCP servers
- **[Integrate MCP Server Guide](https://github.com/pathintegral-institute/mcp.science/blob/main/docs/integrate-mcp-server-step-by-step.md)** - Complete integration guide for MCP servers with AI applications

### Scientific Computing & AI Resources
- **[AI4Science Events](https://ai4.science/events/stanford-quantum-science-hackathon)** - Stanford Quantum Science Hackathon and other AI for science events
- **[AI-4-Science Organization](https://ai-4-science.org/)** - Community and resources for AI applications in scientific research
- **[Project Technical Report](docs/Hackathon.pdf)** - Detailed methodology and results from AI for Science Hackathon (Beijing) validation

### Why These Resources Matter
This project demonstrates practical applications of MCP in scientific computing, specifically for gravitational wave analysis. The resources above provide:

- **Foundation Knowledge**: Understanding MCP architecture and implementation
- **Hands-on Learning**: Building and deploying MCP servers and clients
- **Scientific Context**: Real-world applications of AI in physics and astronomy
- **Community**: Connect with researchers and developers working on similar projects

### Getting Started with MCP
1. Start with the [official MCP documentation](https://modelcontextprotocol.io/docs/getting-started/intro) for core concepts
2. Take the [DeepLearning.AI course](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/) for practical implementation
3. Explore the [MCP.Science repository](https://github.com/pathintegral-institute/mcp.science) for scientific use cases
4. Use this project as a template for your own gravitational wave or scientific analysis applications

## 🙏 Acknowledgments

- LIGO Scientific Collaboration for gravitational wave data
- PyCBC and GWpy communities for analysis tools
- Anthropic for MCP framework and protocol development
- Path Integral Institute for MCP.Science open source servers
- DeepLearning.AI and Elie Schoppik for educational resources
