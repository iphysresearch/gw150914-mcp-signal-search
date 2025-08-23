import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # Get OpenAI configuration from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        # Validate required API key
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required. Please check your .env file.")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Load OpenAI model configuration from environment
        self.model = os.getenv("MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("TEMPERATURE", "1.0"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "10000"))
        
        # Load optimization parameters from environment
        self.max_iterations = int(os.getenv("MAX_OPTIMIZATION_ITERATIONS", "30"))
        self.convergence_threshold = float(os.getenv("SNR_CONVERGENCE_THRESHOLD", "0.005"))
        
        # Load GW data configuration from environment
        self.gps_start = int(os.getenv("GW150914_GPS_START", "1126259446"))
        self.gps_end = int(os.getenv("GW150914_GPS_END", "1126259478"))
        self.default_detectors = [det.strip() for det in os.getenv("DEFAULT_DETECTORS", "H1,L1").split(",")]
        
        # Load temporary directory configuration
        self.temp_data_dir = os.getenv("TEMP_DATA_DIR", "./data/")
        
        # Debug and logging configuration from environment
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.verbose_logging = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
        
        # MCP Server configuration
        self.mcp_log_level = os.getenv("MCP_SERVER_LOG_LEVEL", "INFO")
        
        if self.verbose_logging:
            print(f"MCPClient initialized with environment variables:")
            print(f"  OpenAI API Configuration:")
            print(f"    Model: {self.model}")
            print(f"    Temperature: {self.temperature}")
            print(f"    Max tokens: {self.max_tokens}")
            print(f"    Base URL: {base_url or 'Default OpenAI API'}")
            print(f"  Optimization Parameters:")
            print(f"    Max iterations: {self.max_iterations}")
            print(f"    Convergence threshold: {self.convergence_threshold}")
            print(f"  GW Data Configuration:")
            print(f"    GPS range: {self.gps_start} - {self.gps_end}")
            print(f"    Default detectors: {self.default_detectors}")
            print(f"    Temp data directory: {self.temp_data_dir}")
            print(f"  Logging Configuration:")
            print(f"    Debug mode: {self.debug_mode}")
            print(f"    Verbose logging: {self.verbose_logging}")
            print(f"    MCP log level: {self.mcp_log_level}")

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        # Convert MCP tools to OpenAI function calling format
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # print(f"Available tools: {available_tools}")
        # Initial Claude API call
        if self.debug_mode:
            print(f"Making API call with model: {self.model}, temperature: {self.temperature}, max_tokens: {self.max_tokens}")
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        tool_results = []
        final_text = []
        
        message = response.choices[0].message
        
        # Add assistant message to conversation
        if message.content:
            final_text.append(message.content)
            
        # Handle tool calls if any
        if message.tool_calls:
            # Add assistant message with tool calls to conversation
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })
            
            # Execute each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)  # Parse JSON string to dict
                
                # Execute tool call via MCP
                result = await self.session.call_tool(tool_name, tool_args)
                tool_results.append({"call": tool_name, "result": result})
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content)
                })
            
            # Get next response with tool results
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=messages,
            )
            
            final_text.append(response.choices[0].message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        # Create detector list string for the query
        detector_list = ", ".join([f'"{det}"' for det in self.default_detectors])
        
        # Automated optimization query using environment variables
        query = f"""You are a gravitational wave signal detection optimization agent.

Please optimize the matched filter search by maximizing the SNR using the complete_network_gw_matched_filter_search tool with four input variables: mass1 (m1), mass2 (m2), ra (right ascension), and dec (declination).

Use the GW150914 event data with GPS time range {self.gps_start} to {self.gps_end} and detectors {detector_list}.

Search parameters:
- mass1, mass2: 10-80 solar masses
- ra: 0 to 2π radians
- dec: -π/2 to π/2 radians

Provide optimized parameters within {self.max_iterations} iterations. Terminate early if maximum SNR fluctuates by no more than {self.convergence_threshold*100:.1f}% over 3 consecutive iterations.

Use the canvas to record your 4D exploration history. Before each attempt, reflect on past results to guide your search strategy. Start near the known GW150914 values (m1≈16, m2≈9, ra≈0.95, dec≈-0.27) and use the antenna pattern responses (fp, fc) from results to refine sky location parameters.

Plan your exploration and exploitation wisely across all four dimensions to maximize detection efficiency in the 4D parameter space."""
        
        iteration = 0
        snr_history = []
        param_history = []
        convergence_window = 3
        
        while iteration < self.max_iterations:
            try:
                print(f"\n=== Iteration {iteration + 1}/{self.max_iterations} ===")
                
                response = await self.process_query(query)
                if self.verbose_logging:
                    print(f"Full response: {response}")
                else:
                    print("\n" + response)
                
                # Extract SNR and parameters from response if available
                try:
                    import re
                    import json
                    import os
                    from datetime import datetime
                    
                    snr_match = re.search(r'max_network_snr["\']?\s*:\s*([0-9.]+)', response)
                    mass1_match = re.search(r'mass1["\']?\s*:\s*([0-9.]+)', response)
                    mass2_match = re.search(r'mass2["\']?\s*:\s*([0-9.]+)', response)
                    ra_match = re.search(r'ra["\']?\s*:\s*([0-9.-]+)', response)
                    dec_match = re.search(r'dec["\']?\s*:\s*([0-9.-]+)', response)
                    
                    if snr_match:
                        current_snr = float(snr_match.group(1))
                        snr_history.append(current_snr)
                        
                        # Extract parameters if available
                        current_params = {}
                        if mass1_match:
                            current_params['mass1'] = float(mass1_match.group(1))
                        if mass2_match:
                            current_params['mass2'] = float(mass2_match.group(1))
                        if ra_match:
                            current_params['ra'] = float(ra_match.group(1))
                        if dec_match:
                            current_params['dec'] = float(dec_match.group(1))
                        
                        param_history.append(current_params)
                        
                        print(f"Current SNR: {current_snr:.3f}")
                        if current_params:
                            print(f"Current parameters: {current_params}")
                        
                        # Save iteration results to disk
                        iteration_data = {
                            "iteration": iteration + 1,
                            "timestamp": datetime.now().isoformat(),
                            "snr": current_snr,
                            "parameters": current_params,
                            "snr_history": snr_history,
                            "param_history": param_history
                        }
                        
                        # Create results directory using temp_data_dir from environment
                        results_dir = os.path.join(self.temp_data_dir, "optimization_results")
                        os.makedirs(results_dir, exist_ok=True)
                        
                        # Save individual iteration file
                        iteration_file = os.path.join(results_dir, f"iteration_{iteration + 1:03d}.json")
                        with open(iteration_file, 'w') as f:
                            json.dump(iteration_data, f, indent=2)
                        
                        # Save complete history file
                        history_file = os.path.join(results_dir, "complete_history.json")
                        complete_history = {
                            "total_iterations": iteration + 1,
                            "last_updated": datetime.now().isoformat(),
                            "snr_history": snr_history,
                            "param_history": param_history,
                            "best_snr": max(snr_history),
                            "best_iteration": snr_history.index(max(snr_history)) + 1
                        }
                        with open(history_file, 'w') as f:
                            json.dump(complete_history, f, indent=2)
                        
                        print(f"Results saved to {iteration_file} and {history_file}")
                        
                        # Check for convergence
                        if len(snr_history) >= convergence_window:
                            recent_snrs = snr_history[-convergence_window:]
                            max_recent = max(recent_snrs)
                            min_recent = min(recent_snrs)
                            if max_recent > 0 and (max_recent - min_recent) / max_recent <= self.convergence_threshold:
                                print(f"\nConverged! SNR stable within {self.convergence_threshold*100:.1f}% over {convergence_window} iterations.")
                                print(f"Final SNR: {current_snr:.3f}")
                                if current_params:
                                    print(f"Final parameters: {current_params}")
                                break
                except Exception as e:
                    print(f"Could not extract SNR/parameters: {e}")
                
                iteration += 1
                
                # Update query for next iteration to include history
                if iteration < self.max_iterations:
                    # Prepare history summary
                    history_summary = []
                    recent_count = min(5, len(snr_history))
                    for i in range(recent_count):
                        idx = len(snr_history) - recent_count + i
                        snr_val = snr_history[idx]
                        params = param_history[idx] if idx < len(param_history) else {}
                        history_summary.append(f"SNR: {snr_val:.3f}, params: {params}")
                    
                    query = f"""Continue optimizing the gravitational wave matched filter search. 

This is iteration {iteration + 1}/{self.max_iterations}. 

Previous results (most recent {recent_count}):
{chr(10).join(history_summary)}

Based on your previous results, continue refining the parameters (mass1, mass2, ra, dec) to maximize the network SNR. Use your exploration history to guide the next parameter selection strategy."""
            except Exception as e:
                print(f"\nError in iteration {iteration + 1}: {str(e)}")
                iteration += 1
        
        if iteration >= self.max_iterations:
            print(f"\nReached maximum iterations ({self.max_iterations})")
            if snr_history:
                print(f"Best SNR achieved: {max(snr_history):.3f}")
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
async def main():
    if len(sys.argv) < 2:
        print("Usage: python gw_optimization_client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
