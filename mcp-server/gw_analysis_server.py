from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ImageContent, BlobResourceContents
import logging
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global settings from environment variables
temp_dir = os.getenv("TEMP_DATA_DIR", "./data/")
# Ensure temp_dir ends with slash for consistency
if not temp_dir.endswith('/'):
    temp_dir += '/'
json_file_path = os.path.join(temp_dir, "matched_filter_records.jsonl")

# GW150914 configuration from environment
GW150914_GPS_START = int(os.getenv("GW150914_GPS_START", "1126259446"))
GW150914_GPS_END = int(os.getenv("GW150914_GPS_END", "1126259478"))
SIGNAL_GPS_TIME = float(os.getenv("SIGNAL_GPS_TIME", "1126259462.427"))  # Known merger time

# Default detectors from environment
DEFAULT_DETECTORS = os.getenv("DEFAULT_DETECTORS", "H1,L1").split(",")

# Debug and logging configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"

# Advanced analysis configuration from environment
WAVEFORM_APPROXIMANT = os.getenv("WAVEFORM_APPROXIMANT", "IMRPhenomD")
FREQUENCY_LOW_CUTOFF = float(os.getenv("FREQUENCY_LOW_CUTOFF", "20"))
FREQUENCY_HIGH_CUTOFF = float(os.getenv("FREQUENCY_HIGH_CUTOFF", "2048"))
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "8.0"))
HIGHPASS_FREQUENCY = float(os.getenv("HIGHPASS_FREQUENCY", "15"))

# Set up logging level from environment
log_level = getattr(logging, os.getenv("MCP_SERVER_LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

if VERBOSE_LOGGING:
    logger.info(f"GW Analysis Server initialized with environment variables:")
    logger.info(f"  Data Configuration:")
    logger.info(f"    Temp directory: {temp_dir}")
    logger.info(f"    JSON file path: {json_file_path}")
    logger.info(f"  GW150914 Event Configuration:")
    logger.info(f"    GPS start time: {GW150914_GPS_START}")
    logger.info(f"    GPS end time: {GW150914_GPS_END}")
    logger.info(f"    Signal GPS time: {SIGNAL_GPS_TIME}")
    logger.info(f"    Default detectors: {DEFAULT_DETECTORS}")
    logger.info(f"  Analysis Configuration:")
    logger.info(f"    Waveform approximant: {WAVEFORM_APPROXIMANT}")
    logger.info(f"    Frequency range: {FREQUENCY_LOW_CUTOFF}-{FREQUENCY_HIGH_CUTOFF} Hz")
    logger.info(f"    Detection threshold: {DETECTION_THRESHOLD}")
    logger.info(f"    High-pass frequency: {HIGHPASS_FREQUENCY} Hz")
    logger.info(f"  Logging Configuration:")
    logger.info(f"    Log level: {log_level}")
    logger.info(f"    Debug mode: {DEBUG_MODE}")
    logger.info(f"    Verbose logging: {VERBOSE_LOGGING}")
    logger.info(f"  Environment file loaded: {'.env' if os.path.exists('.env') else 'env.template or system env'}")

# Create the MCP server object
mcp = FastMCP()

@mcp.tool()
async def fetch_and_condition_gw_data(gps_start: int = None, gps_end: int = None, detectors: list[str] = None) -> TextContent:
    """
    Fetch and preprocess gravitational wave data from GWOSC for multiple detectors.
    
    Downloads gravitational wave strain data for the specified GPS time range and detectors,
    applies high-pass filtering for noise reduction, and computes the power spectral density.
    Uses local cache when available to improve performance. Supports up to three detectors
    including LIGO Hanford (H1), LIGO Livingston (L1), and Virgo (V1).
    
    Args:
        gps_start: GPS start time (default from environment: GW150914_GPS_START)
        gps_end: GPS end time (default from environment: GW150914_GPS_END)
        detectors: List of detector names (default from environment: DEFAULT_DETECTORS)
    
    Returns:
        TextContent: JSON string containing data acquisition and preprocessing information,
                    including data lengths, sample rates, and processing parameters for all detectors
    """
    import json
    import numpy as np
    import os
    import tempfile
    
    # Use environment defaults if not provided
    if gps_start is None:
        gps_start = GW150914_GPS_START
    if gps_end is None:
        gps_end = GW150914_GPS_END
    if detectors is None:
        detectors = DEFAULT_DETECTORS.copy()
    
    if DEBUG_MODE:
        logger.debug(f"fetch_and_condition_gw_data called with: gps_start={gps_start}, gps_end={gps_end}, detectors={detectors}")
    
    try:
        from gwpy.timeseries import TimeSeries
        
        detector_data = {}
        
        for detector in detectors:
            # Check for local cache file
            cache_filename = f"{detector}-{gps_start}-{gps_end}.txt"
            cache_path = os.path.join(temp_dir, cache_filename)
            
            # Try to read from cache first
            data_source = "cache"
            if os.path.exists(cache_path):
                try:
                    # Load data from cache file
                    data = TimeSeries.read(cache_path)
                except Exception as cache_error:
                    # Cache read failed, fetch from network
                    data = TimeSeries.fetch_open_data(detector, gps_start, gps_end)
                    data_source = "network"
            else:
                # No cache available, fetch from network
                data = TimeSeries.fetch_open_data(detector, gps_start, gps_end)
                data_source = "network"
                
                # Try to save to cache (if filesystem allows writing)
                try:
                    data.write(cache_path)
                except:
                    pass  # Ignore write failures
            
            # Apply high-pass filter to remove low-frequency noise (frequency from environment)
            high = data.highpass(HIGHPASS_FREQUENCY)
            
            # Calculate power spectral density (4-second FFT length)
            psd = high.psd(4, 2)
            
            # Crop data to 4-second segment for SNR calculation
            crop_start = gps_start + 14  # 1126259460
            crop_end = crop_start + 4    # 1126259464
            zoom = high.crop(crop_start, crop_end)
            
            detector_data[detector] = {
                'data': data,
                'high': high,
                'psd': psd,
                'zoom': zoom,
                'data_source': data_source,
                'cache_path': cache_path if os.path.exists(cache_path) else None
            }
        
        # Store in global variables for subsequent tools
        globals()['gw_detector_data'] = detector_data
        
        if VERBOSE_LOGGING:
            logger.info(f"Successfully processed {len(detectors)} detectors: {detectors}")
            for detector in detectors:
                det_data = detector_data[detector]
                logger.info(f"  {detector}: {det_data['data_source']} source, {len(det_data['data'])} samples")
        
        result = {
            "detectors": detectors,
            "gps_start": gps_start,
            "gps_end": gps_end,
            "crop_start": crop_start,
            "crop_end": crop_end,
            "highpass_frequency": HIGHPASS_FREQUENCY,
            "detector_info": {}
        }
        
        for detector in detectors:
            det_data = detector_data[detector]
            result["detector_info"][detector] = {
                "data_source": det_data['data_source'],
                "cache_path": det_data['cache_path'],
                "data_length": len(det_data['data']),
                "filtered_length": len(det_data['high']),
                "cropped_length": len(det_data['zoom']),
                "sample_rate": float(det_data['data'].sample_rate.value),
                "duration": float(det_data['data'].duration.value),
                "psd_calculated": True
            }

        return TextContent(type="text", text=json.dumps(result))
        
    except Exception as e:
        return TextContent(type="text", text=json.dumps({
            "error": f"Failed to fetch and condition GW data: {str(e)}"
        }))

@mcp.tool()
async def generate_gw_templates(
    mass1: float, 
    mass2: float, 
    ra: float, 
    dec: float, 
    approximant: str = "IMRPhenomD"
) -> TextContent:
    """
    Generate gravitational wave signal templates for multiple detectors with sky location.
    
    Creates frequency-domain gravitational wave templates using the specified binary
    black hole parameters including sky location (right ascension and declination).
    The templates are generated for each detector using PyCBC waveform models and 
    include proper antenna pattern functions for network analysis. Supports up to
    three detectors including LIGO Hanford (H1), LIGO Livingston (L1), and Virgo (V1).
    
    Args:
        mass1: Mass of the first black hole in solar masses 
        mass2: Mass of the second black hole in solar masses 
        ra: Right ascension in radians 
        dec: Declination in radians
        approximant: Waveform approximant model (default: "IMRPhenomD")
    
    Returns:
        TextContent: JSON string containing template generation information for all detectors,
                    including mass parameters, sky location, frequency range, and antenna patterns
    """
    import json
    
    try:
        from pycbc.waveform import get_fd_waveform
        from pycbc.detector import Detector
        import numpy as np
        
        # Check if preprocessed data is available
        if 'gw_detector_data' not in globals():
            return TextContent(type="text", text=json.dumps({
                "error": "No detector data found. Please run fetch_and_condition_gw_data first."
            }))
        
        detector_data = globals()['gw_detector_data']
        detectors = list(detector_data.keys())
        
        # Get PSD from first detector for frequency grid
        first_detector = detectors[0]
        psd = detector_data[first_detector]['psd']
        
        # Generate frequency-domain template waveform using environment configuration
        hp, hc = get_fd_waveform(
            approximant=approximant,
            mass1=mass1,
            mass2=mass2,
            f_lower=FREQUENCY_LOW_CUTOFF,
            f_final=FREQUENCY_HIGH_CUTOFF,
            delta_f=psd.df.value,
        )
        
        # Get GPS time from stored data
        if detector_data[first_detector]['zoom'] is not None:
            gps_time = float(detector_data[first_detector]['zoom'].times[len(detector_data[first_detector]['zoom'])//2].value)
        else:
            gps_time = SIGNAL_GPS_TIME  # signal time
        
        # Generate templates for each detector
        detector_templates = {}
        
        for detector_name in detectors:
            # Get detector information
            detector = Detector(detector_name)
            
            # Calculate antenna pattern functions
            fp, fc = detector.antenna_pattern(ra, dec, 0, gps_time)  # polarization angle = 0
            
            # Combine polarizations with antenna pattern
            h_detector = fp * hp + fc * hc
            
            detector_templates[detector_name] = {
                'template': h_detector,
                'fp': fp,
                'fc': fc
            }
        
        # Store templates and sky parameters for subsequent use
        globals()['gw_detector_templates'] = detector_templates
        globals()['gw_template_params'] = {
            'mass1': mass1,
            'mass2': mass2,
            'ra': ra,
            'dec': dec,
            'gps_time': gps_time
        }
        
        if VERBOSE_LOGGING:
            logger.info(f"Generated templates for {len(detectors)} detectors with parameters:")
            logger.info(f"  Mass1: {mass1:.1f} M☉, Mass2: {mass2:.1f} M☉")
            logger.info(f"  RA: {np.degrees(ra):.1f}°, Dec: {np.degrees(dec):.1f}°")
            for detector_name in detectors:
                det_template = detector_templates[detector_name]
                logger.info(f"  {detector_name}: fp={det_template['fp']:.3f}, fc={det_template['fc']:.3f}")
        
        result = {
            "mass1": mass1,
            "mass2": mass2,
            "ra": ra,
            "dec": dec,
            "ra_degrees": np.degrees(ra),
            "dec_degrees": np.degrees(dec),
            "approximant": approximant,
            "f_lower": FREQUENCY_LOW_CUTOFF,
            "f_final": FREQUENCY_HIGH_CUTOFF,
            "gps_time_used": gps_time,
            "detectors": detectors,
            "detector_templates": {}
        }
        
        for detector_name in detectors:
            det_template = detector_templates[detector_name]
            result["detector_templates"][detector_name] = {
                "template_length": len(det_template['template']),
                "delta_f": float(det_template['template'].delta_f),
                "antenna_pattern_fp": float(det_template['fp']),
                "antenna_pattern_fc": float(det_template['fc'])
            }
        
        result["templates_generated"] = True
        
        return TextContent(type="text", text=json.dumps(result))
        
    except Exception as e:
        return TextContent(type="text", text=json.dumps({
            "error": f"Failed to generate GW templates: {str(e)}"
        }))

@mcp.tool()
async def calculate_network_snr_matched_filter() -> TextContent:
    """
    Perform network matched filtering to calculate coherent signal-to-noise ratio (SNR).
    
    Executes matched filtering between the generated templates and preprocessed gravitational 
    wave data from multiple detectors to compute the network SNR time series. The network SNR
    is calculated as the quadrature sum of individual detector SNRs, providing enhanced 
    detection sensitivity and sky localization capability. Supports up to three detectors
    including LIGO Hanford (H1), LIGO Livingston (L1), and Virgo (V1).
    
    Returns:
        TextContent: JSON string containing network SNR calculation results and statistics,
                    including maximum network SNR, detection time, individual detector contributions,
                    and significance metrics
    """
    import json
    import numpy as np
    
    try:
        from pycbc.filter import matched_filter
        from gwpy.timeseries import TimeSeries
        
        # Check if required data exists
        if 'gw_detector_data' not in globals():
            return TextContent(type="text", text=json.dumps({
                "error": "No detector data found. Please run fetch_and_condition_gw_data first."
            }))
        
        if 'gw_detector_templates' not in globals():
            return TextContent(type="text", text=json.dumps({
                "error": "No templates found. Please run generate_gw_templates first."
            }))
        
        detector_data = globals()['gw_detector_data']
        detector_templates = globals()['gw_detector_templates']
        template_params = globals().get('gw_template_params', {})
        
        detectors = list(detector_data.keys())
        detector_snrs = {}
        
        # Perform matched filtering for each detector
        for detector_name in detectors:
            zoom = detector_data[detector_name]['zoom']
            psd = detector_data[detector_name]['psd']
            h_detector = detector_templates[detector_name]['template']
            
            # Perform matched filtering using environment configuration
            snr = matched_filter(
                h_detector,
                zoom.to_pycbc(),
                psd=psd.to_pycbc(),
                low_frequency_cutoff=HIGHPASS_FREQUENCY,
            )
            
            # Convert to GWpy time series
            snrts = TimeSeries.from_pycbc(snr)
            detector_snrs[detector_name] = snrts
        
        # Calculate network SNR as quadrature sum
        # Network SNR = sqrt(sum(|SNR_i|^2)) where i runs over detectors
        network_snr_squared = None
        
        for detector_name in detectors:
            snr_abs_squared = np.abs(detector_snrs[detector_name].value) ** 2
            if network_snr_squared is None:
                network_snr_squared = snr_abs_squared
            else:
                network_snr_squared += snr_abs_squared
        
        network_snr = np.sqrt(network_snr_squared)
        
        # Create network SNR time series using the time array from first detector
        first_detector = detectors[0]
        network_snrts = TimeSeries(
            network_snr,
            times=detector_snrs[first_detector].times,
            name='Network SNR'
        )
        
        # Store SNR results
        globals()['gw_detector_snrs'] = detector_snrs
        globals()['gw_network_snr'] = network_snrts
        
        # Calculate statistical information
        # Find maximum network SNR around merger time
        target_gps_time = SIGNAL_GPS_TIME  # signal time merger time
        time_window = 0.02  # +-0.02 sec range
        
        # Find indices within the target time window
        times_array = network_snrts.times.value
        time_mask = (times_array >= target_gps_time - time_window) & (times_array <= target_gps_time + time_window)
        
        if np.any(time_mask):
            # Find maximum within the time window
            windowed_snr = network_snr[time_mask]
            windowed_indices = np.where(time_mask)[0]
            local_max_idx = np.argmax(windowed_snr)
            max_idx = windowed_indices[local_max_idx]
            max_network_snr = float(windowed_snr[local_max_idx])
        else:
            # Fallback to global maximum if target time not found
            max_network_snr = float(np.max(network_snr))
            max_idx = np.argmax(network_snr)
        
        max_time = float(network_snrts.times[max_idx].value)
        mean_network_snr = float(np.mean(network_snr))
        std_network_snr = float(np.std(network_snr))
        
        # Get individual detector SNRs at maximum network SNR time
        detector_snr_at_max = {}
        for detector_name in detectors:
            detector_snr_at_max[detector_name] = {
                "snr_real": float(detector_snrs[detector_name].value[max_idx].real),
                "snr_imag": float(detector_snrs[detector_name].value[max_idx].imag),
                "snr_abs": float(np.abs(detector_snrs[detector_name].value[max_idx])),
                "snr_phase": float(np.angle(detector_snrs[detector_name].value[max_idx]))
            }

        timestamp = datetime.now().strftime("%H%M%S")
        result = {
            "timestamp": timestamp,
            "max_network_snr": max_network_snr,
            "max_snr_time": max_time,
            "max_snr_gps": max_time,
            "mean_network_snr": mean_network_snr,
            "std_network_snr": std_network_snr,
            "network_snr_length": len(network_snrts),
            "time_start": float(network_snrts.times[0].value),
            "time_end": float(network_snrts.times[-1].value),
            "detection_significance": max_network_snr / std_network_snr,
            "detectors": detectors,
            "detector_snrs_at_max": detector_snr_at_max,
            "template_parameters": template_params,
            "network_matched_filter_completed": True
        }
        
        # Save the dict
        line = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        with open(json_file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if VERBOSE_LOGGING:
            logger.info(f"Network SNR calculation completed:")
            logger.info(f"  Max network SNR: {max_network_snr:.3f}")
            logger.info(f"  Detection time: {max_time:.3f}")
            logger.info(f"  Detection significance: {result['detection_significance']:.1f}")
            logger.info(f"  Results saved to: {json_file_path}")

        return TextContent(type="text", text=json.dumps(result))
        
    except Exception as e:
        return TextContent(type="text", text=json.dumps({
            "error": f"Failed to calculate network SNR: {str(e)}"
        }))

@mcp.tool()
async def plot_network_snr_timeseries() -> ImageContent:
    """
    Plot the network signal-to-noise ratio time series from matched filtering.
    
    Creates a visualization of the network SNR time series and individual detector
    SNR contributions showing the gravitational wave detection results. Highlights 
    the maximum network SNR point and includes detection threshold lines for interpretation.
    Supports up to three detectors including LIGO Hanford (H1), LIGO Livingston (L1), 
    and Virgo (V1).
    
    Returns:
        ImageContent: PNG image of the network SNR time series plot with individual
                     detector contributions and annotations showing detection significance
    """
    import io
    import base64
    import matplotlib.pyplot as plt
    import numpy as np
    
    try:
        # Check if SNR data exists
        if 'gw_network_snr' not in globals() or 'gw_detector_snrs' not in globals():
            # Create error image
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No network SNR data found.\nPlease run calculate_network_snr_matched_filter first.', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            network_snrts = globals()['gw_network_snr']
            detector_snrs = globals()['gw_detector_snrs']
            template_params = globals().get('gw_template_params', {})
            
            # Create subplot layout: network SNR + individual detector SNRs
            fig, axes = plt.subplots(len(detector_snrs) + 1, 1, figsize=(12, 3 * (len(detector_snrs) + 1)))
            if len(detector_snrs) == 1:
                axes = [axes]
            
            times = network_snrts.times.value
            
            # Plot network SNR
            ax_network = axes[0]
            network_snr_values = network_snrts.value
            ax_network.plot(times, network_snr_values, 'k-', linewidth=2, label='Network SNR')
            
            # Mark maximum network SNR point
            max_idx = np.argmax(network_snr_values)
            max_time = times[max_idx]
            max_snr = network_snr_values[max_idx]
            ax_network.plot(max_time, max_snr, 'ro', markersize=10, 
                           label=f'Max Network SNR: {max_snr:.1f} at GPS {max_time:.3f}')
            
            # Add detection threshold line from environment
            ax_network.axhline(y=DETECTION_THRESHOLD, color='r', linestyle='--', alpha=0.7, 
                             label=f'Detection Threshold (SNR={DETECTION_THRESHOLD})')
            
            # Set axes for network plot
            ax_network.set_xlim(times[0], times[-1])
            ax_network.set_ylabel('Network SNR')
            ax_network.grid(True, alpha=0.3)
            ax_network.legend()
            
            # Create title with template parameters
            title = 'LIGO-Virgo Network GW event Matched Filter SNR Time Series'
            if template_params:
                title += f'\nTemplate: M1={template_params.get("mass1", "?"):.1f}M☉, M2={template_params.get("mass2", "?"):.1f}M☉'
                title += f', RA={np.degrees(template_params.get("ra", 0)):.1f}°, Dec={np.degrees(template_params.get("dec", 0)):.1f}°'
            ax_network.set_title(title)
            
            # Plot individual detector SNRs
            colors = ['b', 'g', 'orange', 'purple', 'brown', 'pink']
            for i, (detector_name, snrts) in enumerate(detector_snrs.items()):
                ax = axes[i + 1]
                snr_abs = np.abs(snrts.value)
                color = colors[i % len(colors)]
                
                ax.plot(times, snr_abs, color=color, linewidth=1.5, label=f'{detector_name} |SNR|')
                
                # Mark maximum point for this detector
                det_max_idx = np.argmax(snr_abs)
                det_max_time = times[det_max_idx]
                det_max_snr = snr_abs[det_max_idx]
                ax.plot(det_max_time, det_max_snr, 'o', color=color, markersize=8,
                       label=f'Max: {det_max_snr:.1f}')
                
                ax.set_xlim(times[0], times[-1])
                ax.set_ylabel(f'{detector_name} SNR')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add detection threshold line from environment
                ax.axhline(y=DETECTION_THRESHOLD, color='r', linestyle='--', alpha=0.5)
            
            # Set x-axis label only for bottom plot
            axes[-1].set_xlabel('GPS Time (s)')
            
            # Add signal time reference line to all plots
            signal_time = SIGNAL_GPS_TIME  # GPS time
            if times[0] <= signal_time <= times[-1]:
                for ax in axes:
                    ax.axvline(x=signal_time, color='g', linestyle=':', alpha=0.7)
                # Add label only to first plot
                axes[0].axvline(x=signal_time, color='g', linestyle=':', alpha=0.7, 
                               label=f'Ground-truth Signal Time')
                axes[0].legend()
        
        plt.tight_layout()
        
        # Save as PNG and encode as base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        # Save the results to temp directory from environment
        fig_path = os.path.join(temp_dir, f"network_snr_plot.png")
        plt.savefig(fig_path, format="png", dpi=150, bbox_inches="tight")
        if DEBUG_MODE:
            logger.debug(f"SNR plot saved to: {fig_path}") 
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        return ImageContent(data=image_base64, mimeType="image/png", type="image")
        
    except Exception as e:
        # Create error image
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error creating network SNR plot:\n{str(e)}', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        return ImageContent(data=image_base64, mimeType="image/png", type="image")

@mcp.tool()
async def complete_network_gw_matched_filter_search(
    mass1: float, 
    mass2: float,
    ra: float,
    dec: float,
    gps_start: int = None, 
    gps_end: int = None, 
    detectors: list[str] = None
) -> TextContent:
    """
    Execute complete gravitational wave network matched filter search pipeline.
    
    Performs the full gravitational wave signal search workflow for multiple detectors:
    data acquisition, preprocessing, template generation (including sky location), 
    network matched filtering, and coherent SNR calculation. This is an end-to-end 
    pipeline that automates the entire network detection process. Supports up to
    three detectors including LIGO Hanford (H1), LIGO Livingston (L1), and Virgo (V1).
    
    Args:
        mass1: First black hole mass in solar masses
        mass2: Second black hole mass in solar masses 
        ra: Right ascension in radians 
        dec: Declination in radians 
        gps_start: GPS start time (default from environment: GW150914_GPS_START)
        gps_end: GPS end time (default from environment: GW150914_GPS_END)
        detectors: List of detector names (default from environment: DEFAULT_DETECTORS)
    
    Returns:
        TextContent: JSON string containing complete network search results and detection
                    statistics, including network significance assessment and individual
                    detector contributions
    """
    import json
    import numpy as np
    
    # Use environment defaults if not provided
    if gps_start is None:
        gps_start = GW150914_GPS_START
    if gps_end is None:
        gps_end = GW150914_GPS_END
    if detectors is None:
        detectors = DEFAULT_DETECTORS.copy()
    
    if DEBUG_MODE:
        logger.debug(f"complete_network_gw_matched_filter_search called with: mass1={mass1}, mass2={mass2}, ra={ra}, dec={dec}")
        logger.debug(f"GPS range: {gps_start}-{gps_end}, detectors: {detectors}")
    
    try:
        # Step 1: Fetch and preprocess data for all detectors
        data_result = await fetch_and_condition_gw_data(gps_start, gps_end, detectors)
        data_info = json.loads(data_result.text)
        if "error" in data_info:
            return TextContent(type="text", text=json.dumps({
                "error": f"Data conditioning failed: {data_info['error']}"
            }))
        
        # Step 2: Generate templates for all detectors with sky location
        template_result = await generate_gw_templates(mass1, mass2, ra, dec)
        template_info = json.loads(template_result.text)
        if "error" in template_info:
            return TextContent(type="text", text=json.dumps({
                "error": f"Template generation failed: {template_info['error']}"
            }))
        
        # Step 3: Calculate network SNR
        snr_result = await calculate_network_snr_matched_filter()
        snr_info = json.loads(snr_result.text)
        if "error" in snr_info:
            return TextContent(type="text", text=json.dumps({
                "error": f"Network SNR calculation failed: {snr_info['error']}"
            }))
        
        # Build complete results
        timestamp = datetime.now().strftime("%H%M%S")
        complete_result = {
            "timestamp": timestamp,
            "network_search_completed": True,
            "detectors": detectors,
            "gps_range": [gps_start, gps_end],
            "template_parameters": {
                "mass1": mass1,
                "mass2": mass2,
                "ra": ra,
                "dec": dec,
                "ra_degrees": np.degrees(ra),
                "dec_degrees": np.degrees(dec),
                "approximant": WAVEFORM_APPROXIMANT
            },
            "network_detection_results": {
                "max_network_snr": snr_info["max_network_snr"],
                "max_snr_time": snr_info["max_snr_time"],
                "detection_significance": snr_info["detection_significance"],
                "is_significant": snr_info["max_network_snr"] > DETECTION_THRESHOLD,
                "detector_contributions": snr_info["detector_snrs_at_max"]
            },
            "detector_antenna_patterns": {},
            "processing_summary": {
                "detectors_processed": len(detectors),
                "network_snr_length": snr_info["network_snr_length"]
            }
        }
        
        # Add antenna pattern information for each detector
        for detector in detectors:
            if detector in template_info["detector_templates"]:
                complete_result["detector_antenna_patterns"][detector] = {
                    "fp": template_info["detector_templates"][detector]["antenna_pattern_fp"],
                    "fc": template_info["detector_templates"][detector]["antenna_pattern_fc"],
                    "template_length": template_info["detector_templates"][detector]["template_length"]
                }
        
        # Add individual detector processing info
        for detector in detectors:
            if detector in data_info["detector_info"]:
                complete_result["processing_summary"][f"{detector}_data_length"] = data_info["detector_info"][detector]["data_length"]
                complete_result["processing_summary"][f"{detector}_filtered_length"] = data_info["detector_info"][detector]["filtered_length"]
        
        # save the json file 
        line = json.dumps(complete_result, ensure_ascii=False, separators=(",", ":"))
        with open(json_file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        return TextContent(type="text", text=json.dumps(complete_result))

        
    except Exception as e:
        return TextContent(type="text", text=json.dumps({
            "error": f"Complete network matched filter search failed: {str(e)}"
        }))


# This is the main entry point for your server
def main():
    logger.info('Starting GW Analysis MCP Server')
    if VERBOSE_LOGGING:
        logger.info(f'Server configuration loaded from environment:')
        logger.info(f'  Temp directory: {temp_dir}')
        logger.info(f'  JSON file path: {json_file_path}')
        logger.info(f'  Debug mode: {DEBUG_MODE}')
    mcp.run('stdio')

if __name__ == "__main__":
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clear previous results file if it exists
    try:
        os.remove(json_file_path)
        if VERBOSE_LOGGING:
            logger.info(f'Cleared previous results file: {json_file_path}')
    except FileNotFoundError:
        pass
    
    main()