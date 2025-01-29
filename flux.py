import sys
import os
# Add ComfyUI directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "ComfyUI"))

import torch
import logging
import gc

# Set memory efficient settings
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Starting ComfyUI initialization...")
    
    # Aggressive memory cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info(f"Available CUDA devices: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Import ComfyUI components after path setup
    from comfy.text_encoders import flux as flux_encoder
    from comfy.ldm.flux import model as flux_model
    
    logger.info("ComfyUI components imported successfully")

except Exception as e:
    logger.error(f"Error occurred: {str(e)}", exc_info=True)
    raise

finally:
    # Ensure cleanup happens
    gc.collect()
    torch.cuda.empty_cache()
