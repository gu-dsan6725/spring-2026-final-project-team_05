"""
Shared observability setup for full contract analysis pipeline
Initializes braintrust logger used by all agent nodes for tracing

Usage in any agent:
    from observability import get_logger
    logger = get_logger()
    if logger:
        with logger.start_span("my_node") as span:
            result = do_work()
            span.log(input={...}, output={...}, metadata={...})

"""

import os
import braintrust  
from dotenv import load_dotenv

load_dotenv()

BRAINTRUST_PROJECT = "contract-pipeline-evals"  

_logger = None 

def get_logger():
    """Return initialized Braintrust logger (None if BRAINTRUST_API_KEY is missing)"""
    global _logger
    if _logger is None:
        api_key = os.environ.get("BRAINTRUST_API_KEY")
        if api_key:
            _logger = braintrust.init_logger(  
                project=BRAINTRUST_PROJECT,
                api_key=api_key,
            )
        else:
            print(
                "[observability] BRAINTRUST_API_KEY not set — "
                "tracing disabled. Add it to .env to enable spans."
            )
    return _logger
