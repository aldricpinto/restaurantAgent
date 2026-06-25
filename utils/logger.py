import os
import sys
import logging

log_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file_path = os.path.join(log_dir, "ophelia.log")
log_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
formatter = logging.Formatter(log_format)

def setup_logger(name: str, stream):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clean up default handlers to prevent duplication if re-imported
    logger.handlers = []
    
    # File handler (debug level)
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Stream handler (info level)
    stream_handler = logging.StreamHandler(stream)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Prevent propagation to the root logger to avoid duplicate console logs
    logger.propagate = False
    
    return logger

agent_logger = setup_logger("ophelia.agent", sys.stdout)
server_logger = setup_logger("ophelia.server", sys.stderr)
