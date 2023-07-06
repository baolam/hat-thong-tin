import sys
sys.path.append("./")

from src.utils import Logger
log_man = Logger("logs")

log_man.log("Hello mọi người", show = True)
log_man.log("Hello bạn", show = True)