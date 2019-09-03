import logging
from .imageconfig import ImageData, Dimension
from .imageobject import ImageObject
from .scene import Scene

logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler('debug.log')
fh.setFormatter(formatter)

logger = logging.getLogger('xzgen')
logger.addHandler(fh)

