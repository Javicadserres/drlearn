from . import drlearn
from . import utils

from .drlearn import DRLearn
from .utils import Memory
# from ._version import get_versions
# __version__ = get_versions()['version']
# del get_versions
from . import _version
__version__ = _version.get_versions()['version']
