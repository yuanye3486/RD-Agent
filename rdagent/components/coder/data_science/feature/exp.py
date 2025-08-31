import pickle
import site
import traceback
from pathlib import Path
from typing import Dict, Optional

from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.core.utils import cache_with_pickle


# Because we use isinstance to distinguish between different types of tasks, we need to use sub classes to represent different types of tasks
class FeatureTask(CoSTEERTask):
    pass
