import sys

from .orient import OrientAgent

def get_agent_cls(name):
    return getattr(sys.modules[__name__], name)
