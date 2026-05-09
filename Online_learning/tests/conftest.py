"""pytest path setup — runs automatically before any test collection."""
import sys, os

_tests = os.path.dirname(os.path.abspath(__file__))             # Online_learning/tests/
_ol    = os.path.dirname(_tests)                                 # Online_learning/
_ma    = os.path.join(_ol, '..', 'mapping-algo')
_dream = os.path.join(_ol, '..', 'LLM_Dreaming')
for _p in [_tests, _ol, _ma, _dream]:
    if _p not in sys.path:
        sys.path.append(_p)
