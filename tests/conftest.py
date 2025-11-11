import sys
import types


def _install_litellm_stub():
    """Provide a lightweight stand-in so importing hodor.agent doesn't need native deps."""
    if "litellm" in sys.modules:
        return

    stub = types.ModuleType("litellm")

    def completion(*args, **kwargs):
        raise RuntimeError("litellm completion stub should not be invoked in unit tests")

    stub.completion = completion

    def supports_reasoning(model_name: str) -> bool:
        return False

    stub.supports_reasoning = supports_reasoning
    stub.drop_params = False

    sys.modules["litellm"] = stub
    # Support "from litellm import completion"
    sys.modules["litellm"].completion = completion


_install_litellm_stub()
