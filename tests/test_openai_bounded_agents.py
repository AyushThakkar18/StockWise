from pathlib import Path

import pytest

from portfoliopilot.openai_bounded_agents import OpenAIBoundedAgent


def test_only_gpt_4o_mini_is_allowed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only permits gpt-4o-mini"):
        OpenAIBoundedAgent("key", tmp_path, model="gpt-4o")
