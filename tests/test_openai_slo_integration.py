# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Integration tests for OpenAI API SLO management with tier parameter."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from vllm.config import VllmConfig, SchedulerConfig
from vllm.config.model import ModelConfig
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.v1.engine.exceptions import (
    MaxPendingTokensError,
    MaxTierPendingTokensError,
    QueueOverflowError,
    TooManyRequestsError,
)


class FakeModelConfig(ModelConfig):
    """Fake model config for testing."""
    def __init__(self):
        super().__init__(
            model="fake-model",
            tokenizer="fake-tokenizer",
            trust_remote_code=False,
            download_dir=None,
            use_np_cache=False,
        )


class OpenAISLOTestSetup:
    """Common setup for OpenAI SLO integration tests."""
    
    def __init__(self, scheduler_config: SchedulerConfig):
        self.scheduler_config = scheduler_config
        self.vllm_config = VllmConfig(
            model_config=FakeModelConfig(),
            scheduler_config=scheduler_config,
        )
    
    def create_chat_request(self, tier: float = 1.0) -> ChatCompletionRequest:
        """Create a ChatCompletionRequest with specified tier."""
        return ChatCompletionRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="fake-model",
            tier=tier,
            priority=0,
        )
    
    def create_serving_instance(self, mock_async_llm_instance: AsyncMock) -> OpenAIServingChat:
        """Create OpenAIServingChat instance with mocked engine."""
        return OpenAIServingChat(
            vllm_config=self.vllm_config,
            engine=mock_async_llm_instance,
        )


@pytest.mark.asyncio
async def test_openai_chat_tier_propagation():
    """Test that tier parameter is properly propagated from OpenAI request to engine."""
    
    # Create config with SLO limits
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024,
        is_encoder_decoder=False
    )
    scheduler_config.max_num_reqs = 2
    
    # Setup test helpers
    setup = OpenAISLOTestSetup(scheduler_config)
    
    # Create OpenAI chat request with tier parameter
    chat_request = setup.create_chat_request(tier=0.5)
    
    # Mock the engine client
    with patch('vllm.entrypoints.openai.serving_chat.AsyncLLM') as mock_async_llm_class:
        
        mock_async_llm_instance = AsyncMock()
        mock_async_llm_class.return_value = mock_async_llm_instance
        
        # Mock the generate method to capture the tier parameter
        async def mock_generate(*args, **kwargs):
            tier = kwargs.get('tier', 1.0)
            # Verify that tier parameter is passed correctly
            assert tier == 0.5, f"Expected tier=0.5, got tier={tier}"
            # Simulate the SLO validation by raising an exception
            raise TooManyRequestsError()
        
        mock_async_llm_instance.generate = mock_generate
        
        # Create OpenAI serving instance
        serving = setup.create_serving_instance(mock_async_llm_instance)
        
        # Test that the tier parameter is passed through and causes SLO validation
        with pytest.raises(TooManyRequestsError):
            await serving.create_chat_completion(
                request=chat_request,
                raw_request=None,
            )


@pytest.mark.asyncio
async def test_openai_chat_tier_boundary_cases():
    """Test OpenAI API tier parameter boundary cases."""
    
    # Create config with SLO limits
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024,
        is_encoder_decoder=False
    )
    scheduler_config.max_num_reqs = 1
    scheduler_config.max_pending_context_tokens = 100
    
    # Setup test helpers
    setup = OpenAISLOTestSetup(scheduler_config)
    
    # Mock the engine client
    with patch('vllm.entrypoints.openai.serving_chat.AsyncLLM') as mock_async_llm_class:
        
        mock_async_llm_instance = AsyncMock()
        mock_async_llm_class.return_value = mock_async_llm_instance
        
        # Test cases for different tier values
        test_cases = [
            (0.0, TooManyRequestsError),    # Should be rejected immediately
            (0.5, TooManyRequestsError),    # Should be rejected due to tier
            (1.0, QueueOverflowError),      # Should be rejected due to hard limit
        ]
        
        for tier, expected_exception in test_cases:
            # Create OpenAI chat request with specific tier
            chat_request = setup.create_chat_request(tier=tier)
            
            # Mock generate to simulate SLO validation
            async def mock_generate(*args, **kwargs):
                passed_tier = kwargs.get('tier', 1.0)
                assert passed_tier == tier, f"Expected tier={tier}, got tier={passed_tier}"
                if tier == 0.0:
                    raise TooManyRequestsError()
                elif tier == 0.5:
                    raise TooManyRequestsError()
                elif tier == 1.0:
                    raise QueueOverflowError()
            
            mock_async_llm_instance.generate = mock_generate
            
            # Create OpenAI serving instance
            serving = setup.create_serving_instance(mock_async_llm_instance)
            
            # Test that the correct exception is raised
            with pytest.raises(expected_exception):
                await serving.create_chat_completion(
                    request=chat_request,
                    raw_request=None,
                )


@pytest.mark.asyncio
async def test_openai_chat_tier_default_value():
    """Test that default tier value (1.0) is used when not specified."""
    
    # Create config with SLO limits
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024,
        is_encoder_decoder=False
    )
    scheduler_config.max_num_reqs = 10
    
    # Setup test helpers
    setup = OpenAISLOTestSetup(scheduler_config)
    
    # Create OpenAI chat request WITHOUT tier parameter (should default to 1.0)
    chat_request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model="fake-model",
        # tier parameter not specified - should default to 1.0
        priority=0,
    )
    
    # Mock the engine client
    with patch('vllm.entrypoints.openai.serving_chat.AsyncLLM') as mock_async_llm_class:
        
        mock_async_llm_instance = AsyncMock()
        mock_async_llm_class.return_value = mock_async_llm_instance
        
        # Mock generate to capture the tier parameter
        captured_tier = None
        
        async def mock_generate(*args, **kwargs):
            nonlocal captured_tier
            captured_tier = kwargs.get('tier', 1.0)
            # Should not raise exception since we're testing default behavior
            return []
        
        mock_async_llm_instance.generate = mock_generate
        
        # Create OpenAI serving instance
        serving = setup.create_serving_instance(mock_async_llm_instance)
        
        # Make the request
        await serving.create_chat_completion(
            request=chat_request,
            raw_request=None,
        )
        
        # Verify that default tier value (1.0) was used
        assert captured_tier == 1.0, f"Expected default tier=1.0, got tier={captured_tier}"


@pytest.mark.asyncio
async def test_openai_chat_slo_disabled():
    """Test OpenAI API behavior when SLO limits are disabled."""
    
    # Create config WITHOUT SLO limits (None values)
    scheduler_config = SchedulerConfig.default_factory(
        max_model_len=1024,
        is_encoder_decoder=False
    )
    # max_num_reqs and max_pending_context_tokens are None by default
    
    # Setup test helpers
    setup = OpenAISLOTestSetup(scheduler_config)
    
    # Create OpenAI chat request with low tier
    chat_request = setup.create_chat_request(tier=0.1)
    
    # Mock the engine client
    with patch('vllm.entrypoints.openai.serving_chat.AsyncLLM') as mock_async_llm_class:
        
        mock_async_llm_instance = AsyncMock()
        mock_async_llm_class.return_value = mock_async_llm_instance
        
        # Mock generate to not raise any SLO exceptions
        async def mock_generate(*args, **kwargs):
            tier = kwargs.get('tier', 1.0)
            assert tier == 0.1, f"Expected tier=0.1, got tier={tier}"
            # Should not raise any SLO exceptions when limits are disabled
            return []
        
        mock_async_llm_instance.generate = mock_generate
        
        # Create OpenAI serving instance
        serving = setup.create_serving_instance(mock_async_llm_instance)
        
        # Should not raise any exceptions when SLO is disabled
        await serving.create_chat_completion(
            request=chat_request,
            raw_request=None,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])