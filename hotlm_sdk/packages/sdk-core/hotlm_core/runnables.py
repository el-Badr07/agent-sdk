import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
)

from hotlm_core.callbacks import CallbackManager  # Import CallbackManager
from hotlm_core.config import (
    RunnableConfig,  # Assuming CallbackManager is handled via config
)
from hotlm_core.utils import generate_uuid  # Import UUID generator
from pydantic import BaseModel, Field

# Define TypeVars for input and output types
Input = TypeVar("Input")
Output = TypeVar("Output")
NextOutput = TypeVar("NextOutput")

# Helper function to get or create CallbackManager
def _get_callback_manager(config: Optional[RunnableConfig]) -> CallbackManager:
    return config.callback_manager if config and config.callback_manager else CallbackManager([])

# Helper function to get or create run_id
def _get_run_id(config: Optional[RunnableConfig]) -> uuid.UUID:
    return config.run_id if config and config.run_id else generate_uuid()

logger = logging.getLogger(__name__)

class BaseRunnable(ABC, Generic[Input, Output]):
    """Base interface for all runnable components in the framework.

    Inspired by Langchain's Runnable interface, providing a standard way
    to invoke, stream, batch, and async operations, with integrated callbacks.
    """

    # Property to easily get the runnable's name (can be overridden)
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def _invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Internal synchronous invoke logic to be implemented by subclasses."""
        pass

    @abstractmethod
    async def _ainvoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Internal asynchronous invoke logic to be implemented by subclasses."""
        pass

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Synchronously invoke the runnable with a single input, handling callbacks."""
        run_id = _get_run_id(config)
        logger.debug(f"Invoke start: {self.name}, run_id={run_id}, input={input}")
        callback_manager = _get_callback_manager(config)
        callback_manager.on_run_start(run_id, self.name, input, config or RunnableConfig())
        try:
            result = self._invoke(input, config)
            logger.debug(f"Invoke end: {self.name}, run_id={run_id}, result={result}")
            callback_manager.on_run_end(run_id, result, config or RunnableConfig())
            return result
        except Exception as e:
            logger.error(f"Invoke error: {self.name}, run_id={run_id}, error={e}")
            callback_manager.on_run_error(run_id, e, config or RunnableConfig())
            raise e

    async def ainvoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        """Asynchronously invoke the runnable with a single input, handling callbacks."""
        run_id = _get_run_id(config)
        callback_manager = _get_callback_manager(config)
        await callback_manager.aon_run_start(run_id, self.name, input, config or RunnableConfig())
        try:
            result = await self._ainvoke(input, config)
            await callback_manager.aon_run_end(run_id, result, config or RunnableConfig())
            return result
        except Exception as e:
            await callback_manager.aon_run_error(run_id, e, config or RunnableConfig())
            raise e

    def stream(self, input: Input, config: Optional[RunnableConfig] = None) -> Iterator[Output]:
        """Synchronously stream the output, handling callbacks (basic implementation)."""
        run_id = _get_run_id(config)
        logger.debug(f"Stream start: {self.name}, run_id={run_id}, input={input}")
        callback_manager = _get_callback_manager(config)
        callback_manager.on_run_start(run_id, self.name, input, config or RunnableConfig())
        try:
            # Default: yield the single result from invoke
            # Subclasses need to override for true streaming and chunk callbacks
            result = self._invoke(input, config)
            logger.debug(f"Stream chunk: {self.name}, run_id={run_id}, chunk={result}")
            yield result
            callback_manager.on_run_end(run_id, result, config or RunnableConfig())
            logger.debug(f"Stream end: {self.name}, run_id={run_id}")
        except Exception as e:
            logger.error(f"Stream error: {self.name}, run_id={run_id}, error={e}")
            callback_manager.on_run_error(run_id, e, config or RunnableConfig())
            raise e

    async def astream(self, input: Input, config: Optional[RunnableConfig] = None) -> AsyncIterator[Output]:
        """Asynchronously stream the output, handling callbacks (basic implementation)."""
        run_id = _get_run_id(config)
        callback_manager = _get_callback_manager(config)
        await callback_manager.aon_run_start(run_id, self.name, input, config or RunnableConfig())
        try:
            # Default: yield the single result from ainvoke
            # Subclasses need to override for true streaming and chunk callbacks
            result = await self._ainvoke(input, config)
            yield result
            await callback_manager.aon_run_end(run_id, result, config or RunnableConfig())
        except Exception as e:
            await callback_manager.aon_run_error(run_id, e, config or RunnableConfig())
            raise e

    def batch(self, inputs: List[Input], config: Optional[RunnableConfig] = None) -> List[Output]:
        """Synchronously invoke with a batch, handling callbacks for each item."""
        # Simple iteration, subclasses should override for optimization
        configs = self._get_batch_configs(config, len(inputs))
        return [self.invoke(input_item, item_config) for input_item, item_config in zip(inputs, configs)]

    async def abatch(self, inputs: List[Input], config: Optional[RunnableConfig] = None) -> List[Output]:
        """Asynchronously invoke with a batch, handling callbacks for each item."""
        # Uses asyncio.gather for concurrency
        configs = self._get_batch_configs(config, len(inputs))
        tasks: List[Coroutine[Any, Any, Output]] = [
            self.ainvoke(input_item, item_config) for input_item, item_config in zip(inputs, configs)
        ]
        return await asyncio.gather(*tasks)

    def _get_batch_configs(self, config: Optional[RunnableConfig], num_tasks: int) -> List[RunnableConfig]:
        """Helper to create individual configs for batch items, propagating shared settings."""
        base_config = config or RunnableConfig()
        configs = []
        for i in range(num_tasks):
            # Create a new config for each item, inheriting shared properties
            # but generating a new run_id if one wasn't provided for the batch
            item_run_id = base_config.run_id if base_config.run_id else generate_uuid()
            item_config = base_config.copy(update={"run_id": item_run_id})
            configs.append(item_config)
        return configs

    # --- Schema Introspection --- (Optional but useful)
    def get_input_schema(self) -> Optional[Type[BaseModel]]:
        """Returns the Pydantic model for the input schema, if available."""
        # Default implementation tries to infer from Generic type args
        # More robust implementations might be needed in subclasses
        try:
            return self.__orig_bases__[0].__args__[0]
        except (AttributeError, IndexError):
            return None

    def get_output_schema(self) -> Optional[Type[BaseModel]]:
        """Returns the Pydantic model for the output schema, if available."""
        try:
            return self.__orig_bases__[0].__args__[1]
        except (AttributeError, IndexError):
            return None

    # --- Composition --- 
    def pipe(
        self, 
        other: "BaseRunnable[Output, NextOutput] | Callable[[Output], NextOutput]"
    ) -> "RunnableSequence[Input, NextOutput]": # Return type hint fixed
        """Compose this runnable with another runnable or function.

        Equivalent to `RunnableSequence(self, other)`.
        Allows chaining using the `|` operator.
        """
        if not isinstance(other, BaseRunnable) and callable(other):
            other = RunnableLambda(other)
        elif not isinstance(other, BaseRunnable):
             raise TypeError(
                f"Expected a Runnable or callable function, got {type(other)}"
            )

        # If self is already a sequence, extend it
        if isinstance(self, RunnableSequence):
            # Extend existing sequence by appending the new runnable
            return RunnableSequence(self.first, *self.middle, self.last, other)
        else:
            # Create a new sequence with two steps
            return RunnableSequence(self, other)

    def __or__(
        self, 
        other: "BaseRunnable[Output, NextOutput] | Callable[[Output], NextOutput]"
    ) -> "RunnableSequence[Input, NextOutput]": # Return type hint fixed
        """Syntactic sugar for pipe method using the | operator."""
        return self.pipe(other)

class RunnableSequence(BaseRunnable[Input, NextOutput]):
    """A sequence of runnables, where the output of one is the input to the next."""
    first: BaseRunnable[Input, Any]
    middle: List[BaseRunnable[Any, Any]]
    last: BaseRunnable[Any, NextOutput]

    def __init__(self, first: BaseRunnable, *steps: BaseRunnable):
        if not steps:
            raise ValueError("RunnableSequence requires at least two steps.")
        self.first = first
        self.middle = list(steps[:-1])
        self.last = steps[-1]

    @property
    def steps(self) -> List[BaseRunnable]:
        return [self.first] + self.middle + [self.last]

    def _invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> NextOutput:
        res = self.first.invoke(input, config) # Callbacks handled by individual steps
        for runnable in self.middle:
            res = runnable.invoke(res, config)
        return self.last.invoke(res, config)

    async def _ainvoke(self, input: Input, config: Optional[RunnableConfig] = None) -> NextOutput:
        res = await self.first.ainvoke(input, config) # Callbacks handled by individual steps
        for runnable in self.middle:
            res = await runnable.ainvoke(res, config)
        return await self.last.ainvoke(res, config)

    def stream(self, input: Input, config: Optional[RunnableConfig] = None) -> Iterator[NextOutput]:
        """Stream through the sequence: run each step then stream the final Runnable."""
        # Execute all steps except last to get the input for streaming
        res = self.first.invoke(input, config)
        for runnable in self.middle:
            res = runnable.invoke(res, config)
        # Stream on the last step
        yield from self.last.stream(res, config)

    async def astream(self, input: Input, config: Optional[RunnableConfig] = None) -> AsyncIterator[NextOutput]:
        """Async stream through the sequence: run each step then stream the final Runnable."""
        # Execute all steps except last to get the input for streaming
        res = await self.first.ainvoke(input, config)
        for runnable in self.middle:
            res = await runnable.ainvoke(res, config)
        # Async stream on the last step
        async for chunk in self.last.astream(res, config):
            yield chunk

    # Batch for sequence can be complex due to intermediate steps.
    # Default batch will invoke the whole sequence for each input.

# Simple Runnable for wrapping functions
class RunnableLambda(BaseRunnable[Input, Output]):
    """Wrap a callable (function, lambda) as a Runnable."""
    func: Callable[[Input], Output | Coroutine[Any, Any, Output]]

    def __init__(self, func: Callable[[Input], Output | Coroutine[Any, Any, Output]]):
        self.func = func
        # Simple name based on function
        try:
            self._name = func.__name__
        except AttributeError:
            self._name = "RunnableLambda"

    @property
    def name(self) -> str:
        return self._name

    def _invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        result = self.func(input)
        if asyncio.iscoroutine(result):
            # If the function is async but called synchronously, run it.
            # This might be undesirable in some contexts, consider raising an error?
            return asyncio.run(result)
        return result # type: ignore

    async def _ainvoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        result = self.func(input)
        if asyncio.iscoroutine(result):
            return await result
        else:
            # If the function is sync but called asynchronously, just return it.
            return result # type: ignore

    # Lambda streaming/batching defaults to single invoke/ainvoke
