import asyncio
import inspect
import logging
from typing import Annotated, Any, Callable, Dict, Optional, Type, Union

from livekit.agents.llm import LLM, ChatContext, FunctionContext
from livekit.agents.llm.function_context import METADATA_ATTR, TypeInfo, ai_callable
from livekit.agents.stt import STT
from livekit.agents.pipeline.speech_handle import SpeechHandle

logger = logging.getLogger(__name__)


class ResultNotSet(Exception):
    """Exception raised when the task result is not set."""


class TaskFailed(Exception):
    """Exception raised when the task fails."""


class SilentSentinel:
    """Sentinel value to indicate the function call shouldn't create a response."""

    def __init__(self, result: Any = None, error: Optional[BaseException] = None):
        self._result = result
        self._error = error

    def __repr__(self) -> str:
        return f"SilentSentinel(result={self._result}, error={self._error})"


class AgentTask:
    # Single class-level storage for all tasks
    _registered_tasks: Dict[Union[str, Type["AgentTask"]], "AgentTask"] = {}

    def __init__(
        self,
        instructions: Optional[str] = None,
        functions: Optional[list[Callable]] = None,
        llm: Optional[LLM] = None,
        name: Optional[str] = None,
    ) -> None:
        self._chat_ctx = ChatContext()
        self._instructions = instructions
        if instructions:
            self._chat_ctx.append(text=instructions, role="system")

        self._fnc_ctx = FunctionContext()
        functions = functions or []
        # register ai functions from the list
        for fnc in functions:
            if not hasattr(fnc, METADATA_ATTR):
                fnc = ai_callable()(fnc)
            self._fnc_ctx._register_ai_function(fnc)

        # register ai functions from the class
        for _, member in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(member, METADATA_ATTR) and member not in functions:
                self._fnc_ctx._register_ai_function(member)

        self._llm = llm
        self._stt = None

        # Auto-register if name is provided
        if name is not None:
            self.register_task(self, name)
        self._name = name

    @classmethod
    def register_task(
        cls, task: "AgentTask", name: Optional[str] = None
    ) -> "AgentTask":
        """Register a task instance globally"""
        # Register by name if provided
        if name is not None:
            if name in cls._registered_tasks:
                raise ValueError(f"Task with name '{name}' already registered")
            cls._registered_tasks[name] = task
        else:
            # register by type
            task_type = type(task)
            if task_type in cls._registered_tasks:
                raise ValueError(
                    f"Task of type {task_type.__name__} already registered"
                )
            cls._registered_tasks[task_type] = task

        return task

    def inject_chat_ctx(self, chat_ctx: ChatContext) -> None:
        # filter duplicate messages
        existing_messages = {msg.id: msg for msg in self._chat_ctx.messages}
        for msg in chat_ctx.messages:
            if msg.id not in existing_messages:
                self._chat_ctx.messages.append(msg)

    @property
    def instructions(self) -> Optional[str]:
        return self._instructions

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @chat_ctx.setter
    def chat_ctx(self, chat_ctx: ChatContext) -> None:
        self._chat_ctx = chat_ctx

    @property
    def fnc_ctx(self) -> FunctionContext:
        return self._fnc_ctx

    @property
    def llm(self) -> Optional[LLM]:
        return self._llm

    @property
    def stt(self) -> Optional[STT]:
        return self._stt

    @classmethod
    def get_task(cls, key: Union[str, Type["AgentTask"]]) -> "AgentTask":
        """Get task instance by name or class"""
        if key not in cls._registered_tasks:
            raise ValueError(f"Task with name or class {key} not found")
        return cls._registered_tasks[key]

    @classmethod
    def all_registered_tasks(cls) -> list["AgentTask"]:
        """Get all registered tasks"""
        return list(set(cls._registered_tasks.values()))

    def __repr__(self) -> str:
        if self._name:
            return f"{self.__class__.__name__}(name={self._name})"
        return f"{self.__class__.__name__}()"


class AgentInlineTask(AgentTask):
    def __init__(
        self,
        instructions: Optional[str] = None,
        functions: Optional[list[Callable]] = None,
        llm: Optional[LLM] = None,
        name: Optional[str] = None,
        preset_result: Optional[Any] = None,
    ) -> None:
        super().__init__(instructions, functions, llm, name)

        self._done_fut: asyncio.Future[None] = asyncio.Future()
        self._result: Optional[Any] = preset_result

        self._parent_task: Optional[AgentTask] = None
        self._parent_speech: Optional[SpeechHandle] = None

    async def run(self, proactive_reply: bool = True) -> Any:
        from ..pipeline.pipeline_agent import AgentCallContext

        call_ctx = AgentCallContext.get_current()
        agent = call_ctx.agent

        self._parent_task = agent.current_agent_task
        self._parent_speech = call_ctx.speech_handle
        agent.update_task(self)
        logger.debug(
            "running inline task",
            extra={"task": str(self), "parent_task": str(self._parent_task)},
        )
        try:
            # generate reply to the user
            if proactive_reply:
                speech_handle = SpeechHandle.create_assistant_speech(
                    allow_interruptions=agent._opts.allow_interruptions,
                    add_to_chat_ctx=True,
                )
                self._proactive_reply_task = asyncio.create_task(
                    agent._synthesize_answer_task(None, speech_handle)
                )
                if self._parent_speech is not None:
                    self._parent_speech.add_nested_speech(speech_handle)
                else:
                    agent._add_speech_for_playout(speech_handle)

            # wait for the task to complete
            await self._done_fut
            if self.exception:
                raise self.exception

            if self._result is None:
                raise ResultNotSet()
            return self._result
        finally:
            # reset the parent task
            agent.update_task(self._parent_task)
            logger.debug(
                "inline task completed",
                extra={
                    "result": self._result,
                    "error": self.exception,
                    "task": str(self),
                    "parent_task": str(self._parent_task),
                },
            )

    @ai_callable()
    def on_success(self) -> SilentSentinel:
        """Called when user confirms the information is correct.
        This function is called to indicate the job is done.
        """
        if not self._done_fut.done():
            self._done_fut.set_result(None)
        return SilentSentinel(result=self.result, error=self.exception)

    @ai_callable()
    def on_error(
        self,
        reason: Annotated[str, TypeInfo(description="The reason for the error")],
    ) -> SilentSentinel:
        """Called when user wants to stop or refuses to provide the information.
        Only focus on your job.
        """
        if not self._done_fut.done():
            self._done_fut.set_exception(TaskFailed(reason))
        return SilentSentinel(result=self.result, error=self.exception)

    @property
    def done(self) -> bool:
        return self._done_fut.done()

    @property
    def result(self) -> Any:
        return self._result

    @property
    def exception(self) -> Optional[BaseException]:
        return self._done_fut.exception()

    def __repr__(self) -> str:
        speech_id = self._parent_speech.id if self._parent_speech else None
        return (
            f"{self.__class__.__name__}(parent_speech={speech_id}, name={self._name})"
        )