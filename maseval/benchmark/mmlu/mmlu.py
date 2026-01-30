"""MMLU Benchmark - Multiple Choice Question Answering Evaluation.

Implements MMLU evaluation compatible with lm-evaluation-harness output format,
with anchor point-based task selection for DISCO prediction.

Reference: Based on disco-public evaluation methodology.
Dataset: MMLU (Massive Multitask Language Understanding)

Usage:
    from maseval.benchmark.mmlu import (
        MMLUBenchmark, load_tasks, AnchorPointsTaskQueue
    )

    # Load tasks filtered to anchor points
    tasks = load_tasks(
        data_path="/path/to/mmlu_prompts_examples.json",
        anchor_points_path="/path/to/anchor_points.pkl",
    )

    # Create benchmark with HuggingFace model
    class MyMMLUBenchmark(MMLUBenchmark):
        def get_model_adapter(self, model_id, **kwargs):
            from transformers import pipeline
            from maseval.interface.inference import HuggingFaceModelAdapter
            pipe = pipeline("text-generation", model=model_id)
            return HuggingFaceModelAdapter(model=pipe, model_id=model_id)

    benchmark = MyMMLUBenchmark()
    results = benchmark.run(tasks=tasks, agent_data={"model_id": "meta-llama/Llama-2-7b"})
"""

import json
import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

# numpy is optional - only needed for anchor points processing
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

from maseval import (
    AgentAdapter,
    Benchmark,
    Environment,
    Evaluator,
    ModelAdapter,
    Task,
)
from maseval.core.task import AdaptiveTaskQueue, SequentialTaskQueue
from maseval.core.tracing import TraceableMixin
from maseval.core.config import ConfigurableMixin


# =============================================================================
# Task Queue
# =============================================================================


class AnchorPointsTaskQueue(AdaptiveTaskQueue):
    """Task queue that iterates through tasks in anchor points order.

    This queue is used for DISCO-based evaluation where we only evaluate
    on a subset of anchor tasks and predict performance on the full dataset.

    The queue iterates through tasks in the order specified by anchor_points,
    and stops when all anchor tasks have been processed.
    """

    def __init__(self, tasks: List[Task], anchor_points: Optional[List[int]] = None):
        """Initialize anchor points task queue.

        Args:
            tasks: Full list of tasks (ordered by doc_id).
            anchor_points: Optional list of task indices (doc_ids) to evaluate.
                If None, evaluates all tasks in order.
        """
        # If anchor_points provided, filter tasks to only include anchor tasks
        # This dramatically improves performance by avoiding O(n²) iteration
        if anchor_points is not None:
            # Build index mapping for quick lookup
            task_by_doc_id: Dict[int, Task] = {}
            for i, task in enumerate(tasks):
                doc_id = task.metadata.get("doc_id", i)
                task_by_doc_id[doc_id] = task

            # Filter to only anchor tasks, preserving anchor order
            anchor_tasks = []
            for doc_id in anchor_points:
                task = task_by_doc_id.get(doc_id)
                if task is not None:
                    anchor_tasks.append(task)

            # Store original for reference
            self._all_tasks = tasks
            self._task_by_doc_id = task_by_doc_id
            tasks = anchor_tasks

        super().__init__(tasks)
        self._anchor_points = anchor_points
        self._anchor_idx = 0

        # Initialize state immediately (since __iter__ is overridden and skips initial_state())
        self._state = self.initial_state()

    def __iter__(self) -> Iterator[Task]:
        """Yield tasks in anchor point order.

        Since tasks are pre-filtered during __init__, we simply iterate
        over the stored tasks in order. This avoids the infinite loop
        issue in AdaptiveTaskQueue.__iter__ which relies on on_task_repeat_end
        to remove tasks from _remaining.
        """
        return iter(self._tasks)

    def initial_state(self) -> Dict[str, Any]:
        """Initialize state for anchor point iteration."""
        return {
            "anchor_idx": 0,
            "completed_anchors": [],
        }

    def select_next_task(self, remaining: Sequence[Task], state: Dict[str, Any]) -> Optional[Task]:
        """Select the next anchor task to execute.

        Args:
            remaining: Tasks not yet executed.
            state: Current state with anchor_idx.

        Returns:
            Next anchor task, or None if all anchors processed.
        """
        # Simply return the first remaining task since we pre-filtered to anchor tasks only
        return remaining[0] if remaining else None

    def update_state(self, task: Task, report: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Update state after task completion.

        Args:
            task: Completed task.
            report: Execution report.
            state: Current state.

        Returns:
            Updated state.
        """
        doc_id = task.metadata.get("doc_id")
        state["completed_anchors"].append(doc_id)
        state["anchor_idx"] += 1

        return state


# =============================================================================
# Environment
# =============================================================================


class MMLUEnvironment(Environment):
    """Simple environment for MMLU multiple choice evaluation.

    MMLU tasks don't require tools - the environment just holds
    the task context (question, choices, etc.).
    """

    def setup_state(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize state from task data."""
        return {
            "query": task_data.get("query", ""),
            "choices": task_data.get("environment_data", {}).get("choices", []),
            "full_prompt": task_data.get("environment_data", {}).get("full_prompt", ""),
            "use_full_prompt": task_data.get("environment_data", {}).get("use_full_prompt", False),
        }

    def create_tools(self) -> Dict[str, Any]:
        """MMLU doesn't use tools."""
        return {}

    def get_prompt(self) -> str:
        """Get the prompt to send to the model.

        Returns full_prompt if use_full_prompt is True, otherwise query.
        """
        if self.state.get("use_full_prompt", False):
            return self.state.get("full_prompt", self.state.get("query", ""))
        return self.state.get("query", "")


# =============================================================================
# Evaluator
# =============================================================================


class MMLUEvaluator(Evaluator):
    """Evaluator for MMLU multiple choice questions.

    Computes accuracy metrics (acc and acc_norm) by comparing model predictions
    with gold answers.
    """

    def __init__(
        self,
        task: Task,
        environment: Environment,
        user: Optional[Any] = None,
    ):
        """Initialize MMLU evaluator.

        Args:
            task: Task being evaluated (contains gold answer).
            environment: Environment (provides choices).
            user: Unused for MMLU.
        """
        self.task = task
        self.environment = environment
        self.gold = task.evaluation_data.get("gold", 0)
        self.choices = task.environment_data.get("choices", ["A", "B", "C", "D"])

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant traces for evaluation.

        For MMLU, we need the model's response from agent traces.
        """
        # Get agent traces
        agents = traces.get("agents", {})
        if agents:
            # Get first agent's messages
            first_agent = next(iter(agents.values()), {})
            messages = first_agent.get("messages", [])
            return {"messages": messages}
        return {"messages": []}

    def __call__(self, traces: Dict[str, Any], final_answer: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the model's response.

        Args:
            traces: Filtered traces with messages.
            final_answer: The model's final answer.

        Returns:
            Dict with acc, acc_norm, predicted, gold, correct, and optionally logprobs fields.
        """
        # Parse the model's answer
        predicted = self._parse_answer(final_answer or "")

        # Check if correct
        correct = predicted == self.gold

        result = {
            "acc": 1.0 if correct else 0.0,
            "acc_norm": 1.0 if correct else 0.0,
            "predicted": predicted,
            "gold": self.gold,
            "correct": correct,
            "doc_id": self.task.metadata.get("doc_id"),
        }

        # Extract logprobs from traces if available (for logprobs-based evaluation)
        messages = traces.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict) and "logprobs" in msg:
                result["logprobs"] = msg["logprobs"]
                break

        return result

    def _parse_answer(self, response: str) -> int:
        """Parse model response to extract answer choice.

        Handles various formats:
        - Direct letter: "A", "B", "C", "D"
        - With period: "A."
        - As part of sentence: "The answer is A"

        Args:
            response: Model's response string.

        Returns:
            Index of the predicted choice (0-3), or -1 if unparseable.
        """
        if not response:
            return -1

        response = response.strip().upper()

        # Direct letter match
        for i, choice in enumerate(["A", "B", "C", "D"]):
            if response == choice or response.startswith(f"{choice}."):
                return i

        # Look for "answer is X" pattern
        for i, choice in enumerate(["A", "B", "C", "D"]):
            if f"ANSWER IS {choice}" in response:
                return i
            if f"ANSWER: {choice}" in response:
                return i

        # Last character check
        last_char = response.rstrip(".")[-1] if response else ""
        for i, choice in enumerate(["A", "B", "C", "D"]):
            if last_char == choice:
                return i

        return -1


# =============================================================================
# Model Adapter Wrapper for MCQ
# =============================================================================


class MMLUModelAgent(TraceableMixin, ConfigurableMixin):
    """Simple agent wrapper that passes prompts to a model for MCQ evaluation.

    This is a minimal agent that just forwards prompts to the model
    and returns the response. It supports tracing for MASEval integration.
    """

    def __init__(self, model: ModelAdapter, name: str = "mmlu_agent"):
        """Initialize MMLU model agent.

        Args:
            model: ModelAdapter to use for generation.
            name: Agent name for tracing.
        """
        super().__init__()
        self.model = model
        self.name = name
        self._messages: List[Dict[str, Any]] = []

    def run(self, prompt: str) -> str:
        """Run the model on a prompt.

        Args:
            prompt: The prompt to send to the model.

        Returns:
            Model's response string.
        """
        # Record input message
        self._messages.append({"role": "user", "content": prompt})

        # Generate response
        response = self.model.generate(prompt)

        # Record output message
        self._messages.append({"role": "assistant", "content": response})

        return response

    def gather_traces(self) -> Dict[str, Any]:
        """Gather traces for this agent."""
        return {
            **super().gather_traces(),
            "name": self.name,
            "messages": list(self._messages),
        }

    def gather_config(self) -> Dict[str, Any]:
        """Gather configuration."""
        return {
            **super().gather_config(),
            "name": self.name,
            "model_id": self.model.model_id,
        }


class MMLUAgentAdapter(AgentAdapter):
    """AgentAdapter wrapper for MMLUModelAgent."""

    def __init__(self, agent: MMLUModelAgent, name: str):
        """Initialize adapter.

        Args:
            agent: MMLUModelAgent instance.
            name: Adapter name.
        """
        super().__init__(agent, name)

    def _run_agent(self, query: str) -> Any:
        """Execute the agent."""
        return self.agent.run(query)

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get agent messages."""
        return self.agent._messages


# =============================================================================
# Benchmark
# =============================================================================


class MMLUBenchmark(Benchmark):
    """MMLU Benchmark - Framework-agnostic base class.

    Evaluates language models on MMLU multiple choice questions.
    Supports anchor point-based evaluation for DISCO prediction.

    Users must subclass and implement:
    - get_model_adapter() to provide model adapters

    Usage:
        class MyMMLUBenchmark(MMLUBenchmark):
            def get_model_adapter(self, model_id, **kwargs):
                from transformers import pipeline
                from maseval.interface.inference import HuggingFaceModelAdapter
                pipe = pipeline("text-generation", model=model_id)
                return HuggingFaceModelAdapter(model=pipe, model_id=model_id)

        benchmark = MyMMLUBenchmark()
        results = benchmark.run(tasks=tasks, agent_data={"model_id": "llama-7b"})
    """

    def __init__(
        self,
        use_full_prompt: bool = False,
        callbacks: Optional[List[Any]] = None,
        n_task_repeats: int = 1,
        **kwargs: Any,
    ):
        """Initialize benchmark.

        Args:
            use_full_prompt: If True, use full_prompt (with few-shot examples)
                instead of just the query.
            callbacks: Benchmark callbacks.
            n_task_repeats: Repetitions per task.
        """
        super().__init__(callbacks=callbacks, n_task_repeats=n_task_repeats, max_invocations=1, **kwargs)
        self.use_full_prompt = use_full_prompt

    def setup_environment(
        self,
        agent_data: Dict[str, Any],
        task: Task,
    ) -> MMLUEnvironment:
        """Create environment for a task."""
        task_data = {
            "query": task.query,
            "environment_data": {
                **task.environment_data,
                "use_full_prompt": self.use_full_prompt or agent_data.get("use_full_prompt", False),
            },
        }
        return MMLUEnvironment(task_data)

    def setup_user(
        self,
        agent_data: Dict[str, Any],
        environment: MMLUEnvironment,
        task: Task,
    ) -> None:
        """MMLU doesn't use a user simulator."""
        return None

    def setup_agents(
        self,
        agent_data: Dict[str, Any],
        environment: MMLUEnvironment,
        task: Task,
        user: Optional[Any],
    ) -> Tuple[Sequence[AgentAdapter], Dict[str, AgentAdapter]]:
        """Create model agent for MCQ evaluation.

        Args:
            agent_data: Agent config with model_id.
            environment: MMLU environment.
            task: Current task.
            user: Unused.

        Returns:
            Tuple of (agents_to_run, agents_dict).
        """
        model_id = agent_data.get("model_id", "unknown")
        model = self.get_model_adapter(model_id, register_name="mmlu_model")

        agent = MMLUModelAgent(model, name="mmlu_agent")
        adapter = MMLUAgentAdapter(agent, "mmlu_agent")

        return [adapter], {"mmlu_agent": adapter}

    def setup_evaluators(
        self,
        environment: MMLUEnvironment,
        task: Task,
        agents: Sequence[AgentAdapter],
        user: Optional[Any],
    ) -> Sequence[Evaluator]:
        """Create MMLU evaluator."""
        return [MMLUEvaluator(task, environment)]

    def run_agents(
        self,
        agents: Sequence[AgentAdapter],
        task: Task,
        environment: MMLUEnvironment,
        query: str = "",
    ) -> Any:
        """Execute agent on the MMLU prompt."""
        # Get the prompt from environment
        prompt = environment.get_prompt()

        # Run the agent
        agent = agents[0]
        return agent.run(prompt)

    @abstractmethod
    def get_model_adapter(self, model_id: str, **kwargs) -> ModelAdapter:
        """Provide a ModelAdapter for the model.

        Must be implemented by subclass.

        Args:
            model_id: Model identifier.
            **kwargs: Additional arguments (e.g., register_name for tracing).

        Returns:
            ModelAdapter instance.
        """
        pass

    def evaluate(
        self,
        evaluators: Sequence[Evaluator],
        agents: Dict[str, AgentAdapter],
        final_answer: Any,
        traces: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate model response."""
        results = []
        for evaluator in evaluators:
            filtered_traces = evaluator.filter_traces(traces)
            result = evaluator(filtered_traces, final_answer)
            results.append(result)
        return results


# =============================================================================
# Data Loading
# =============================================================================


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tasks(
    data_path: Union[str, Path],
    anchor_points_path: Optional[Union[str, Path]] = None,
    limit: Optional[int] = None,
) -> Union[AnchorPointsTaskQueue, SequentialTaskQueue]:
    """Load MMLU tasks from JSON file.

    Args:
        data_path: Path to MMLU prompts JSON file (mmlu_prompts_examples.json format).
        anchor_points_path: Optional path to anchor points pickle file.
            If provided, returns an AnchorPointsTaskQueue that evaluates
            only the anchor tasks in order.
        limit: Optional limit on number of tasks to load.

    Returns:
        TaskQueue containing MMLU tasks.

    Raises:
        ImportError: If anchor_points_path is provided but numpy is not installed.
    """
    data_path = Path(data_path)

    # Load JSON data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Apply limit before filtering
    if limit is not None:
        data = data[:limit]

    # Convert to Tasks
    tasks = []
    for i, item in enumerate(data):
        task = Task(
            query=item.get("query", item.get("example", "")),
            id=f"mmlu_{i}",
            environment_data={
                "choices": item.get("choices", ["A", "B", "C", "D"]),
                "full_prompt": item.get("full_prompt", ""),
                "example": item.get("example", ""),
            },
            evaluation_data={
                "gold": item.get("gold", 0),
            },
            metadata={
                "doc_id": i,
                "task_type": "mmlu",
            },
        )
        tasks.append(task)

    # Load anchor points if provided
    anchor_points = None
    if anchor_points_path is not None:
        anchor_points = load_pickle(anchor_points_path)
        # Convert numpy array to list if necessary
        if HAS_NUMPY and isinstance(anchor_points, np.ndarray):
            anchor_points = anchor_points.tolist()
        elif not HAS_NUMPY and hasattr(anchor_points, "tolist"):
            # Fallback: try tolist() method if it exists
            anchor_points = anchor_points.tolist()

    # Create appropriate queue
    if anchor_points is not None:
        return AnchorPointsTaskQueue(tasks, anchor_points)
    else:
        return SequentialTaskQueue(tasks)


def compute_benchmark_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics across all benchmark results.

    Args:
        results: List of result dicts from benchmark.run().

    Returns:
        Dict with accuracy metrics and task counts.
    """
    if not results:
        return {
            "total_tasks": 0,
            "acc": 0.0,
            "acc_norm": 0.0,
        }

    total_tasks = len(results)
    correct_count = 0
    acc_sum = 0.0
    acc_norm_sum = 0.0

    for res in results:
        if res.get("status") != "success":
            continue

        evals = res.get("eval") or []
        for entry in evals:
            acc_sum += entry.get("acc", 0.0)
            acc_norm_sum += entry.get("acc_norm", 0.0)
            if entry.get("correct", False):
                correct_count += 1

    return {
        "total_tasks": total_tasks,
        "correct_count": correct_count,
        "acc": acc_sum / total_tasks if total_tasks > 0 else 0.0,
        "acc_norm": acc_norm_sum / total_tasks if total_tasks > 0 else 0.0,
    }
