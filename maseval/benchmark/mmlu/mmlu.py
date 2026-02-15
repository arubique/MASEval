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
# Constants (configurable defaults)
# =============================================================================

DEFAULT_CHOICES = ["A", "B", "C", "D"]
DEFAULT_DEVICE = "cuda:0"
DEFAULT_BATCH_SIZE = 8
DEFAULT_AGENT_NAME = "mmlu_agent"
DEFAULT_MODEL_REGISTER_NAME = "mmlu_model"
TARGET_DELIMITER = " "  # lm-eval convention for MCQ
MMLU_TASK_NAME = "mmlu_prompts"
TASK_TYPE_MMLU = "mmlu"
FALLBACK_MODEL_ID = "unknown"
STATUS_SUCCESS = "success"


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
        self.choices = task.environment_data.get("choices", DEFAULT_CHOICES)

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
        for i, choice in enumerate(DEFAULT_CHOICES):
            if response == choice or response.startswith(f"{choice}."):
                return i

        # Look for "answer is X" pattern
        for i, choice in enumerate(DEFAULT_CHOICES):
            if f"ANSWER IS {choice}" in response:
                return i
            if f"ANSWER: {choice}" in response:
                return i

        # Last character check
        last_char = response.rstrip(".")[-1] if response else ""
        for i, choice in enumerate(DEFAULT_CHOICES):
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

    def __init__(self, model: ModelAdapter, name: str = DEFAULT_AGENT_NAME):
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

    def gather_traces(self) -> Dict[str, Any]:
        """Gather execution traces from this agent.

        Override to handle plain list messages (not MessageHistory).
        """
        from maseval.core.tracing import TraceableMixin

        messages = self.get_messages()
        return {
            **TraceableMixin.gather_traces(self),
            "name": self.name,
            "agent_type": type(self.agent).__name__,
            "message_count": len(messages),
            "messages": messages,  # Already a list, no need for to_list()
            "callbacks": [type(cb).__name__ for cb in self.callbacks],
            "logs": self.logs,
        }


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
        model_id = agent_data.get("model_id", FALLBACK_MODEL_ID)
        model = self.get_model_adapter(model_id, register_name=DEFAULT_MODEL_REGISTER_NAME)

        agent = MMLUModelAgent(model, name=DEFAULT_AGENT_NAME)
        adapter = MMLUAgentAdapter(agent, DEFAULT_AGENT_NAME)

        return [adapter], {DEFAULT_AGENT_NAME: adapter}

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


class HuggingFaceMMLUBenchmark(MMLUBenchmark):
    """MMLU Benchmark using HuggingFace transformers models.

    This concrete implementation uses log-likelihood based MCQ evaluation
    with the same optimizations as lm-evaluation-harness:

    1. Single forward pass per question (one-token continuation optimization)
    2. Batching multiple questions together
    3. Efficient log-softmax computation
    4. Proper left-padding for batch processing
    """

    def __init__(
        self,
        model_id: str,
        device: str = DEFAULT_DEVICE,
        trust_remote_code: bool = True,
        use_full_prompt: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        **kwargs,
    ):
        """Initialize HuggingFace MMLU benchmark.

        Args:
            model_id: HuggingFace model identifier.
            device: Device to run model on.
            trust_remote_code: Trust remote code when loading model (default True).
            use_full_prompt: Use full prompt with few-shot examples (default True).
            batch_size: Batch size for evaluation (number of questions per batch).
            **kwargs: Additional arguments passed to MMLUBenchmark.
        """
        super().__init__(use_full_prompt=use_full_prompt, **kwargs)
        self._model_id = model_id
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._batch_size = batch_size
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load the model and tokenizer for log-likelihood computation."""
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading model: {self._model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id,
                trust_remote_code=self._trust_remote_code,
            )
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model with torch_dtype="auto" to match lm-evaluation-harness exactly
            # This uses the model's native dtype (bfloat16 for most modern models)
            # Then move to device manually
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_id,
                trust_remote_code=self._trust_remote_code,
                torch_dtype="auto",
            )
            self._model = self._model.to(self._device)
            self._model.eval()

            # Note: We don't pre-cache choice token IDs here because they depend on context.
            # Token IDs are computed dynamically in _get_choice_token_id_in_context()
            # to match lm-evaluation-harness behavior exactly.

        return self._model, self._tokenizer

    def _get_choice_token_id_separate(self, choice: str) -> int:
        """Get the token ID for a choice when tokenized SEPARATELY.

        CRITICAL: lm-evaluation-harness encodes context and continuation separately,
        then concatenates. This means "A" is always tokenized standalone (token 330),
        NOT in context after "Answer:" (which would be token 28741).

        We must match this behavior to get identical log-likelihood values.

        Args:
            choice: The choice string (e.g., "A").

        Returns:
            Token ID for the choice (standalone tokenization).
        """
        _, tokenizer = self._load_model()

        # Tokenize choice ALONE (not in context) - this is how lm-eval does it
        choice_tokens = tokenizer.encode(choice, add_special_tokens=False)

        if len(choice_tokens) == 1:
            return choice_tokens[0]
        else:
            # Multi-token choice - return None to trigger multi-token fallback
            return None

    def _encode_pair(self, context: str, continuation: str) -> tuple:
        """Encode a context-continuation pair like lm-evaluation-harness.

        This matches lm-eval's _encode_pair method exactly:
        1. Encode whole = context + continuation
        2. Encode context alone
        3. continuation_enc = whole[len(context_enc):]

        This handles tokenization boundary effects correctly.

        Args:
            context: The context/prompt string.
            continuation: The continuation string (e.g., " A" with target_delimiter).

        Returns:
            Tuple of (context_enc, continuation_enc) token lists.
        """
        _, tokenizer = self._load_model()

        # Handle trailing spaces in context (move to continuation)
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        # Encode whole string together, then split
        whole_enc = tokenizer.encode(context + continuation, add_special_tokens=True)
        context_enc = tokenizer.encode(context, add_special_tokens=True)

        # Continuation tokens are what's left after context
        continuation_enc = whole_enc[len(context_enc) :]

        return context_enc, continuation_enc

    def _compute_logprobs_single_token(self, prompt: str, choices: list) -> list:
        """Compute log-likelihoods using single-token optimization.

        For MCQ with single-letter answers (A, B, C, D), we can compute all
        choices in one forward pass since they share the same context.

        IMPORTANT: To match lm-evaluation-harness EXACTLY:
        1. Use target_delimiter=" " before choices (e.g., " A" not "A")
        2. Use _encode_pair to handle tokenization boundaries correctly
        3. Input = (context + continuation)[:-1]
        4. Apply log_softmax to get log probabilities

        Args:
            prompt: The prompt/question text.
            choices: List of answer choice strings (e.g., ["A", "B", "C", "D"]).

        Returns:
            List of log-likelihoods, one per choice.
        """
        import torch

        model, _ = self._load_model()

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = TARGET_DELIMITER

        # Encode first choice to get the shared context
        first_continuation = f"{target_delimiter}{choices[0]}"
        context_enc, first_cont_enc = self._encode_pair(prompt, first_continuation)

        # Build input: (context + continuation)[:-1]
        full_sequence = context_enc + first_cont_enc
        input_tokens = full_sequence[:-1]  # Remove last token

        input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self._device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab_size)

            # Select logits at position where continuation is predicted
            # For single-token continuation, this is the last position
            inplen = len(input_tokens)
            contlen = len(first_cont_enc)
            selected_logits = logits[inplen - contlen : inplen]

            # Compute log-softmax
            log_probs = torch.nn.functional.log_softmax(selected_logits, dim=-1)

            # Get log prob for each choice's continuation token
            logprobs = []
            for choice in choices:
                continuation = f"{target_delimiter}{choice}"
                _, cont_enc = self._encode_pair(prompt, continuation)

                # Sum log probs for multi-token continuations
                total = 0.0
                for i, token_id in enumerate(cont_enc):
                    total += log_probs[i, token_id].item()
                logprobs.append(total)

        return logprobs

    def _compute_logprobs_batched(self, prompts: list, choices_list: list) -> list:
        """Compute log-likelihoods for a batch of prompts.

        For exact match with lm-evaluation-harness, we process each prompt
        individually using _compute_logprobs_single_token which uses the
        correct _encode_pair tokenization logic.

        Args:
            prompts: List of prompt strings.
            choices_list: List of choice lists (one per prompt).

        Returns:
            List of log-likelihood lists, one per prompt.
        """
        # For exact match with lm-eval, process individually
        # This ensures correct tokenization via _encode_pair
        all_logprobs = []
        for prompt, choices in zip(prompts, choices_list):
            logprobs = self._compute_logprobs_single_token(prompt, choices)
            all_logprobs.append(logprobs)

        return all_logprobs

    def precompute_all_logprobs_lmeval(self, tasks) -> dict:
        """Precompute log-likelihoods for ALL tasks using lm-eval's batching.

        CRITICAL: lm-evaluation-harness batches ALL requests together and uses
        its Collator class to reorder/group them. This affects floating-point
        precision for some edge cases. To get EXACT matches, we must process
        ALL requests together in a single batch.

        This method:
        1. Creates Instance objects for all task/choice combinations
        2. Calls lm-eval's HFLM.loglikelihood() with ALL instances
        3. Returns a mapping from doc_id to logprobs

        Args:
            tasks: Iterable of Task objects with prompt and choices.

        Returns:
            Dict mapping doc_id -> list of log-likelihoods for each choice.
        """
        import sys

        # Add lm-eval to path
        sys.path.insert(0, "/home/oh/arubinstein17/github/disco-public")
        sys.path.insert(0, "/home/oh/arubinstein17/github/disco-public/external/lm-evaluation-harness")

        from lm_eval.models.huggingface import HFLM
        from lm_eval.api.instance import Instance

        # Create HFLM model (this handles model loading internally)
        print(f"Loading HFLM model for {self._model_id}")
        lm = HFLM(
            pretrained=self._model_id,
            trust_remote_code=self._trust_remote_code,
            batch_size=self._batch_size,
            device=self._device,
        )

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = TARGET_DELIMITER
        choices = DEFAULT_CHOICES
        continuations = [f"{target_delimiter}{c}" for c in choices]

        # Build ALL instances like lm-eval task system does
        all_instances = []
        instance_map = {}  # (doc_id, choice_idx) -> position in results

        for task in tasks:
            doc_id = task.metadata.get("doc_id")
            # Get prompt from task - use full_prompt from environment_data if available
            if self.use_full_prompt and "full_prompt" in task.environment_data:
                prompt = task.environment_data["full_prompt"]
            else:
                prompt = task.query

            for i, cont in enumerate(continuations):
                inst = Instance(
                    request_type="loglikelihood",
                    doc={"doc_id": doc_id},
                    arguments=(prompt, cont),
                    idx=i,
                    metadata=(MMLU_TASK_NAME, doc_id, 1),
                )
                instance_map[(doc_id, i)] = len(all_instances)
                all_instances.append(inst)

        print(f"Precomputing logprobs for {len(all_instances)} instances ({len(all_instances) // len(choices)} tasks)")

        # Call loglikelihood with ALL instances at once - this is the key!
        results = lm.loglikelihood(all_instances)

        # Map results back to doc_ids
        doc_logprobs = {}
        for task in tasks:
            doc_id = task.metadata.get("doc_id")
            logprobs = []
            for i in range(len(choices)):
                pos = instance_map[(doc_id, i)]
                logprob, _ = results[pos]
                logprobs.append(logprob)
            doc_logprobs[doc_id] = logprobs

        # Store for later use
        self._precomputed_logprobs = doc_logprobs

        return doc_logprobs

    def _compute_logprobs_multi_token(self, prompt: str, choices: list) -> list:
        """Compute log-likelihoods for multi-token continuations.

        This is the fallback for when answer choices have multiple tokens.
        Uses _encode_pair to match lm-evaluation-harness exactly.

        Args:
            prompt: The prompt/question text.
            choices: List of answer choice strings.

        Returns:
            List of log-likelihoods, one per choice.
        """
        import torch

        model, _ = self._load_model()

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = TARGET_DELIMITER

        all_logprobs = []
        for choice in choices:
            continuation = f"{target_delimiter}{choice}"

            # Use _encode_pair for correct tokenization
            context_enc, continuation_enc = self._encode_pair(prompt, continuation)

            # Build input: (context + continuation)[:-1]
            full_sequence = context_enc + continuation_enc
            input_tokens = full_sequence[:-1]

            input_ids = torch.tensor([input_tokens], dtype=torch.long, device=self._device)

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0]  # (seq_len, vocab_size)

                # Select continuation logits
                inplen = len(input_tokens)
                contlen = len(continuation_enc)
                selected = logits[inplen - contlen : inplen]

                # Compute log-softmax
                log_probs = torch.nn.functional.log_softmax(selected, dim=-1)

                # Sum log probs for all continuation tokens
                total = 0.0
                for i, token_id in enumerate(continuation_enc):
                    total += log_probs[i, token_id].item()

                all_logprobs.append(total)

        return all_logprobs

    def run_agents(
        self,
        agents,
        task,
        environment,
        query: str = "",
    ):
        """Execute log-likelihood based MCQ evaluation.

        Uses precomputed logprobs if available (for exact lm-eval match),
        otherwise falls back to single-forward-pass optimization for
        single-token answers, or multi-token batched computation.
        """
        # Get the prompt from environment
        prompt = environment.get_prompt()
        choices = environment.state.get("choices", DEFAULT_CHOICES)
        doc_id = task.metadata.get("doc_id") if task else None

        # Check if we have precomputed logprobs (for exact lm-eval match)
        if hasattr(self, "_precomputed_logprobs") and doc_id is not None:
            logprobs = self._precomputed_logprobs.get(doc_id)
            if logprobs is not None:
                # Use precomputed values for exact match
                best_idx = logprobs.index(max(logprobs))
                answer = choices[best_idx]

                # Store logprobs in environment for later retrieval
                environment.state["logprobs"] = logprobs
                environment.state["predicted_idx"] = best_idx

                # Record in agent messages for tracing
                agent = agents[0]
                agent.agent._messages.append({"role": "user", "content": prompt})
                agent.agent._messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "logprobs": logprobs,
                    }
                )

                return answer

        # Fall back to computing logprobs on-the-fly
        # Load model
        self._load_model()

        # lm-eval uses target_delimiter=" " for multiple choice tasks
        target_delimiter = TARGET_DELIMITER

        # Check if all choices result in single-token continuations
        # using _encode_pair to get the correct tokenization
        all_single_token = True
        for choice in choices:
            continuation = f"{target_delimiter}{choice}"
            _, cont_enc = self._encode_pair(prompt, continuation)
            if len(cont_enc) != 1:
                all_single_token = False
                break

        if all_single_token:
            # Use optimized single-token path (one forward pass)
            logprobs = self._compute_logprobs_single_token(prompt, choices)
        else:
            # Fall back to multi-token computation
            logprobs = self._compute_logprobs_multi_token(prompt, choices)

        # Select the choice with highest log-probability
        best_idx = logprobs.index(max(logprobs))
        answer = choices[best_idx]

        # Store logprobs in environment for later retrieval if needed
        environment.state["logprobs"] = logprobs
        environment.state["predicted_idx"] = best_idx

        # Record in agent messages for tracing
        agent = agents[0]
        agent.agent._messages.append({"role": "user", "content": prompt})
        agent.agent._messages.append(
            {
                "role": "assistant",
                "content": answer,
                "logprobs": logprobs,
            }
        )

        return answer

    def get_model_adapter(self, model_id: str, **kwargs):
        """Provide a HuggingFace ModelAdapter.

        Note: For logprobs-based evaluation, we don't actually use the adapter
        for generation. This is kept for API compatibility.

        Args:
            model_id: Model identifier (ignored, uses instance model_id).
            **kwargs: Additional arguments (e.g., register_name).

        Returns:
            HuggingFaceModelAdapter instance.
        """
        from maseval.interface.inference import HuggingFaceModelAdapter

        # Create a minimal adapter for compatibility
        # The actual evaluation uses _compute_logprobs_*
        class DummyCallable:
            def __call__(self, prompt, **kwargs):
                return ""

        adapter = HuggingFaceModelAdapter(
            model=DummyCallable(),
            model_id=self._model_id,
        )

        # Register for tracing if requested
        register_name = kwargs.get("register_name")
        if register_name:
            self.register("models", register_name, adapter)

        return adapter


# =============================================================================
# Data Loading
# =============================================================================


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_anchor_points(path: Union[str, Path]) -> List[int]:
    """Load anchor points from a .json or .pkl file. Returns a list of doc_ids."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Anchor points file not found: {path}")
    if path.suffix.lower() == ".json":
        with open(path) as f:
            anchor_points = json.load(f)
    else:
        anchor_points = load_pickle(path)
    if HAS_NUMPY and isinstance(anchor_points, np.ndarray):
        anchor_points = anchor_points.tolist()
    elif not HAS_NUMPY and hasattr(anchor_points, "tolist"):
        anchor_points = anchor_points.tolist()
    return list(anchor_points)


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
                "choices": item.get("choices", DEFAULT_CHOICES),
                "full_prompt": item.get("full_prompt", ""),
                "example": item.get("example", ""),
            },
            evaluation_data={
                "gold": item.get("gold", 0),
            },
            metadata={
                "doc_id": i,
                "task_type": TASK_TYPE_MMLU,
            },
        )
        tasks.append(task)

    # Load anchor points if provided
    anchor_points = None
    if anchor_points_path is not None:
        anchor_points = load_anchor_points(anchor_points_path)

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
        if res.get("status") != STATUS_SUCCESS:
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
