"""This module defines the base Agent class for solving math problems."""

from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from .api import APIQuery
import time

class AgentInstance:
    def __init__(self, problem_statement, model_config):
        self.problem_statement = problem_statement
        self.querier = APIQuery(**model_config)

    def run(self):
        raise NotImplementedError("Subclasses should implement this method.")


class Agent:
    """
    A base class for agents that solve math problems.

    Attributes:
        model_config (dict): Configuration for the model.
        n_threads (int): The number of threads to use for solving problems in parallel.
        querier (APIQuery): An instance of the APIQuery class for making API calls.
    """
    def __init__(self, model_config, n_threads=16):
        """
        Initializes the Agent.

        Args:
            model_config (dict): Configuration for the model.
            n_threads (int): The number of threads to use for solving problems in parallel.
        """
        self.n_threads = n_threads
        self.model_config = model_config
        self.querier = APIQuery(**model_config)

    def create_new_instance(self, problem_statement):
        """
        Creates a new instance of the Agent with the same configuration.

        Args:
            problem_statement (str): The problem statement for the new instance.
        Returns:
            AgentInstance: A new instance of the AgentInstance class.
        """
        return NotImplementedError("Subclasses should implement this method.")
    
    def run(self, problem):
        """
        Solves a single problem using the agent.

        Args:
            problem (str): The problem statement.
        """
        instance = self.create_new_instance(problem)
        return instance.run()

    def solve(self, problems, **tokenizer_kwargs):
        """
        Solves a list of problems using the agent.

        Args:
            problems (list): A list of problem statements to be solved.
            **tokenizer_kwargs: Additional keyword arguments for the tokenizer.

        Yields:
            tuple: A tuple containing the index, solution, cost, and history for each problem.
        """
        logger.info("Solving problems with the agent.")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(self.run, problem) for problem in problems]
            for idx, future in enumerate(futures):
                solution, cost, history = future.result()
                if isinstance(solution, str):
                    solution = [{
                        "role": "assistant",
                        "content": solution if solution is not None else "No solution found."
                    }]
                cost["time"] = time.time() - start_time
                yield idx, solution, cost, history