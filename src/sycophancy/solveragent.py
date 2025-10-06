"""
This module defines a solver agent for math problems."""

from loguru import logger
from .agent import Agent, AgentInstance

class SolverAgentInstance(AgentInstance):
    def __init__(self, problem_statement, model_config, prompts, return_if_not_found=False, 
                 max_iterations=30, correct_count=5, error_count=10):
        super().__init__(problem_statement, model_config)
        self.prompts = prompts
        self.total_cost = {
            "output_tokens": 0,
            "input_tokens": 0,
            "cost": 0
        }
        self.return_if_not_found = return_if_not_found
        self.max_iterations = max_iterations
        self.correct_count = correct_count
        self.error_count = error_count
        self.problem_statement = problem_statement
    
    def remove_think(self, solution):
        """
        Removes the <think> and </think> tags from the solution string.

        Returns the substring after the last </think> tag, stripped of
        leading/trailing whitespace. If no </think> tag is found, returns the
        original solution.

        Args:
            solution (str): The solution string.

        Returns:
            str: The solution string with the think tags removed.
        """
        if "</think>" in solution:
            return solution.split("</think>")[-1].strip()
        if "</summary>" in solution:
            return solution.split("</summary>")[-1].strip()
        return solution.strip() 

    def extract_detailed_solution(self, solution, marker='Detailed Solution', after=True):
        """
        Extracts the text after '### Detailed Solution ###' from the solution string.

        Returns the substring after the marker, stripped of leading/trailing
        whitespace. If the marker is not found, returns an empty string.

        Args:
            solution (str): The solution string.
            marker (str, optional): The marker to search for. Defaults to 'Detailed Solution'.
            after (bool, optional): Whether to return the text after the marker.
                Defaults to True.

        Returns:
            str: The extracted text.
        """
        idx = solution.find(marker)
        if idx == -1:
            logger.debug(f"Marker '{marker}' not found in solution.")
            return ''
        if(after):
            return solution[idx + len(marker):].strip()
        else:
            return solution[:idx].strip()
    
    def verify_solution(self, problem_statement, solution, history):
        """
        Verifies a solution.

        Args:
            problem_statement (str): The problem statement.
            solution (str): The solution to verify.
            history (list): The history of the agent's steps.

        Returns:
            tuple: A tuple containing the bug report, the verification result, and the cost.
        """
        dsol = self.extract_detailed_solution(solution)
        logger.debug("Verifying solution...")

        newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{self.prompts['verification_reminder_prompt']}
"""
        query = [
            {
                "role": "system",
                "content": self.prompts['verification_system_prompt']
            },
            {
                "role": "user",
                "content": newst
            }
        ]
        
        _, out, cost = list(self.querier.run_queries([query]))[0]

        out = out[-1]["content"]

        out = self.remove_think(out)

        history.append({
            "step": "verify_solution",
            "query": query,
            "solution": out,
            "cost": cost
        })


        check_correctness = """Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?""" \
                + "\n\n" + out
        query = [
            {"role": "user", "content": check_correctness},
        ]
        _, o, cost2 = list(self.querier.run_queries([query]))[0]

        o = o[-1]["content"]

        o = self.remove_think(o)

        history.append({
            "step": "verify_solution_2",
            "query": query,
            "solution": o,
            "cost": cost2
        })

        bug_report = ""

        if("yes" not in o.lower()):
            logger.debug("Verification failed. Extracting bug report.")
            bug_report = self.extract_detailed_solution(out, "Detailed Verification", False)
        else:
            logger.debug("Verification successful.")

        self.total_cost["output_tokens"] += cost['output_tokens'] + cost2['output_tokens']
        self.total_cost["input_tokens"] += cost['input_tokens'] + cost2['input_tokens']
        self.total_cost["cost"] += cost['cost'] + cost2['cost']
        
        return bug_report, o
    
    def init_explorations(self, problem_statement, verbose=True, history=[]):
        """
        Initializes the explorations for a problem.

        Args:
            problem_statement (str): The problem statement.
            verbose (bool, optional): Whether to log verbose output. Defaults to True.
            history (list, optional): The history of the agent's steps. Defaults to [].

        Returns:
            tuple: A tuple containing the query, solution, verification result,
                good verification result, and cost.
        """
        logger.debug("Initializing explorations.")
        query = [
            {"role": "system", "content": self.prompts['system_prompt']},
            {"role": "user", "content": problem_statement}
        ]
        
        _, output1, cost = list(self.querier.run_queries([query]))[0]

        output1 = output1[-1]["content"]

        output1 = self.remove_think(output1)

        query.append(
            {
                "role": "assistant",
                "content": output1,
                "cost": cost
            }
        )
        query.append(
            {"role": "user",
            "content": self.prompts['self_improvement_prompt']
            }
        )

        _, solution, cost2 = list(self.querier.run_queries([query]))[0]

        solution = solution[-1]["content"]

        solution = self.remove_think(solution)

        history.append({
            "step": "init",
            "query": query,
            "solution": solution,
            "cost": cost2
        })
        
        verify, good_verify = self.verify_solution(problem_statement, solution, history)

        self.total_cost["output_tokens"] += cost['output_tokens'] + cost2['output_tokens']
        self.total_cost["input_tokens"] += cost['input_tokens'] + cost2['input_tokens']
        self.total_cost["cost"] += cost['cost'] + cost2['cost']

        return query, solution, verify, good_verify
    
    def run(self):
        """
        The main agent logic.

        Args:
            problem_statement (str or tuple): The problem statement.

        Returns:
            tuple: A tuple containing the solution, cost, and history.
        """
        if isinstance(self.problem_statement, tuple):
            self.problem_statement = self.problem_statement[0]
        history = []
        logger.debug(f"Starting agent for problem: {self.problem_statement[:50]}...")
        query, solution, verify, good_verify = self.init_explorations(self.problem_statement, True, 
                                                                            history)

        if (solution is None):
            logger.debug("Agent failed during initialization.")
            return None, self.total_cost, history

        error_count = 0
        correct_count = 0
        
        for index_iteration in range(self.max_iterations):
            logger.debug(f"Agent iteration {index_iteration+1}")
            logger.debug(f"Number of iterations: {index_iteration}, number of corrects: {correct_count}, number of errors: {error_count}")

            if("yes" not in good_verify.lower()):
                # clear
                correct_count = 0
                error_count += 1

                #self improvement
                logger.debug("Verification failed, attempting self-improvement.")
                # establish a new prompt that contains the solution and the verification
                query = [
                    {"role": "system", "content": self.prompts['system_prompt']},
                    {"role": "user", "content": self.problem_statement}
                ]
                query.append({
                    "role": "assistant",
                    "content": solution
                })

                query.append({
                    "role": "user",
                    "content": self.prompts['correction_prompt'] + "\n\n" + verify
                })

                _, solution, cost3 = list(self.querier.run_queries([query]))[0]
                solution = solution[-1]["content"]
                solution = self.remove_think(solution)
                self.total_cost["output_tokens"] += cost3['output_tokens']
                self.total_cost["input_tokens"] += cost3['input_tokens']
                self.total_cost["cost"] += cost3['cost']

                history.append(
                    {
                        "step": "self_improvement",
                        "query": query,
                        "solution": solution,
                        "cost": cost3
                    }
                )

            logger.debug("Verifying the current solution.")
            verify, good_verify = self.verify_solution(self.problem_statement, solution, history)

            if ("yes" in good_verify.lower()):
                logger.debug("Solution verified.")
                correct_count += 1
                error_count = 0
            else:
                logger.debug("Solution still not correct after verification.")
    
            if (correct_count >= self.correct_count):
                logger.success("Agent found a solution.")
                return solution, self.total_cost, history

            elif (error_count >= self.error_count):
                logger.debug("Agent failed to find a solution after correction attempt.")
                return solution if self.return_if_not_found else None, self.total_cost, history

        logger.debug("Agent failed to find a solution within 30 iterations.")
        return solution if self.return_if_not_found else None, self.total_cost, history

class SolverAgent(Agent):
    """
    An agent that attempts to solve math problems through a series of
    self-correction and verification steps.
    """
    def __init__(self, model_config, prompts, return_if_not_found=False, 
                 max_iterations=30, correct_count=5, error_count=10, 
                 n_threads=16):
        """
        Initializes the SolverAgent.

        Args:
            model_config (dict): The configuration for the model.
            prompts (dict): A dictionary of prompts for the agent.
            return_if_not_found (bool, optional): Whether to return the last solution
                if a correct one is not found. Defaults to False.
            max_iterations (int, optional): The maximum number of iterations. Defaults to 30.
            correct_count (int, optional): The number of consecutive correct verifications
                required to consider a solution correct. Defaults to 5.
            error_count (int, optional): The number of consecutive errors before
                giving up. Defaults to 10.
            n_threads (int, optional): The number of threads to use. Defaults to 16.
        """
        super().__init__(model_config, n_threads)
        self.prompts = prompts
        self.total_cost = dict()
        self.return_if_not_found = return_if_not_found
        self.max_iterations = max_iterations
        self.correct_count = correct_count
        self.error_count = error_count

    def create_new_instance(self, problem_statement):
        """
        Creates a new instance of the SolverAgentInstance with the same configuration.

        Args:
            problem_statement (str): The problem statement for the new instance.

        Returns:
            SolverAgentInstance: A new instance of the SolverAgentInstance class.
        """
        return SolverAgentInstance(problem_statement, self.model_config, 
                                   self.prompts, self.return_if_not_found, 
                                   self.max_iterations, self.correct_count, 
                                   self.error_count)
    
