from string import Formatter
from loguru import logger
from sycophancy.baselines import VerifierBaseline
class Scanner():
    def __init__(self, system_prompt=None, prompt_template=None, querier=None, **kwargs):
        assert(querier is not None), "The verifier must define a model querier"
        self.querier = querier
        self.cost = 0
        self.detailed_cost = []
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
    
    def safe_format_prompt(self, problem_statement, model_solution, original_statement, gt_solution):
        data = {
            'problem': problem_statement,
            'solution': model_solution,
            'original_statement': original_statement,
            'gt_solution': gt_solution
        }

        return self.prompt_template.format(**data)

    def build_query(self, problem_statement, model_solution, original_statement, gt_solution):
        messages = []
        if self.system_prompt is not None: 
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.safe_format_prompt(problem_statement, model_solution, original_statement, gt_solution)})
        return messages
    
    def build_queries(self, problems):
        """
        Build a list of queries from a list of problems.

        Args:
            problems (list): A list of problem instances.

        Returns:
            list: A list of queries generated from the given problems.
        """
        queries = []
        for problem in problems:
            gt_solution = problem.get('solutions', [])[0]['solution'] if problem.get('solutions', []) else None
            queries.append(self.build_query(problem["problem"], problem["solution"], problem['original_problem'], gt_solution))
        return queries
    
    def add_response(self, query, response):
        if isinstance(response, tuple) and response[0] is None:
            query.append({"role": "api_error", "content": str(response[1])})
        else:
            query.append({"role": "assistant", "content": response})
        return query

    def verify(self, problems):
        """
        Solves the initial round of problems by building queries, running them, and appending responses.

        Args:
            problems (list): A list of problems to be solved.

        Returns:
            list: A list of queries with appended responses from the assistant.
        """
        logger.info("Verifying problems.")
        queries = self.build_queries(problems)

        self.cost = 0
        self.detailed_cost = [{
            "cost": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        } for _ in range(len(problems))]

        for idx, response, detailed_cost in self.querier.run_queries(queries):
            messages = self.add_response(queries[idx], response)
            self.detailed_cost[idx]["cost"] += detailed_cost["cost"]
            self.detailed_cost[idx]["input_tokens"] += detailed_cost["input_tokens"]
            self.detailed_cost[idx]["output_tokens"] += detailed_cost["output_tokens"]
            yield idx, messages, detailed_cost

class Filter(Scanner):
    def safe_format_prompt(self, problem_statement):
        data = {
            'problem': problem_statement
        }

        return self.prompt_template.format(**data)

    def build_query(self, problem_statement):
        messages = []
        if self.system_prompt is not None: 
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.safe_format_prompt(problem_statement)})
        return messages
    
    def build_queries(self, problems):
        """
        Build a list of queries from a list of problems.

        Args:
            problems (list): A list of problem instances.

        Returns:
            list: A list of queries generated from the given problems.
        """
        queries = []
        for problem in problems:
            queries.append(self.build_query(problem["problem"]))
        return queries