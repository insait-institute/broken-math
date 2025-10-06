from .api import APIQuery
from .parser import parse_answer, extract_answer, check_answers
from .configs import *
from .solveragent import SolverAgent

from copy import deepcopy
import os
import glob
import re
import json
from datetime import datetime
import random
from filelock import FileLock, Timeout
from loguru import logger

def solve(project_config : OverallConfig, solver_config : SolverConfig, 
          files_to_run : List[str], config_name : str, timeout: int = 30, batch_id: str = None):
    total_cost = 0

    config = load_config(os.path.join(project_config.model_config_folder, config_name + ".yaml"))

    if config.get('type', 'default') == "agent":
        agent_config_path = os.path.join("configs", config["agent_config"] + ".yaml")
        with open(agent_config_path, "r") as f:
            agent_config = yaml.safe_load(f)

        human_readable_id = config["human_readable_id"]
        del config["human_readable_id"]
        if "date" in config:
            del config["date"]
        model_config_path = os.path.join("configs", config["original_model_config"] + ".yaml")
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        del model_config["human_readable_id"]
        if "date" in model_config:
            del model_config["date"]
        if "concurrent_requests" in model_config:
            agent_config["n_threads"] = model_config["concurrent_requests"]
        
        agent = agent_config.pop("type")
        agent_params = {**model_config}
        if agent == "solver":
            cot_solver = SolverAgent(agent_params, **agent_config)
        else:
            raise ValueError(f"Unknown agent type: {agent}")
        for sol_index in range(solver_config.n_solutions):
            return single_agent_loop(cot_solver, solver_config, project_config, timeout, files_to_run, config_name, sol_index)
    
    del config["human_readable_id"]
    if "date" in config:
        del config["date"]

    logger.info(f"Initializing APIQuery for {config_name}...")
    querier = APIQuery(
        **config
    )
    try:
        for sol_index in range(solver_config.n_solutions):
            total_cost += single_loop(querier, solver_config, project_config, files_to_run, config_name, timeout, sol_index, config, batch_id)
    finally:
        if config.get("api") == "vllm":
            logger.info("Freeing vLLM model resources.")
            querier.free_model()
    return total_cost

def reject_result(result, cost, gold_answer=None):
    if result is None or len(result) == 0:
        return True
    if cost["output_tokens"] > 30000:
        # check if it is a power of 2
        if cost["output_tokens"] & (cost["output_tokens"] - 1) == 0:
            return True
        if cost["output_tokens"] % 1000 == 0:
            return True
    if gold_answer is not None:
        parsed_gold_answer = parse_answer(gold_answer)[0]
        extracted_answer = extract_answer(result)[0]
        if not check_answers(parsed_gold_answer, extracted_answer):
            return True
    return False

def single_loop(querier:APIQuery, solver_config : SolverConfig, project_config : OverallConfig,
                files_to_run : List[str], config_name : str, timeout: int = 30, 
                sol_index: int = 0, config: dict = None, batch_id: str = None):
    total_cost = 0

    for i in range(solver_config.n_attempts):
        problems_to_run = generate_problems_to_run(solver_config, project_config, files_to_run, config_name, sol_index, i, batch_id)
        
        if len(problems_to_run) == 0:
            if i == 0:
                logger.info(f"No problems to run for {config_name}")
            continue

        total_cost += run_problem(solver_config, project_config, config_name, timeout, i, problems_to_run, querier, batch_id)


    logger.info(f"Total cost for {config_name}: {total_cost}")
    return total_cost

def run_problem(solver_config, project_config, config_name, timeout, i, problems_to_run, querier, batch_id):
    total_cost = 0
    counter = 0

    for idx, result, cost in (querier.run_queries([p[1] for p in problems_to_run]) if batch_id is None else querier.retrieve_queries([p[1] for p in problems_to_run], batch_id)):
        counter += 1
        total_cost += cost["cost"]
        if counter % 10 == 0:
            logger.info(f"Processed {counter} problems with {config_name}, total cost: {total_cost}")
        full_result = result
        if "</think>" in result:
            think_result = result.split("</think>")[0]
            result = result.split("</think>")[-1]
        else:
            think_result = None
        problem_path = problems_to_run[idx][0]
        data        = json.load(open(problem_path, "r"))
        rejected = False
        grading = None
        if solver_config.can_reject and reject_result(result, cost, data.get("gold_answer")):
            logger.info(f"Rejected result for {problem_path} with {config_name}")
            if i == solver_config.n_attempts - 1:
                result = f"The model was unable to solve this problem. After {solver_config.n_attempts} attempts, each solution was either incorrect, hit the token limit (64000), or was empty."
                rejected = True
                grading = {
                        "details": [
                            {
                                "part_id": 1,
                                "score": 0,
                                "feedback": "The model was unable to solve this problem. Therefore, the solution is automatically graded as incorrect."
                            }
                        ],
                        "score": 0,
                        "uncertain": False,
                        "no_feedback": False,
                        "long": False,
                        "auto_grade": True,
                    }
            else:
                continue

        save_file(data, solver_config, project_config, config_name, timeout, i, full_result,
                  result, cost, problem_path, rejected, grading, think_result)
    return total_cost

def save_file(data, solver_config, project_config, config_name, timeout, i, full_result,
              result, cost, problem_path, rejected, grading, think_result=None):
    file_solved = problem_path.replace(project_config.unsolved_base_folder, project_config.solved_base_folder)
    file_lock_path = problem_path.replace(project_config.unsolved_base_folder, project_config.lock_base_folder)
    os.makedirs(os.path.dirname(file_solved), exist_ok=True)
    os.makedirs(os.path.dirname(file_lock_path), exist_ok=True)

            # --------- begin critical section ---------
    lock = FileLock(file_lock_path + ".lock", unlink_on_release=True)      # one lock per JSON file
    try:
        with lock.acquire(timeout=timeout):
                    # 1) read the current contents (if any)
            if (os.path.exists(file_solved) and (not solver_config.overwrite or i > 0)):
                with open(file_solved, "r") as fh:
                    data = json.load(fh)
            else:
                data[solver_config.attempt_key] = []

                    # 2) update in memory
            data[solver_config.attempt_key].append(
                        {
                            "model_id":  config_name,
                            "solution":  result,
                            "full_solution": full_result,
                            "thinking": think_result,
                            "cost":      cost,
                            "grading":   grading,
                            "rejected": rejected,
                            "annotations": None,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
            if solver_config.allow_shuffle:
                random.shuffle(data[solver_config.attempt_key])
                    # put all rejected attempts at the end
            data[solver_config.attempt_key] = sorted(data[solver_config.attempt_key], key=lambda x: x.get("rejected", False))

                    # 3) **atomic write**: write to a tmp file, then replace
            tmp = file_solved + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=4)
            os.replace(tmp, file_solved)        # atomic on POSIX & Windows
    except Timeout:
        logger.warning(f"Could not obtain lock on {file_solved} within {timeout}s")

def generate_problems_to_run(solver_config, project_config, files_to_run, config_name, sol_index, i, batch_id):
    problems_to_run = []
    for file in files_to_run:
        data = json.load(open(file, "r"))
        file_solved = file.replace(project_config.unsolved_base_folder, project_config.solved_base_folder)
        if os.path.exists(file_solved) and (not solver_config.overwrite or i > 0):
            data_solved = json.load(open(file_solved, "r"))
            n_already_solved = sum(
                    1 for attempt in data_solved[solver_config.attempt_key] if attempt["model_id"] == config_name and len(attempt['solution'])>0
                )
            if n_already_solved > sol_index or batch_id is not None:
                if i == 0:
                    logger.info(f"Already solved {file} with {config_name}")
                continue

        conversation = deepcopy(solver_config.prior_conversation)

        conversation += [
            {
                "role": "user",
                "content": solver_config.prompt_gold if data.get("gold_answer") else solver_config.prompt
            }
        ]

        for interaction in conversation:
            interaction['content'] = interaction['content'].format(**data)
             
        problems_to_run.append((file, conversation)) 

    return problems_to_run

def single_agent_loop(cot_solver, solver_config, project_config, timeout, files_to_run, config_name, sol_index):

    total_cost = 0

    for i in range(solver_config.n_attempts):
        problems_to_run = generate_problems_to_run(solver_config, project_config, files_to_run, config_name, sol_index, i, None)
        
        if len(problems_to_run) == 0:
            if i == 0:
                logger.info(f"No problems to run for {config_name}")
            continue
        counter = 0

        for idx, messages, cost, _ in cot_solver.solve([p[1][-1]['content'] for p in problems_to_run]):
          
            total_cost += cost['cost']

            if counter % 10 == 0:
                logger.info(f"Processed {counter} problems with {config_name}, total cost: {total_cost}")
            full_result = messages
            result = messages[-1]['content']
            if "</think>" in result:
                think_result = result.split("</think>")[0]
                result = result.split("</think>")[-1]
            else:
                think_result = None
            problem_path = problems_to_run[idx][0]
            data        = json.load(open(problem_path, "r"))
            rejected = False
            grading = None

            save_file(data, solver_config, project_config, config_name, timeout, i, full_result,
                  result, cost, problem_path, rejected, grading, think_result)

    logger.info(f"Total cost for {config_name}: {total_cost}")   

    return total_cost