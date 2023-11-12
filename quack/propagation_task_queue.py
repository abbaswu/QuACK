import _ast
import ast
import logging
from collections import defaultdict, deque

from propagation_task_generation import PropagationTask


class PropagationTaskQueue:
    def __init__(self, maxsize=0):
        self.queue: deque[tuple[_ast.AST, PropagationTask]] = deque()
        self.nodes_to_enqueued_propagation_task_sets: defaultdict[_ast.AST, set[PropagationTask]] = defaultdict(set)

    def enqueue(self, node: _ast.AST, propagation_task: PropagationTask) -> None:
        if propagation_task not in self.nodes_to_enqueued_propagation_task_sets[node]:
            logging.info('Enqueue propagation task %s for node %s', propagation_task, ast.unparse(node))
            self.queue.append((node, propagation_task))
            self.nodes_to_enqueued_propagation_task_sets[node].add(propagation_task)
        else:
            logging.info('Propagation task %s for node %s already enqueued', propagation_task, node)

    def dequeue(self) -> tuple[_ast.AST, PropagationTask]:
        node, propagation_task = self.queue.popleft()
        return node, propagation_task

    def __bool__(self):
        return bool(self.queue)
