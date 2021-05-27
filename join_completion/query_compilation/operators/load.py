from join_completion.query_compilation.operators.operator import Incomplete_Join_Operation


class LoadCompleteTable(Incomplete_Join_Operation):
    def __init__(self, table):
        Incomplete_Join_Operation.__init__(self)
        self.table = table

    def execute(self, current_join):
        assert current_join is None
        return self.table.incomplete_dataset

    def step_name(self):
        return f'L{self.table.name}'

    def __str__(self):
        return f'LoadCompleteTable({self.table.name})'
