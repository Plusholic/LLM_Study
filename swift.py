import json

class PythonClass:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def to_json(self):
        return json.dumps(self.__dict__)
