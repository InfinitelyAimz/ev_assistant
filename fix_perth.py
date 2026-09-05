import pathlib

target_path = pathlib.Path('venv311/Lib/site-packages/chatterbox/perth.py')
code_content = """class PerthImplicitWatermarker:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return None
"""

target_path.write_text(code_content)
print("Successfully dropped mock perth module into Chatterbox folder!")