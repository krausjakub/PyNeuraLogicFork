import jpype
from neuralogic import is_initialized, initialize

if not is_initialized():
    initialize(debug_mode=True)

class_reference = jpype.JClass("cz.cvut.fel.ida.algebra.functions.MyClass")


class_instance = class_reference()

pozdrav = class_instance.pozdrav()

print(pozdrav)