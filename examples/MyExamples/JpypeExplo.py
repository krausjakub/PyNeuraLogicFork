from neuralogic import is_initialized, initialize

if not is_initialized():
    initialize(debug_mode=True)


from neuralogic.nn import get_evaluator
from neuralogic.core import Template, R, Settings, Transformation, Combination, Aggregation
from neuralogic.core.constructs.function.tree import FunctionContainer
from neuralogic.core.settings import Settings
from neuralogic.dataset import Dataset
from neuralogic.optim import SGD

F = FunctionContainer()
train_dataset = Dataset()
template = Template()

train_dataset.add_example([R.b[3], R.c[5]])
train_dataset.add_queries([R.a[8]])


#template += (R.a <= (R.b, R.c)) | [Combination.AVG, Transformation.IDENTITY, Aggregation.AVG]
template += (R.a <= (x:=R.b, y:= R.c)) >> F.avg[F.identity(x + y)]

# expression above has following eval order:
# R.a    ||   x:= R.b    ||     y:= R.c    ||     R.a <= ()   ||    x + y   ||    F.identity  ||    >>   ||   template +=



settings = Settings(optimizer=SGD(), epochs=1)
neuralogic_evaluator = get_evaluator(template, settings)
build_dataset = neuralogic_evaluator.build_dataset(train_dataset)

for _ in neuralogic_evaluator.train(train_dataset):
    pass
