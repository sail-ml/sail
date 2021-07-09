from generate_utils import generate, module_generator

generate.run("generate_utils/functions.yaml", "functions.h")
module_generator.run("generate_utils/modules.yaml", "py_module/module.h", "module_def.h")