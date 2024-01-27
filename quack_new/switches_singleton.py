"""
Optional switches for the program.
NOTE: This module is intended to be used as a singleton.
"""
from relations import NonEquivalenceRelationType

# Switches set to False in the ablation study.
propagate_attribute_accesses = True
propagate_stdlib_function_calls = True
propagate_user_defined_function_calls = True
shortcut_single_class_covering_all_attributes = True
handle_parameter_default_values = True
predict_type_parameters = True
propagate_instance_attribute_accesses = True

# Switches set to True in the ablation study.
parameters_only = False
return_values_only = False
simplified_type_ascription = False
