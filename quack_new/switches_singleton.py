"""
Optional switches for the program.
NOTE: This module is intended to be used as a singleton.
"""
from relations import NonEquivalenceRelationType


propagate_attribute_accesses = True
propagate_stdlib_function_calls = True
propagate_user_defined_function_calls = True
shortcut_single_class_covering_all_attributes = True
valid_relations_to_induce_equivalent_relations = frozenset([
    NonEquivalenceRelationType.KeyOf,
    NonEquivalenceRelationType.ValueOf,
    NonEquivalenceRelationType.IterTargetOf,
    # NonEquivalenceRelationType.ArgumentOf,
    # NonEquivalenceRelationType.ReturnedValueOf,
    NonEquivalenceRelationType.AttrOf,
    NonEquivalenceRelationType.ElementOf,
    NonEquivalenceRelationType.SendTargetOf,
    NonEquivalenceRelationType.YieldFromAwaitResultOf
])
handle_parameter_default_values = True
propagate_instance_attribute_accesses = True
log_term_frequency = False
parameters_only = False
return_values_only = False
simplified_type_ascription = False
