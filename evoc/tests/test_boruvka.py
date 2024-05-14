import numpy as np
import pytest
from evoc.boruvka import (
    merge_components,
    update_component_vectors,
    boruvka_tree_query,
    parallel_boruvka,
)
