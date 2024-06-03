__version__ = '0.0.1'

from .suffix_manager import (
    AttackPrompt,
    PromptManager,
    MultiPromptAttack,
    ProgressiveMultiPromptAttack,
    get_embedding_layer,
    get_embedding_matrix,
    get_embeddings,
    get_nonascii_toks,
    get_goals_and_targets,
    get_workers
)

from .gcg import GCGAttackPrompt as AttackPrompt
from .gcg import GCGPromptManager as PromptManager
from .gcg import GCGMultiPromptAttack as MultiPromptAttack
