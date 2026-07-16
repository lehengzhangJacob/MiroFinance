# no-skill placeholder

EvolutionController.run_arm requires exactly one .md file in the skill
directory. The memory-only and plain ablation configs never construct a
SkillLibrary, so this file is never read or injected. Its sha256 in the arm
manifest only documents that the no-skill placeholder was mounted.
