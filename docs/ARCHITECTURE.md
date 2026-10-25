# Architecture

The product has frozen, fresh, and comparison authorities. Frozen artifacts are immutable evidence (the only post-freeze edits are listed in `provenance/POST_FREEZE_EDITS.csv`); a fresh run owns every newly computed parent; the comparison layer joins them without overwriting either authority.

