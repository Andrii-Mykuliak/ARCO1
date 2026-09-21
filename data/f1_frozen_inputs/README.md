# f1_frozen_inputs

The two sentence-embedding realisations the Monte-Carlo and factorial diagnostic
experiments are defined on.

NOT DISTRIBUTED - the Data Availability statement ships aggregate results,
figures and tables, not embedding or projection arrays. `MANIFEST.json` is
distributed and records what each array is: SHA-256, shape, dtype, encoder and
which experiment consumes it.

Holders of a licensed copy of the highlight corpus can place the two arrays here
under the names in the manifest. `require_frozen_input` checks each against its
recorded hash before any experiment runs, so a re-encoding that does not
reproduce the realisation stops rather than silently running a different
experiment.

Without them `CONFIG = F1` stops at Stage 13, where the extended robustness
experiments are defined. Stages 1 to 12 do not depend on these arrays: the
canonical partition, the cross-register correspondence and the matched-event
comparison are reproduced without them.

Category labels are deliberately not packaged; see `MANIFEST.json`.
