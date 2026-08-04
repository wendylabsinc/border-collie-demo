# Whole-fruit curation policy

The demo targets intact physical fruit props. Public images and boxes are
training-eligible only after review against this policy.

## Accept

- a whole, intact, unpeeled banana;
- a whole apple or pear;
- multiple whole fruits only when each fruit has its own tight box; and
- partial occlusion only when the fruit's whole-object shape remains clear.

## Reject

- sliced, cut, peeled, mashed, cooked, juiced, or plated fruit;
- drawings, packaging, screens, logos, or other depictions;
- a single box around a dense bunch, pile, or mixed dish;
- boxes that include substantial non-fruit content; and
- ambiguous examples that cannot be confidently matched to a stage prop.

Rejected records may be retained as documented hard negatives when their
license permits it, but they must never carry a positive whole-fruit label.
The curation decision, reviewer or method, source image ID, and manifest hash
must be preserved with each training run.
