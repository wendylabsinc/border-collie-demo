# Issue Tracker

Issues for this repository live in the **Wendy Labs Inc** Linear workspace, under the **Engineering** team and **Official Border Collie Demo** project.

## Workflow

- Use the Linear integration to find, create, and update issues.
- Create new issues under the Engineering team and Official Border Collie Demo project.
- Search for an existing issue before creating a duplicate.
- Preserve the issue's existing project, cycle, assignee, and priority unless the task requires changing them.
- Use the triage roles defined in `docs/agents/triage-labels.md` when classifying work.

## Wayfinding operations

- Represent a Wayfinder map as an issue labeled `wayfinder:map`.
- Create every decision ticket in the same team and project, with the map issue as its parent.
- Label each decision ticket with exactly one Wayfinder type: `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`.
- Use Linear's native `blocked by` and `blocks` issue relations for dependencies.
- Claim a ticket by assigning it before starting work. An open, unassigned child whose blockers are complete is on the frontier.
- Record a ticket's answer in a resolution comment, move it to the team's completed state, and append its linked one-line gist to the map's **Decisions so far** section.
