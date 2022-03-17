## What changes are proposed in this pull request?

(Please fill in changes proposed in this fix)

## How is this patch tested?

(Details)

## Does this PR change the documentation?

- [ ] No. You can skip the rest of this section.
- [ ] Yes. Make sure the changed pages / sections render correctly by following the steps below.

1. Navigate to diviner/docs via the command line.
2. Execute `make clean` from the docs directory.
3. Execute `make html` from the docs directory.
4. Navigate to the `diviner/docs/_build/html/` directory.
5. Open `index.html` in your browser and validate that your changes render correctly.

## Release Notes

### Is this a user-facing change?

- [ ] No. You can skip the rest of this section.
- [ ] Yes. Give a description of this change to be included in the release notes for Diviner users.

(Details in 1-2 sentences. You can just refer to another PR with a description if this PR is part of a larger change.)

### What component(s) does this PR affect?

Components 
- [ ] `area/models`: Forecasting framework interfaces and associated training and prediction code for Diviner
- [ ] `area/analysis`: Ad-hoc analytic EDA functionality (i.e., trend decomposition)
- [ ] `area/data`: Data structure transformation code
- [ ] `area/examples`: Example code
- [ ] `area/serialization`: Model serialization and deserialization
- [ ] `area/utilities`: Common or per-framework utility code
- [ ] `area/scoring`: Cross validation, back-testing, and associated scoring functionality
- [ ] `area/docs`: Diviner documentation pages

<!--
Insert an empty named anchor here to allow jumping to this section with a fragment URL
(e.g. https://github.com/mlflow/mlflow/pull/123#user-content-release-note-category).
Note that GitHub prefixes anchor names in markdown with "user-content-".
-->
<a name="release-note-category"></a>
### How should the PR be classified in the release notes? Choose one:

- [ ] `rn/breaking-change` - The PR will be mentioned in the "Breaking Changes" section
- [ ] `rn/none` - No description will be included. The PR will be mentioned only by the PR number in the "Small Bugfixes and Documentation Updates" section
- [ ] `rn/feature` - A new user-facing feature worth mentioning in the release notes
- [ ] `rn/bug-fix` - A user-facing bug fix worth mentioning in the release notes
- [ ] `rn/documentation` - A user-facing documentation change worth mentioning in the release notes
