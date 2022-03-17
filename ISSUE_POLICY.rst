Issue Policy
============

The Diviner Issue Policy outlines the categories of GitHub issues for this repository and discusses the guidelines & processes
associated with each type of issue.

Before filing an issue, make sure to `search for related issues <https://github.com/databricks/diviner/issues>`_ and check if
they address yours.

Issue Categories
----------------

Our policy is that GitHub issues fall into one of the following categories:

1. Feature Requests
2. Bug Reports
3. Documentation Fixes
4. Usage Issues

Each category has its own template, allowing us to adequately triage an issue or request without having to request
additional information from a reporter. Please fully fill out the template when submitting an issue and only use a
general request blank template if you are certain that your issue does not meet the guidelines described below.

Feature Requests
^^^^^^^^^^^^^^^^

Guidelines
##########

Feature requests that are likely to be accepted:

* Are non-breaking changes with previous versions (backwards compatible).
* Are extensible (having feature parity among supported forecasting frameworks).
* Provide value to end users that outweighs the increased complexity and maintenance burden for the additional functionality.
* Have adequate test coverage to ensure that the feature is functional.

Lifecycle
#########

Feature requests typically go through the following lifecycle:

#.  A **feature request** is submitted through a `GitHub Issue template <https://github.com/databricks/diviner/issues>`_,
    containing the high-level description of the proposed feature addition as well as the motivation behind its inclusion.
    It is helpful to include a general overview of the implementation of the proposed feature as well.

#.  The `issue is triaged <ISSUE_TRIAGE.rst>`_ to identify whether more information is needed from the author,
    give an indication of priority, and route feature requests to appropriate committers.

#. The feature request is discussed with a committer. The committer will provide input on the implementation overview or
   ask for a more detailed design, if applicable.

#. After discussion & agreement on the feature request and its implementation, an implementation owner is identified.

#. The implementation owner begins developing the feature and ultimately files associated pull requests against the
   Diviner Repository.

Bug Reports
^^^^^^^^^^^

Bug reports typically go through the following lifecycle:

#.  A **bug report** is submitted through an
    `Issue template <https://github.com/databricks/diviner/issues>`_, which contains a high-level description of
    the bug and information required to reproduce it.

#.  The `bug report is triaged <ISSUE_TRIAGE.rst>`_ to identify whether more information is needed from the author,
    give an indication of priority, and route to request appropriate committers.

#.  A Diviner committer reproduces the bug and provides feedback about how to implement a fix.

#.  After an approach has been agreed upon, an owner for the fix is identified. Diviner committers may choose to adopt
    ownership of severe bugs to ensure a timely fix.

#.  The fix owner begins implementing the fix and ultimately files associated pull requests.

Documentation Fixes
^^^^^^^^^^^^^^^^^^^

Documentation issues typically go through the following lifecycle:

#.  A **documentation fix** is submitted through a
    `Documentation Issue template <https://github.com/databricks/diviner/issues>`_, which contains a high-level description of
    the documentation error and the location (page) is provided with a sample of the error.

#.  The `documentation fix is triaged <ISSUE_TRIAGE.rst>`_ to identify whether more information is needed from the author,
    give an indication of priority, and route to request appropriate committers.

#.  A Diviner committer confirms the documentation issue and provides feedback on how to address the problem.

#.  After an approach has been agreed upon, an owner for the fix is identified. Diviner committers may choose to adopt
    ownership of glaring documentation errors to ensure a timely fix.

#.  The fix owner begins implementing the fix and ultimately files associated pull requests.

Usage Issues
^^^^^^^^^^^^

Usage issues typically go through the following lifecycle:

#.  A **usage issue** is submitted through a
    `Usage Issue template <https://github.com/databricks/diviner/issues>`_, which contains a high-level description of
    the question and a sample of the code being used. It is highly recommended to include the schema of the input data
    to aid in diagnosing issues. A sample of the data can be helpful as well.

    .. note::
        If you are submitting a usage issue and are including sample data, ensure that any sensitive data is removed or
        a corollary substitution is made so that sensitive data is not included with the usage issue report.

#.  The `usage issue is triaged <ISSUE_TRIAGE.rst>`_ to identify whether more information is needed from the author to
    allow for troubleshooting.

#.  A Diviner committer confirms the cause of the error and provides feedback on how to address the issue.

#.  If a bug is uncovered, the Diviner committer may file a bug report on behalf of the author and follow the process
    of triaging a bug report.

#.  If the issue is confirmed to be addressed by the author or if no response to the issue is made within 2 weeks, the
    issue will be closed.