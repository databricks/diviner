We happily welcome contributions to the Diviner package. We use [GitHub Issues](/../../issues/) to track community reported issues and [GitHub Pull Requests ](/../../pulls/) for accepting changes.
Contributions are licensed on a license-in/license-out basis.

# Contributing Guide

## Communication
Before starting work on a major feature, please reach out to us via GitHub by filing an issue. We will make sure no one else is already working on it and ask you to open a GitHub issue.
A "major feature" is defined as any change that is > 100 LOC altered (not including tests), or changes any user-facing behavior, API signature, or introduces a new wrapped grouped model implementation.
We will use the GitHub issue to discuss the feature and come to an agreement on the design approach and help to answer any questions you may have on implementation details or designs.
The GitHub review process for major features is also important so that organizations with commit access can come to agreement on design.
If it is appropriate to write a design document, the document must be hosted either in the GitHub tracking issue, or linked to from the issue and hosted in a world-readable location.
Small patches, examples, documentation updates, and bug fixes don't need prior communication and can be submitted as a PR directly for review.

## Coding Style
We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with one exception: lines can be up to 100 characters in length, not 79.

## Code formatting
We use the Python formatting plugin [Black](https://black.readthedocs.io/en/stable/) to ensure that code formatting throughout this package is consistent. To ensure that any submissions are consistent, ensure that 
you have installed the required version in [dev-requirements](/dev-requirements.txt). Prior to committing changes to your local branch, simply run `black .` from the root of the repository. 
It is also strongly recommended to run pylint prior to commits to minimize the number of requested changes you may see in a pull request. 
To install all required code formatting tools, simply run `pip install -r "dev-requirements.txt"` from within your containerized development environment to ensure that you have the required versions of all tools.

## Sign your work
The sign-off is a simple line at the end of the explanation for the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify the below (from developercertificate.org):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

```
Signed-off-by: Joe Smith <joe.smith@email.com>
Use your real name (sorry, no pseudonyms or anonymous contributions.)
```

If you set your `user.name` and `user.email` git configs, you can sign your commit automatically with git commit -s.