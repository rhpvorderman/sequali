Release checklist
- [ ] Check outstanding issues on JIRA and Github.
- [ ] Create a release branch.
  - [ ] Set version to a stable number.
  - [ ] Change current development version in `CHANGELOG.rst` to stable version.
  - [ ] Change the version in `__init__.py`
- [ ] Merge the release branch into `main`.
- [ ] Created an annotated tag with the stable version number. Include changes 
from CHANGELOG.rst.
- [ ] Push tag to remote. This triggers the wheel/sdist build on github CI.
- [ ] merge `main` branch back into `develop`.
- [ ] Build the new tag on readthedocs. Only build the last patch version of
each minor version. So `1.1.1` and `1.2.0` but not `1.1.0`, `1.1.1` and `1.2.0`.
- [ ] Create a new release on github.
- [ ] Update the package on bioconda.
