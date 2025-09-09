Release checklist
- [ ] Check outstanding issues on JIRA and GitHub.
- [ ] Check [latest documentation](https://sequali.readthedocs.io/en/latest) 
      looks fine.
- [ ] Create a release branch.
  - [ ] Change current development version in `CHANGELOG.rst` to stable version.
- [ ] Check memory leaks with `tox -e asan`
- [ ] Merge the release branch into `main`.
- [ ] Created an annotated tag with the stable version number. Include changes 
from CHANGELOG.rst.
- [ ] Push tag to remote. This triggers the wheel/sdist build on GitHub CI.
- [ ] merge `main` branch back into `develop`.
- [ ] Build the new tag on readthedocs. Only build the last patch version of
each minor version. So `1.1.1` and `1.2.0` but not `1.1.0`, `1.1.1` and `1.2.0`.
- [ ] Create a new release on GitHub.
- [ ] Update the package on BioConda.
- [ ] Update [Galaxy wrapper](
      https://github.com/galaxyproject/tools-iuc/blob/main/tools/sequali/sequali.xml
  ). 
