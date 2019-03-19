# Making a release

Releases are deployed by Travis to peterkroon's PyPI, but only for tagged
commits. This requires "build on push" is enabled on Travis, otherwise those
don't get built at all. However, since we dev on branches in the main repo we
only want Travis to build for master, and anything that looks like a SemVer
version tag ('v1.2.3\[-prerelease\]\[+metadata\]'). Hence the ugly regex in the
Travis yaml. (Note: PBR doesn't seem to be able to deal with all prerelease and
metadata options allowed by SemVer)

To make a release:
- Implement your awesome feature, including docs.
- Make a PR of your feature, get it reviewed and make sure Travis passes.
- Get your feature merged in master
- Create a tag: `git tag v0.5.0 -a`. Come up with a sane version number.
- Push the tag: `git push --tags`.
- Profit.

Note that Travis will build releases *twice*. Once because something is pushed
to master, and once because it's a tagged commit. But since we don't release
too often this is OK. Also, if this is to change Travis should first come up
with way better documentation.

