"""Path permission check: only paths resolving under an allowed root may be accessed."""

import os


class PathPermissions:
    """Allowed-root set; validate() resolves symlinks and enforces containment.

    Each Uharness owns one instance, so concurrent harnesses cannot clobber
    each other's roots.
    """

    def __init__(self, roots=()):
        self.set_roots(roots)

    def set_roots(self, roots) -> None:
        """Replace the allowed-root list; each root is resolved (~ and symlinks included)."""
        self._roots = [os.path.realpath(os.path.expanduser(p)) for p in roots]

    def validate(self, path: str) -> str:
        """Resolve path (symlinks included) and check it is under an allowed root.

        Returns the resolved absolute path. Raises PermissionError otherwise.
        With no roots set, every path is denied.
        """
        resolved = os.path.realpath(os.path.expanduser(path))
        for root in self._roots:
            if resolved == root or resolved.startswith(root + os.sep):
                return resolved
        raise PermissionError(f"path {path} is outside allowed paths")


# module default for standalone tool use; Uharness keeps its own instance
_DEFAULT = PathPermissions()


def set_allowed_roots(paths) -> None:
    """Replace the module default's roots (standalone tool use)."""
    _DEFAULT.set_roots(paths)


def validate_path(path: str) -> str:
    """Validate against the module default; returns the resolved path."""
    return _DEFAULT.validate(path)
